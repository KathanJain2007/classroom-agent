"""
agent3.py — Agent 3: The Examiner.

Reads outputs/questions.json produced by Agent 2, evaluates and assigns calibrated
cognitive difficulty (EASY, MEDIUM, HARD), checks and reports paper balance
against a DifficultyTarget without falsifying grades, allocates marks based on
a deterministic (type, difficulty) matrix, generates itemized marking schemes,
and formats the final assessment into student-facing question papers and
examiner-facing marking guides.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Optional, Union

from pydantic import ValidationError

from schemas import (
    AnyQuestion,
    CodeQuestion,
    Concept,
    Difficulty,
    DifficultyTarget,
    MCQQuestion,
    QuestionType,
    Quiz,
    TheoryQuestion,
)
from step1_test import (
    MODEL_NAME,
    MODEL_PRIORITY,
    call_llm_with_retry,
    get_llm_client,
    parse_json_response,
    select_active_model,
)

INPUT_QUESTIONS_PATH = "outputs/questions.json"
INPUT_CONCEPTS_PATH = "outputs/concepts.json"
FINAL_QUIZ_PATH = "outputs/final_quiz.json"
MARKING_SCHEME_JSON_PATH = "outputs/marking_scheme.json"
QUESTION_PAPER_TXT_PATH = "outputs/question_paper.txt"
MARKING_SCHEME_TXT_PATH = "outputs/marking_scheme.txt"

INTER_CALL_DELAY_SECONDS = 2.0

# Global LLM call counter
llm_call_count = 0


def tracked_call_llm(
    client: object,
    messages: list[dict],
    model: str,
    temperature: float = 0.1,
    response_format: Optional[dict] = None,
) -> Optional[str]:
    """Wraps call_llm_with_retry to track total LLM calls in Agent 3."""
    global llm_call_count
    llm_call_count += 1
    return call_llm_with_retry(
        client=client,
        messages=messages,
        model=model,
        temperature=temperature,
        response_format=response_format,
    )


def convert_option_indices_to_letters(explanation: str) -> str:
    """
    Deterministically converts 0-indexed option references in explanation text to letters
    (0 -> A, 1 -> B, 2 -> C, 3 -> D).
    Examples:
      'making option 0 correct' -> 'making Option A correct'
      'Option 1 is a common mix-up' -> 'Option B is a common mix-up'
      'choice 2' -> 'choice C'
      'options 0 and 1' -> 'options A and B'
      'option (0)' -> 'option (A)'
    """
    if not explanation:
        return explanation

    idx_map = {"0": "A", "1": "B", "2": "C", "3": "D"}

    def repl_multi(m: re.Match) -> str:
        prefix = m.group(1)
        first = idx_map.get(m.group(2), m.group(2))
        conj = m.group(3)
        second = idx_map.get(m.group(4), m.group(4))
        return f"{prefix}{first}{conj}{second}"

    text = re.sub(
        r"\b(options?\s+|choices?\s+)([0-3])(\s*(?:,|and|or)\s*)([0-3])\b",
        repl_multi,
        explanation,
        flags=re.IGNORECASE,
    )

    def repl_single(m: re.Match) -> str:
        prefix = m.group(1)
        lead_bracket = m.group(2) or ""
        digit = m.group(3)
        trail_bracket = m.group(4) or ""
        letter = idx_map.get(digit, digit)
        return f"{prefix}{lead_bracket}{letter}{trail_bracket}"

    text = re.sub(
        r"\b(options?\s+|choices?\s*)(\(?\[?)([0-3])(\)?\]?)\b",
        repl_single,
        text,
        flags=re.IGNORECASE,
    )

    return text


def check_breakdown_criteria_alignment(q: AnyQuestion, alloc: list[dict]) -> bool:
    """
    Checks that each breakdown's criteria mention terms appearing in that question's own text.
    Warns on an obvious mismatch (e.g. deque question having criteria mentioning append/realloc).
    """
    doc_text = (
        f"{q.question_text} {q.concept} "
        f"{getattr(q, 'expected_answer', '')} "
        f"{getattr(q, 'code_snippet', '')} "
        f"{getattr(q, 'model_answer', '')} "
        f"{' '.join(getattr(q, 'key_points', []))}"
    ).lower()

    criteria_text = " ".join(item.get("criterion", "") for item in alloc).lower()
    # Extract substantive words (length >= 4, ignoring generic meta words)
    words = [
        w for w in re.findall(r"\b[a-z]{4,}\b", criteria_text)
        if w not in {
            "this", "that", "with", "from", "using", "correct", "implementation",
            "marks", "points", "proper", "handling", "usage", "quality", "syntax",
            "clear", "provides", "explains", "describes", "structure", "structures",
            "detail", "detailed", "different", "differences", "demonstrates",
            "function", "logic", "cases", "boundary", "management", "pointer",
            "order", "given", "concept", "concepts", "answer", "code", "programming"
        }
    ]

    matches = [w for w in words if w in doc_text]
    if words and not matches:
        return False
    return True

# ---------------------------------------------------------------------------
# Part C: Deterministic Marks Allocation Table
# ---------------------------------------------------------------------------
# Derived strictly from question type and assessed cognitive difficulty.
# Maintained as a module-level constant for easy adjustment by curriculum heads.
# ---------------------------------------------------------------------------
MARKS_TABLE: dict[tuple[QuestionType, Difficulty], int] = {
    (QuestionType.MCQ, Difficulty.EASY): 1,
    (QuestionType.MCQ, Difficulty.MEDIUM): 2,
    (QuestionType.MCQ, Difficulty.HARD): 3,
    (QuestionType.CODE, Difficulty.EASY): 4,
    (QuestionType.CODE, Difficulty.MEDIUM): 6,
    (QuestionType.CODE, Difficulty.HARD): 8,
    (QuestionType.THEORY, Difficulty.EASY): 4,
    (QuestionType.THEORY, Difficulty.MEDIUM): 6,
    (QuestionType.THEORY, Difficulty.HARD): 10,
}


# ---------------------------------------------------------------------------
# Part A: Difficulty Assignment
# ---------------------------------------------------------------------------
# ARCHITECTURAL NOTE:
# The function assign_difficulty() is deliberately isolated so its LLM-based
# implementation can later be swapped for a trained classifier (e.g. an ML model
# trained on historical university exam papers) without modifying the rest of Agent 3.
# This function defines the exact seam where that model plugs in.
# ---------------------------------------------------------------------------

_DIFFICULTY_CACHE: dict[str, tuple[Difficulty, str]] = {}


def assign_difficulty(question: AnyQuestion) -> tuple[Difficulty, str]:
    """
    Returns (difficulty, reasoning) for a given question.

    Deliberately isolated so this function can be swapped for a trained
    ML difficulty-prediction classifier without touching the rest of Agent 3.
    """
    if question.id in _DIFFICULTY_CACHE:
        return _DIFFICULTY_CACHE[question.id]

    # Fallback default if called outside the batch workflow
    return Difficulty.MEDIUM, "Standard application question requiring operational understanding."


def batch_assess_difficulties(
    questions: list[AnyQuestion],
    client: object,
    model: str = MODEL_NAME,
) -> dict[str, tuple[Difficulty, str]]:
    """
    Assesses cognitive difficulty across all questions in a single batched LLM call.
    Uses strict pedagogical criteria to categorize as EASY, MEDIUM, or HARD.
    Judges the question as written, not the underlying concept.
    """
    items_payload = []
    for q in questions:
        item = {
            "id": q.id,
            "type": q.type.value,
            "concept": q.concept,
            "question_text": q.question_text,
        }
        if isinstance(q, MCQQuestion):
            item["options"] = q.options
        elif isinstance(q, CodeQuestion):
            item["language"] = q.language
            if q.code_snippet:
                item["code_snippet"] = q.code_snippet[:250] + ("..." if len(q.code_snippet) > 250 else "")
        items_payload.append(item)

    system_prompt = (
        "You are an expert university chief examiner evaluating the cognitive difficulty "
        "of assessment questions for an undergraduate Computer Science course."
    )

    user_prompt = f"""Evaluate the cognitive difficulty of each of the following {len(questions)} questions.

DIFFICULTY RUBRIC:
- EASY: Recall or recognition of a single fact, definition, or syntax rule; one-step direct reasoning.
- MEDIUM: Applying a concept to a scenario, tracing a short sequence of operations, comparing two mechanisms, or solving a routine problem.
- HARD: Multi-step reasoning, identifying a subtle bug, synthesizing several concepts, designing an algorithm, or analyzing complex trade-offs.

CRITICAL INSTRUCTION:
Judge each question strictly AS WRITTEN, not merely by the topic name. An advanced topic with a simple definitional question is EASY; an introductory topic with a subtle tracing problem is HARD.

QUESTIONS TO EVALUATE:
{json.dumps(items_payload, indent=2)}

REQUESTED OUTPUT SHAPE (Return valid JSON):
{{
  "assessments": [
    {{
      "id": "<question id>",
      "difficulty": "EASY" | "MEDIUM" | "HARD",
      "reasoning": "<1-2 sentence justification citing specific features of the question>"
    }}
  ]
}}
"""

    raw_response = tracked_call_llm(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    if not raw_response:
        time.sleep(INTER_CALL_DELAY_SECONDS)
        raw_response = tracked_call_llm(
            client=client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.1,
            response_format=None,
        )

    parsed = parse_json_response(raw_response) if raw_response else None
    results: dict[str, tuple[Difficulty, str]] = {}

    if parsed and "assessments" in parsed:
        for entry in parsed["assessments"]:
            q_id = entry.get("id")
            diff_str = str(entry.get("difficulty", "MEDIUM")).upper()
            reason = entry.get("reasoning", "Assessed by cognitive rubric.")

            diff_enum = Difficulty.MEDIUM
            if diff_str in ("EASY", "MEDIUM", "HARD"):
                diff_enum = Difficulty[diff_str]

            results[q_id] = (diff_enum, reason)

    # Ensure every question has an entry
    for q in questions:
        if q.id not in results:
            results[q.id] = (Difficulty.MEDIUM, "Assigned fallback difficulty.")

    # Populate cache for assign_difficulty()
    global _DIFFICULTY_CACHE
    _DIFFICULTY_CACHE = results
    return results


# ---------------------------------------------------------------------------
# Part B: Paper Balance Evaluation
# ---------------------------------------------------------------------------

def evaluate_paper_balance(
    questions: list[AnyQuestion],
    target: DifficultyTarget,
) -> tuple[str, dict[str, int], dict[str, int]]:
    """
    Compares the actual difficulty distribution against the DifficultyTarget.
    Reports mismatches honestly without falsifying question difficulty.

    Why honest reporting is required:
    An examiner relabelling a genuinely hard question as easy to meet a statistical
    quota would mislead the instructor and students. Authentic reporting ensures
    the professor sees exactly where shortfalls exist in the question pool.
    """
    total = len(questions)
    target_easy = round(total * target.easy_ratio)
    target_medium = round(total * target.medium_ratio)
    target_hard = total - target_easy - target_medium

    target_counts = {"EASY": target_easy, "MEDIUM": target_medium, "HARD": target_hard}

    actual_counts = {
        "EASY": sum(1 for q in questions if q.difficulty == Difficulty.EASY),
        "MEDIUM": sum(1 for q in questions if q.difficulty == Difficulty.MEDIUM),
        "HARD": sum(1 for q in questions if q.difficulty == Difficulty.HARD),
    }

    shortfalls: list[str] = []
    if actual_counts["EASY"] < target_easy:
        shortfalls.append(f"{target_easy - actual_counts['EASY']} easy")
    if actual_counts["MEDIUM"] < target_medium:
        shortfalls.append(f"{target_medium - actual_counts['MEDIUM']} medium")
    if actual_counts["HARD"] < target_hard:
        shortfalls.append(f"{target_hard - actual_counts['HARD']} hard")

    shortfall_text = f"Shortfall: {', '.join(shortfalls)} question(s)." if shortfalls else "Distribution matches target."
    balance_note = (
        f"Target: {target_easy} easy / {target_medium} medium / {target_hard} hard. "
        f"Actual: {actual_counts['EASY']} easy / {actual_counts['MEDIUM']} medium / {actual_counts['HARD']} hard. "
        f"{shortfall_text}"
    )

    return balance_note, target_counts, actual_counts


# ---------------------------------------------------------------------------
# Part D: Marking Scheme Generation (Batched LLM Call with Fallback Regeneration)
# ---------------------------------------------------------------------------

def generate_single_mark_breakdown(
    q: Union[CodeQuestion, TheoryQuestion],
    client: object,
    model: str,
) -> list[dict]:
    """
    Targeted regeneration for a single question whose batched mark breakdown criteria mismatched.
    Explicitly includes the full question text, concept, code snippet/model answer, and total marks.
    """
    item = {
        "id": q.id,
        "type": q.type.value,
        "concept": q.concept,
        "total_marks": q.marks,
        "question_text": q.question_text,
    }
    if isinstance(q, CodeQuestion):
        item["expected_answer"] = q.expected_answer
        if q.code_snippet:
            item["code_snippet"] = q.code_snippet[:300]
    elif isinstance(q, TheoryQuestion):
        item["model_answer"] = q.model_answer
        item["key_points"] = q.key_points

    system_prompt = (
        "You are an expert university examiner creating an itemized mark breakdown specifically "
        "tailored to the given question."
    )
    user_prompt = f"""Create an itemized mark breakdown for the following question. The criteria MUST be specific to this exact question and its topic.

QUESTION:
{json.dumps(item, indent=2)}

INSTRUCTIONS:
- The sum of marks in 'allocation' MUST EXACTLY equal {q.marks}.
- Every criterion must explicitly refer to the concepts and mechanisms present in this specific question.

REQUESTED OUTPUT SHAPE (Return valid JSON):
{{
  "allocation": [
    {{
      "criterion": "<Specific criterion for this question>",
      "marks": <integer points awarded>
    }}
  ]
}}
"""
    raw = tracked_call_llm(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    if not raw:
        time.sleep(INTER_CALL_DELAY_SECONDS)
        raw = tracked_call_llm(
            client=client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.1,
            response_format=None,
        )

    parsed = parse_json_response(raw) if raw else None
    if parsed and "allocation" in parsed and isinstance(parsed["allocation"], list) and parsed["allocation"]:
        alloc = parsed["allocation"]
        current_sum = sum(int(a.get("marks", 0)) for a in alloc)
        diff = q.marks - current_sum
        if diff != 0 and alloc:
            alloc[-1]["marks"] = max(1, int(alloc[-1].get("marks", 1)) + diff)
        return alloc
    return []


def batch_generate_mark_breakdowns(
    questions: list[Union[CodeQuestion, TheoryQuestion]],
    client: object,
    model: str = MODEL_NAME,
) -> tuple[dict[str, list[dict]], list[tuple[str, str]]]:
    """
    Generates itemized criteria-level mark breakdowns for CODE and THEORY questions.
    Executed in a single batched LLM call for all non-MCQ questions.
    Validates that allocations sum precisely to each question's allocated marks.
    If criteria do not align with the question text, attempts single-question regeneration once.
    If mismatch persists, inserts placeholder [MARK BREAKDOWN NEEDS MANUAL REVIEW — generated criteria did not match this question].
    """
    if not questions:
        return {}, []

    items_payload = []
    for q in questions:
        item = {
            "id": q.id,
            "type": q.type.value,
            "concept": q.concept,
            "total_marks": q.marks,
            "question_text": q.question_text,
        }
        if isinstance(q, CodeQuestion):
            item["expected_answer"] = q.expected_answer
        elif isinstance(q, TheoryQuestion):
            item["model_answer"] = q.model_answer
            item["key_points"] = q.key_points
        items_payload.append(item)

    system_prompt = (
        "You are an expert university examiner creating an official grading rubric and "
        "marking scheme for a university examination."
    )

    user_prompt = f"""Create an itemized mark breakdown for each of the following {len(questions)} questions.

INSTRUCTIONS:
- For CODE questions: split the total marks across:
  1) Core correctness and algorithmic logic
  2) Handling edge cases and boundary conditions
  3) Code quality, syntax, and proper pointer/memory management
- For THEORY questions: split the total marks across the provided key marking points and concepts.
- CRITICAL REQUIREMENT: For each question, the sum of marks in the 'allocation' list MUST EXACTLY EQUAL its 'total_marks'.

QUESTIONS:
{json.dumps(items_payload, indent=2)}

REQUESTED OUTPUT SHAPE (Return valid JSON):
{{
  "breakdowns": [
    {{
      "id": "<question id>",
      "allocation": [
        {{
          "criterion": "<Specific, concrete criterion description>",
          "marks": <integer points awarded>
        }}
      ]
    }}
  ]
}}
"""

    raw = tracked_call_llm(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    if not raw:
        time.sleep(INTER_CALL_DELAY_SECONDS)
        raw = tracked_call_llm(
            client=client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.1,
            response_format=None,
        )

    parsed = parse_json_response(raw) if raw else None
    breakdowns: dict[str, list[dict]] = {}
    mismatch_reports: list[tuple[str, str]] = []

    if parsed and "breakdowns" in parsed:
        for entry in parsed["breakdowns"]:
            q_id = entry.get("id")
            alloc = entry.get("allocation", [])
            if isinstance(alloc, list) and alloc:
                breakdowns[q_id] = alloc

    # Validate and normalize allocations against question.marks
    for q in questions:
        alloc = breakdowns.get(q.id)
        if not alloc:
            # Fallback allocation if LLM call failed for this question
            if isinstance(q, CodeQuestion):
                half = q.marks // 2
                rem = q.marks - half
                alloc = [
                    {"criterion": "Core algorithmic correctness and bug identification", "marks": half},
                    {"criterion": "Edge cases, memory management, and syntax correctness", "marks": rem},
                ]
            else:
                pts = q.key_points if q.key_points else ["Demonstrates thorough conceptual understanding"]
                per_pt = max(1, q.marks // len(pts))
                alloc = [{"criterion": pt, "marks": per_pt} for pt in pts]
                alloc_sum = sum(a["marks"] for a in alloc)
                if alloc_sum != q.marks:
                    alloc[-1]["marks"] += (q.marks - alloc_sum)

        # Ensure exact sum matching
        current_sum = sum(int(a.get("marks", 0)) for a in alloc)
        diff = q.marks - current_sum
        if diff != 0:
            print(
                f"  [Warning] Mark allocation for {q.id} summed to {current_sum}m instead of {q.marks}m. "
                f"Adjusting last criterion by {diff:+d}m."
            )
            alloc[-1]["marks"] = max(1, int(alloc[-1].get("marks", 1)) + diff)

        # Assert that each breakdown's criteria mention terms appearing in that question's own text
        if not check_breakdown_criteria_alignment(q, alloc):
            print(
                f"  [Warning] Mark breakdown for {q.id} ('{q.concept}') does not align with question text!\n"
                f"            Initial Criteria: {[c.get('criterion') for c in alloc]}"
            )
            print(f"  Attempting targeted single-question regeneration for {q.id}...")
            regen_alloc = generate_single_mark_breakdown(q, client, model)
            if regen_alloc and check_breakdown_criteria_alignment(q, regen_alloc):
                print(f"  -> Successfully regenerated aligned mark breakdown for {q.id}.")
                alloc = regen_alloc
                mismatch_reports.append((q.id, "Resolved via targeted single-question regeneration"))
            else:
                print(f"  -> Regeneration still mismatched for {q.id}. Inserting manual review placeholder.")
                placeholder = "[MARK BREAKDOWN NEEDS MANUAL REVIEW — generated criteria did not match this question]"
                alloc = [{"criterion": placeholder, "marks": q.marks}]
                q.verification_status = "needs_review"
                note_suffix = "Mark breakdown criteria mismatch"
                q.verification_note = f"{q.verification_note} | {note_suffix}" if q.verification_note else note_suffix
                mismatch_reports.append((q.id, "Persistent mismatch — flagged for manual review with placeholder"))

        breakdowns[q.id] = alloc

    return breakdowns, mismatch_reports


# ---------------------------------------------------------------------------
# Part E: Paper Ordering & Document Formatting
# ---------------------------------------------------------------------------

def order_questions_for_paper(questions: list[AnyQuestion]) -> list[AnyQuestion]:
    """
    Orders questions for the final exam paper:
    1. All MCQs first (ordered by difficulty: EASY -> MEDIUM -> HARD)
    2. All CODE questions next (ordered by difficulty: EASY -> MEDIUM -> HARD)
    3. All THEORY questions last (ordered by difficulty: EASY -> MEDIUM -> HARD)
    """
    diff_order = {Difficulty.EASY: 0, Difficulty.MEDIUM: 1, Difficulty.HARD: 2}

    mcqs = [q for q in questions if q.type == QuestionType.MCQ]
    codes = [q for q in questions if q.type == QuestionType.CODE]
    theories = [q for q in questions if q.type == QuestionType.THEORY]

    mcqs.sort(key=lambda q: (diff_order.get(q.difficulty, 1), q.id))
    codes.sort(key=lambda q: (diff_order.get(q.difficulty, 1), q.id))
    theories.sort(key=lambda q: (diff_order.get(q.difficulty, 1), q.id))

    return mcqs + codes + theories


def format_question_paper_text(
    quiz: Quiz,
    ordered_questions: list[AnyQuestion],
) -> str:
    """
    Generates student-facing exam paper text.
    Strictly excludes all answers, explanations, and verification notes.
    """
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("UNIVERSITY EXAMINATIONS BOARD")
    lines.append("DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING")
    lines.append("ASSESSMENT: DATA STRUCTURES & ALGORITHMS")
    lines.append("=" * 80)
    lines.append(f"Total Marks: {quiz.total_marks:<4} | Time Allowed: 2 Hours | Total Questions: {len(ordered_questions)}")
    source_docs = ", ".join(quiz.source_documents)
    lines.append(f"Syllabus Source Documents: {source_docs}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("INSTRUCTIONS TO CANDIDATES:")
    lines.append("1. Answer ALL questions across all sections.")
    lines.append("2. Read each question carefully before attempting.")
    lines.append("3. For Multiple Choice Questions, write the letter of the single best option.")
    lines.append("4. For Programming Questions, write syntactically correct and safe code.")
    lines.append("")

    mcqs = [q for q in ordered_questions if q.type == QuestionType.MCQ]
    codes = [q for q in ordered_questions if q.type == QuestionType.CODE]
    theories = [q for q in ordered_questions if q.type == QuestionType.THEORY]

    mcq_marks = sum(q.marks for q in mcqs)
    code_marks = sum(q.marks for q in codes)
    theory_marks = sum(q.marks for q in theories)

    lines.append("SECTION BREAKDOWN:")
    lines.append(f"  - Section A: Multiple Choice Questions (Questions 1–{len(mcqs)}, {mcq_marks} Marks)")
    lines.append(f"  - Section B: Code Analysis & Implementation (Questions {len(mcqs)+1}–{len(mcqs)+len(codes)}, {code_marks} Marks)")
    lines.append(f"  - Section C: Theoretical Design & Comparative Analysis (Questions {len(mcqs)+len(codes)+1}–{len(ordered_questions)}, {theory_marks} Marks)")
    lines.append("=" * 80)
    lines.append("")

    q_num = 1

    # Section A
    if mcqs:
        lines.append("=" * 80)
        lines.append(f"SECTION A: MULTIPLE CHOICE QUESTIONS ({mcq_marks} Marks)")
        lines.append("=" * 80)
        for q in mcqs:
            lines.append(f"\nQuestion {q_num}. [{q.marks} Mark{'s' if q.marks > 1 else ''}]")
            lines.append(f"{q.question_text}")
            opt_letters = ["A", "B", "C", "D"]
            for i, opt in enumerate(q.options):
                lbl = opt_letters[i] if i < len(opt_letters) else str(i)
                lines.append(f"   {lbl}) {opt}")
            q_num += 1
        lines.append("")

    # Section B
    if codes:
        lines.append("=" * 80)
        lines.append(f"SECTION B: CODE ANALYSIS & IMPLEMENTATION ({code_marks} Marks)")
        lines.append("=" * 80)
        for q in codes:
            lines.append(f"\nQuestion {q_num}. [{q.marks} Marks]")
            lines.append(f"Language: {q.language}")
            lines.append(f"{q.question_text}")
            if q.code_snippet:
                lines.append("\nCode Snippet:")
                for s_line in q.code_snippet.splitlines():
                    lines.append(f"    {s_line}")
            q_num += 1
        lines.append("")

    # Section C
    if theories:
        lines.append("=" * 80)
        lines.append(f"SECTION C: THEORETICAL DESIGN & ANALYSIS ({theory_marks} Marks)")
        lines.append("=" * 80)
        for q in theories:
            lines.append(f"\nQuestion {q_num}. [{q.marks} Marks]")
            lines.append(f"{q.question_text}")
            q_num += 1
        lines.append("")

    lines.append("=" * 80)
    lines.append("[END OF QUESTION PAPER]")
    lines.append("=" * 80)

    return "\n".join(lines)


def format_marking_scheme_text(
    quiz: Quiz,
    ordered_questions: list[AnyQuestion],
    breakdowns: dict[str, list[dict]],
) -> str:
    """
    Generates examiner-facing marking scheme document containing solutions,
    mark breakdown tables, rationale, and verification status.
    """
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("EXAMINER'S OFFICIAL MARKING SCHEME & SOLUTION GUIDE")
    lines.append("DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING")
    lines.append("=" * 80)
    lines.append(f"Total Examination Marks: {quiz.total_marks:<4} | Total Questions: {len(ordered_questions)}")
    lines.append(f"Cognitive Balance Summary: {quiz.balance_note}")
    lines.append("=" * 80)
    lines.append("")

    opt_letters = ["A", "B", "C", "D"]

    for idx, q in enumerate(ordered_questions, start=1):
        v_tag = q.verification_status.upper()
        if q.verification_status == "verified":
            v_str = f"VERIFIED ({q.verification_note or 'Passed independent check'})"
        elif q.verification_status == "needs_review":
            v_str = f"NEEDS REVIEW ({q.verification_note or 'Verifier disagreement'})"
        else:
            v_str = "NOT APPLICABLE (Unverified by design)"

        lines.append("-" * 80)
        lines.append(f"QUESTION {idx} [{q.type.value}] — {q.marks} Mark{'s' if q.marks > 1 else ''} | Difficulty: {q.difficulty.value}")
        lines.append(f"Concept: {q.concept} | Source: {q.source_file} (Original ID: {q.id})")
        lines.append(f"Verification Status: {v_str}")
        lines.append("-" * 80)
        lines.append(f"Question Text:\n{q.question_text}\n")

        if isinstance(q, MCQQuestion):
            lines.append("Options:")
            for i, opt in enumerate(q.options):
                lbl = opt_letters[i] if i < len(opt_letters) else str(i)
                lines.append(f"  {lbl}) {opt}")
            correct_lbl = (
                opt_letters[q.correct_option_index]
                if 0 <= q.correct_option_index < len(opt_letters)
                else str(q.correct_option_index)
            )
            lines.append(f"\nCorrect Option: Option {correct_lbl}")
            lines.append(f"Mark Scheme: [{q.marks} Mark{'s' if q.marks > 1 else ''} for Option {correct_lbl}; 0 marks for other options]")
            explanation_text = convert_option_indices_to_letters(q.explanation)
            lines.append(f"Explanation & Distractor Analysis:\n{explanation_text}")

        elif isinstance(q, CodeQuestion):
            if q.code_snippet:
                lines.append("Provided Code Snippet:")
                for s_line in q.code_snippet.splitlines():
                    lines.append(f"    {s_line}")
                lines.append("")
            lines.append(f"Expected Answer / Solution:\n{q.expected_answer}\n")
            lines.append(f"Mark Allocation Breakdown (Total: {q.marks} Marks):")
            alloc = breakdowns.get(q.id, [])
            for item in alloc:
                lines.append(f"  • {item['criterion']}: [{item['marks']} Mark{'s' if item['marks'] > 1 else ''}]")
            lines.append(f"\nPedagogical Explanation:\n{q.explanation}")

        elif isinstance(q, TheoryQuestion):
            lines.append(f"Model Answer:\n{q.model_answer}\n")
            lines.append(f"Mark Allocation Breakdown (Total: {q.marks} Marks):")
            alloc = breakdowns.get(q.id, [])
            for item in alloc:
                lines.append(f"  • {item['criterion']}: [{item['marks']} Mark{'s' if item['marks'] > 1 else ''}]")

        lines.append("")

    lines.append("=" * 80)
    lines.append("[END OF MARKING SCHEME]")
    lines.append("=" * 80)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    start_time = time.time()
    print("=" * 60)
    print("AGENT 3: THE EXAMINER — ASSESSMENT ASSEMBLY & CALIBRATION")
    print("=" * 60)

    # 1. Load Agent 2 questions
    if not os.path.isfile(INPUT_QUESTIONS_PATH):
        print(f"\n[Error] Questions file '{INPUT_QUESTIONS_PATH}' not found.")
        print("Please run agent2.py first to generate the raw question pool.")
        sys.exit(1)

    with open(INPUT_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        raw_quiz = json.load(f)

    # Parse questions through schema
    loaded_quiz = Quiz.model_validate(raw_quiz)
    questions = loaded_quiz.questions

    # Defensively ensure every question has a strictly unique ID q1, q2, ... qN and normalized option references
    seen_ids: set[str] = set()
    for idx, q in enumerate(questions, start=1):
        if not q.id or q.id in seen_ids:
            q.id = f"q{idx}"
        seen_ids.add(q.id)
        if isinstance(q, MCQQuestion):
            q.explanation = convert_option_indices_to_letters(q.explanation)

    print(f"Loaded {len(questions)} questions from '{INPUT_QUESTIONS_PATH}'.")

    # 2. Load concepts for syllabus coverage tracking
    all_concepts: list[Concept] = []
    if os.path.isfile(INPUT_CONCEPTS_PATH):
        try:
            with open(INPUT_CONCEPTS_PATH, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                all_concepts = [Concept.model_validate(c) for c in c_data.get("concepts", [])]
        except Exception as exc:
            # Do not swallow this silently: with all_concepts empty, the syllabus
            # coverage figure is computed against nothing and reports a
            # meaningless 100%, which looks like success rather than a load failure.
            print(
                f"[Warning] Could not load concepts from '{INPUT_CONCEPTS_PATH}': {exc}\n"
                f"          Syllabus coverage percentage will be unreliable for this run."
            )

    # 3. Initialize LLM client and select active model
    client, provider = get_llm_client()
    active_model, selection_reason = select_active_model(client)
    print(f"Examiner LLM provider initialized ({provider.upper()} / {active_model}).")
    print(f"Active Model Selection : {active_model} ({selection_reason})")

    # -----------------------------------------------------------------------
    # Part A: Difficulty Assignment (Batched Assessment)
    # -----------------------------------------------------------------------
    print("\nAssessing cognitive difficulty across all questions (Batched LLM call)...")
    difficulty_results = batch_assess_difficulties(questions, client, active_model)

    print("\n" + "=" * 60)
    print("DIFFICULTY ASSIGNMENTS WITH REASONING")
    print("=" * 60)
    for q in questions:
        diff_enum, reason = assign_difficulty(q)
        q.difficulty = diff_enum
        print(f"  [{q.id}] [{q.type.value:6s}] Assessed: {diff_enum.value:<6s} — {q.concept}")
        print(f"        Reasoning: {reason}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Part B: Paper Balance
    # -----------------------------------------------------------------------
    target = DifficultyTarget(easy_ratio=0.3, medium_ratio=0.5, hard_ratio=0.2)
    balance_note, target_counts, actual_counts = evaluate_paper_balance(questions, target)

    print("\nPAPER COGNITIVE BALANCE EVALUATION:")
    print(f"  - Target Distribution : {target_counts['EASY']} Easy / {target_counts['MEDIUM']} Medium / {target_counts['HARD']} Hard")
    print(f"  - Actual Distribution : {actual_counts['EASY']} Easy / {actual_counts['MEDIUM']} Medium / {actual_counts['HARD']} Hard")
    print(f"  - Status Report       : {balance_note}")

    # -----------------------------------------------------------------------
    # Part C: Deterministic Marks Allocation
    # -----------------------------------------------------------------------
    print("\nAllocating marks from (type, difficulty) matrix...")
    for q in questions:
        key = (q.type, q.difficulty)
        allocated_marks = MARKS_TABLE.get(key, 2)
        q.marks = allocated_marks

    total_marks = sum(q.marks for q in questions)
    print(f"Total Examination Marks computed: {total_marks} Marks.")

    # -----------------------------------------------------------------------
    # Part D: Itemized Marking Scheme Breakdown
    # -----------------------------------------------------------------------
    time.sleep(INTER_CALL_DELAY_SECONDS)
    code_and_theory_questions = [q for q in questions if isinstance(q, (CodeQuestion, TheoryQuestion))]
    print(f"\nGenerating criteria-level mark breakdowns for {len(code_and_theory_questions)} CODE & THEORY questions...")
    breakdowns, mismatch_reports = batch_generate_mark_breakdowns(code_and_theory_questions, client, active_model)

    # -----------------------------------------------------------------------
    # Part E: Paper Ordering & Document Assembly
    # -----------------------------------------------------------------------
    ordered_questions = order_questions_for_paper(questions)

    # Build final Quiz object
    final_quiz = Quiz(
        source_documents=loaded_quiz.source_documents,
        total_marks=total_marks,
        questions=ordered_questions,
        balance_note=balance_note,
    )

    # 1. Write outputs/final_quiz.json
    os.makedirs(os.path.dirname(FINAL_QUIZ_PATH), exist_ok=True)
    with open(FINAL_QUIZ_PATH, "w", encoding="utf-8") as f:
        json.dump(final_quiz.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Final Quiz object written to: {FINAL_QUIZ_PATH}")

    # 2. Write outputs/marking_scheme.json
    marking_scheme_json_data = {
        "source_documents": final_quiz.source_documents,
        "total_marks": final_quiz.total_marks,
        "balance_note": balance_note,
        "questions": [],
    }
    opt_letters = ["A", "B", "C", "D"]
    for idx, q in enumerate(ordered_questions, start=1):
        item: dict = {
            "question_number": idx,
            "id": q.id,
            "concept": q.concept,
            "type": q.type.value,
            "difficulty": q.difficulty.value,
            "marks": q.marks,
            "verification_status": q.verification_status,
            "verification_note": q.verification_note,
        }
        if isinstance(q, MCQQuestion):
            item["correct_option_index"] = q.correct_option_index
            item["correct_option_letter"] = (
                opt_letters[q.correct_option_index]
                if 0 <= q.correct_option_index < len(opt_letters)
                else str(q.correct_option_index)
            )
            item["explanation"] = q.explanation
        elif isinstance(q, CodeQuestion):
            item["expected_answer"] = q.expected_answer
            item["explanation"] = q.explanation
            item["mark_breakdown"] = breakdowns.get(q.id, [])
        elif isinstance(q, TheoryQuestion):
            item["model_answer"] = q.model_answer
            item["key_points"] = q.key_points
            item["mark_breakdown"] = breakdowns.get(q.id, [])

        marking_scheme_json_data["questions"].append(item)

    with open(MARKING_SCHEME_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(marking_scheme_json_data, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Structured Marking Scheme written to: {MARKING_SCHEME_JSON_PATH}")

    # 3. Write outputs/question_paper.txt
    question_paper_text = format_question_paper_text(final_quiz, ordered_questions)
    with open(QUESTION_PAPER_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(question_paper_text)
    print(f"[Saved] Student-facing Question Paper written to: {QUESTION_PAPER_TXT_PATH}")

    # 4. Write outputs/marking_scheme.txt
    marking_scheme_text = format_marking_scheme_text(final_quiz, ordered_questions, breakdowns)
    with open(MARKING_SCHEME_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(marking_scheme_text)
    print(f"[Saved] Examiner-facing Marking Scheme written to: {MARKING_SCHEME_TXT_PATH}")

    # -----------------------------------------------------------------------
    # Final Terminal Summary
    # -----------------------------------------------------------------------
    marks_by_type: dict[str, int] = {}
    for q in ordered_questions:
        marks_by_type[q.type.value] = marks_by_type.get(q.type.value, 0) + q.marks

    marks_by_difficulty: dict[str, int] = {}
    for q in ordered_questions:
        marks_by_difficulty[q.difficulty.value] = marks_by_difficulty.get(q.difficulty.value, 0) + q.marks

    represented_concepts = {q.concept for q in ordered_questions}
    total_concepts_count = len(all_concepts) if all_concepts else len(represented_concepts)
    concept_pct = (len(represented_concepts) / total_concepts_count) * 100 if total_concepts_count else 100.0

    needs_review_questions = [
        (idx, q) for idx, q in enumerate(ordered_questions, start=1) if q.verification_status == "needs_review"
    ]

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("EXAMINATION ASSEMBLY & EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Active Model       : {active_model} ({selection_reason})")
    print(f"Total LLM Calls    : {llm_call_count}")
    print(f"Wall-Clock Time    : {elapsed_time:.1f}s")
    print(f"Total Questions    : {len(ordered_questions)}")
    print(f"Total Marks        : {total_marks} Marks")
    print(f"Cognitive Balance  : {balance_note}")

    print("\nMark Breakdown Mismatch Resolutions:")
    if mismatch_reports:
        for qid, res in mismatch_reports:
            print(f"  - [{qid}] {res}")
    else:
        print("  None — all mark breakdowns aligned with question text on first pass.")

    print("\nMarks Allocation by Question Type:")
    for t_name in ["MCQ", "CODE", "THEORY"]:
        print(f"  - {t_name:6s}: {marks_by_type.get(t_name, 0):2d} Marks")

    print("\nMarks Allocation by Difficulty Band:")
    for d_name in ["EASY", "MEDIUM", "HARD"]:
        count = actual_counts.get(d_name, 0)
        m = marks_by_difficulty.get(d_name, 0)
        print(f"  - {d_name:6s}: {count} Questions, {m:2d} Marks")

    print(f"\nSyllabus Concept Coverage: {len(represented_concepts)}/{total_concepts_count} Concepts ({concept_pct:.1f}%)")

    print("\nQuestions Requiring Instructor Review:")
    if needs_review_questions:
        for q_idx, q in needs_review_questions:
            print(f"  - Question {q_idx} (ID: {q.id}, Concept: '{q.concept}'):")
            print(f"      Status: {q.verification_status.upper()} — {q.verification_note}")
    else:
        print("  None — all questions passed independent verification or are open-ended theory questions.")

    print("=" * 60)


if __name__ == "__main__":
    main()
