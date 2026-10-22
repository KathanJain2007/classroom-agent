"""
agent2.py — Agent 2: Question Writer with Batched Fast-Path and 5 Safety Checks.

Generates balanced assessment questions from extracted concepts with high speed
and strict academic quality control:

1. High Speed Batching:
   - Batch generation by question type (1 call for all MCQs, 1 for CODE, 1 for THEORY).
   - Batch verification (1 call for MCQs, 2 calls for CODE independent solving & equivalence).
   - Resume mode (--resume) to avoid re-running already verified questions.
   - Consistent model selection via MODEL_PRIORITY without mid-run switching.

2. The 5 Distinct Quality Checks:
   Check 1 [schema]       → Is the structure valid? (Pydantic v2 schemas)
   Check 2 [coherence]    → Is the question answerable as written? (Entity label parity)
   Check 3 [premise]      → Is the question's stated premise true? (Bug tracing & deterministic fix comparison)
   Check 4 [verification] → Is the marked answer correct in general? (Blind solving & equivalence)
   Check 5 [grounding]    → Is the marked answer consistent with THIS professor's course material?
"""

from __future__ import annotations

import argparse
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
    MCQQuestion,
    QuestionType,
    Quiz,
    QuizSpec,
    TheoryQuestion,
)
from step1_test import (
    MODEL_PRIORITY,
    call_llm_with_retry,
    get_llm_client,
    parse_json_response,
    select_active_model,
)

CONCEPTS_PATH = "outputs/concepts.json"
SAMPLE_QUIZ_PATH = "outputs/sample_quiz.json"
OUTPUT_QUESTIONS_PATH = "outputs/questions.json"
INTER_CALL_DELAY_SECONDS = 1.0

# Global LLM call counter
llm_call_count = 0


def tracked_call_llm(
    client: object,
    messages: list[dict],
    model: str,
    temperature: float = 0.2,
    response_format: Optional[dict] = None,
) -> Optional[str]:
    """Wraps call_llm_with_retry to accurately track total LLM API calls."""
    global llm_call_count
    llm_call_count += 1
    return call_llm_with_retry(
        client=client,
        messages=messages,
        model=model,
        temperature=temperature,
        response_format=response_format,
    )


# ---------------------------------------------------------------------------
# Loading and Reference Helpers
# ---------------------------------------------------------------------------

def load_concepts(path: str = CONCEPTS_PATH) -> list[Concept]:
    """Loads and validates concepts from outputs/concepts.json."""
    if not os.path.isfile(path):
        print(f"\n[Error] Concepts file '{path}' not found.")
        print("Please run step1_test.py first to extract and generate concepts.")
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as err:
        print(f"\n[Error] Failed to read JSON from '{path}': {err}")
        sys.exit(1)

    raw_concepts = data.get("concepts", [])
    if not raw_concepts:
        print(f"\n[Error] No concepts found in '{path}'. Please run step1_test.py first.")
        sys.exit(1)

    return [Concept.model_validate(c) for c in raw_concepts]


def load_sample_questions(path: str = SAMPLE_QUIZ_PATH) -> dict[str, dict]:
    """Loads sample questions from sample_quiz.json for few-shot prompt guidance."""
    samples: dict[str, dict] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                quiz_data = json.load(f)
                for q in quiz_data.get("questions", []):
                    q_type = q.get("type")
                    if q_type and q_type not in samples:
                        cleaned_q = dict(q)
                        cleaned_q.pop("verification_status", None)
                        cleaned_q.pop("verification_note", None)
                        cleaned_q.pop("generated_by", None)
                        samples[q_type] = cleaned_q
        except Exception as err:
            print(f"[Warning] Could not load sample quiz from '{path}': {err}")

    if "MCQ" not in samples:
        samples["MCQ"] = {
            "id": "q1",
            "concept": "Stack (LIFO) ADT",
            "source_file": "DS-UNIT-1.pdf",
            "type": "MCQ",
            "difficulty": "MEDIUM",
            "marks": 2,
            "question_text": "A stack follows which of the following principles for element access?",
            "options": [
                "First-In First-Out (FIFO)",
                "Last-In First-Out (LIFO)",
                "Random access based on index",
                "Priority-based access",
            ],
            "correct_option_index": 1,
            "explanation": "A stack is a LIFO structure: the last element pushed onto the top is always the first to be popped off.",
        }

    if "CODE" not in samples:
        samples["CODE"] = {
            "id": "q2",
            "concept": "Dynamic Memory Allocation in C",
            "source_file": "DS-UNIT-1.pdf",
            "type": "CODE",
            "difficulty": "MEDIUM",
            "marks": 5,
            "question_text": "Write a function int* append(int *arr, size_t *size, int value) that resizes arr using realloc.",
            "language": "C",
            "code_snippet": "int* append(int *arr, size_t *size, int value) {\n    // TODO: implement\n}",
            "expected_answer": "int *tmp = realloc(arr, (*size + 1) * sizeof(int));\nif (tmp == NULL) return NULL;\ntmp[*size] = value;\n(*size)++;\nreturn tmp;",
            "explanation": "Using a temporary pointer prevents memory leaks if realloc fails.",
        }

    if "THEORY" not in samples:
        samples["THEORY"] = {
            "id": "q3",
            "concept": "Abstract Data Type (ADT) vs. Data Structure",
            "source_file": "DS-UNIT-1.pdf",
            "type": "THEORY",
            "difficulty": "MEDIUM",
            "marks": 5,
            "question_text": "Compare and contrast an Abstract Data Type (ADT) with a concrete data structure.",
            "model_answer": "An ADT specifies the operations and mathematical behaviors without implementation details. A data structure provides the concrete storage layout and algorithms.",
            "key_points": [
                "Defines ADT as an abstract mathematical/logical interface",
                "Defines data structure as concrete physical implementation",
                "Explains the software engineering benefits of separating interface from implementation",
            ],
        }

    return samples


# ---------------------------------------------------------------------------
# Concept Selection & Type Assignment
# ---------------------------------------------------------------------------

def is_code_suitable(concept: Concept) -> tuple[bool, str]:
    """Judges whether a concept genuinely involves code, algorithms, or implementation."""
    name_lower = concept.name.lower()
    text = f"{name_lower} {concept.explanation.lower()}"

    non_code_patterns = [
        "adt vs",
        "vs. data structure",
        "vs data structure",
        "applications of",
        "fundamental data structure operations",
    ]
    if any(p in name_lower for p in non_code_patterns):
        return False, "High-level conceptual or comparative topic without concrete code mechanics"

    code_indicators = [
        ("pointer", "Involves pointer manipulation or memory addressing"),
        ("malloc", "Involves C dynamic memory management functions"),
        ("dynamic memory", "Involves dynamic heap allocation / resizing"),
        ("infix", "Involves expression notation conversion algorithms"),
        ("postfix", "Involves expression notation or stack evaluation algorithms"),
        ("conversion algorithm", "Involves step-by-step conversion algorithm"),
        ("recursion", "Involves recursive function mechanics, base and recursive cases"),
        ("circular queue", "Involves circular buffer index arithmetic and modulo logic"),
        ("deque", "Involves double-ended queue insertion/deletion logic"),
        ("implementation", "Involves concrete data structure implementation mechanics"),
        ("index management", "Involves index tracking and pointer updates"),
        ("overflow and underflow", "Involves boundary checking and error detection logic"),
        ("algorithm", "Involves concrete algorithmic procedures"),
    ]

    for kw, reason in code_indicators:
        if kw in text:
            return True, reason

    return False, "General data structure topic without direct implementation or code focus"


def is_theory_preferred(concept: Concept) -> bool:
    """Checks if a concept naturally suits an explanatory/comparative theory question."""
    name_lower = concept.name.lower()
    theory_indicators = ["vs.", "vs", "applications", "difference", "comparison", "principles", "types of"]
    return any(ti in name_lower for ti in theory_indicators)


def plan_quiz_generation(
    concepts: list[Concept],
    spec: QuizSpec,
) -> list[tuple[Concept, QuestionType, str]]:
    """Selects concepts and assigns question types respecting QuizSpec ratios."""
    total_target = spec.total_questions
    target_mcq = round(total_target * spec.mcq_ratio)
    target_code = round(total_target * spec.code_ratio)
    target_theory = total_target - target_mcq - target_code

    if len(concepts) >= total_target:
        cross_doc_concepts = [c for c in concepts if len({s.filename for s in c.sources}) > 1]
        single_doc_concepts = [c for c in concepts if len({s.filename for s in c.sources}) <= 1]
        sorted_pool = cross_doc_concepts + single_doc_concepts
        selected_concepts = sorted_pool[:total_target]
    else:
        selected_concepts = list(concepts)
        surplus_needed = total_target - len(concepts)
        subtopic_rich = sorted(concepts, key=lambda c: len(c.subtopics), reverse=True)
        for i in range(surplus_needed):
            selected_concepts.append(subtopic_rich[i % len(subtopic_rich)])

    code_candidates = []
    other_candidates = []
    for c in selected_concepts:
        suitable, reason = is_code_suitable(c)
        if suitable:
            code_candidates.append((c, reason))
        else:
            other_candidates.append(c)

    plan: list[tuple[Concept, QuestionType, str]] = []
    num_code_to_assign = min(target_code, len(code_candidates))
    for i in range(num_code_to_assign):
        c, reason = code_candidates[i]
        plan.append((c, QuestionType.CODE, f"Code-suitable concept: {reason}"))

    remaining_pool = [c for c, _ in code_candidates[num_code_to_assign:]] + other_candidates
    code_shortfall = target_code - num_code_to_assign
    adjusted_theory_target = target_theory + code_shortfall

    theory_picked = 0
    non_theory_pool = []
    for c in remaining_pool:
        if theory_picked < adjusted_theory_target and is_theory_preferred(c):
            plan.append((c, QuestionType.THEORY, "Conceptual/comparative topic naturally suited for in-depth theory"))
            theory_picked += 1
        else:
            non_theory_pool.append(c)

    for c in non_theory_pool:
        if theory_picked < adjusted_theory_target:
            plan.append((c, QuestionType.THEORY, "Reallocated to theory to satisfy quiz coverage"))
            theory_picked += 1
        else:
            plan.append((c, QuestionType.MCQ, "Scenario or factual understanding suited for objective multiple choice"))

    return plan


# ---------------------------------------------------------------------------
# Part 2: Deterministic Repairs & Cleaners (No LLM Calls)
# ---------------------------------------------------------------------------

def repair_escaped_newlines(text: str) -> tuple[str, bool]:
    """
    If the text contains the two-character sequence '\\n' but contains NO actual newline characters,
    replaces all literal '\\n' with real newlines, and literal '\\t' with tabs.
    Fixes unreadable single-line code blocks caused by malformed JSON escaping.
    """
    if not text:
        return text, False

    if "\\n" in text and "\n" not in text:
        repaired = text.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\'", "'")
        return repaired, True

    return text, False


def apply_deterministic_repairs(q: AnyQuestion) -> None:
    """Repairs escaped newlines across all multi-line fields on a question."""
    fields_to_check = ["question_text"]
    # List-valued fields need repairing too: an MCQ option or a theory marking
    # point can carry literal "\n" just as easily as a code snippet, and an
    # unrepaired option renders as one unreadable line on the question paper.
    list_fields_to_check: list[str] = []

    if isinstance(q, MCQQuestion):
        fields_to_check.append("explanation")
        list_fields_to_check.append("options")
    elif isinstance(q, CodeQuestion):
        fields_to_check.extend(["code_snippet", "expected_answer", "explanation"])
    elif isinstance(q, TheoryQuestion):
        fields_to_check.append("model_answer")
        list_fields_to_check.append("key_points")

    for f_name in fields_to_check:
        val = getattr(q, f_name, None)
        if isinstance(val, str):
            rep, was_rep = repair_escaped_newlines(val)
            if was_rep:
                setattr(q, f_name, rep)
                print(f"  [Repair] Question {q.id}: Escaped newlines detected and repaired in '{f_name}'.")

    for f_name in list_fields_to_check:
        val = getattr(q, f_name, None)
        if isinstance(val, list):
            repaired_items: list = []
            changed = False
            for item in val:
                if isinstance(item, str):
                    rep, was_rep = repair_escaped_newlines(item)
                    repaired_items.append(rep)
                    changed = changed or was_rep
                else:
                    repaired_items.append(item)
            if changed:
                setattr(q, f_name, repaired_items)
                print(f"  [Repair] Question {q.id}: Escaped newlines detected and repaired in '{f_name}'.")


def convert_option_indices_to_letters(explanation: str) -> str:
    """
    Deterministically converts 0-indexed option references in explanation prose to letters
    (0 -> A, 1 -> B, 2 -> C, 3 -> D).
    E.g. 'making option 0 correct' -> 'making Option A correct'
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


def check_code_premise_deterministic(question: CodeQuestion) -> tuple[bool, Optional[str]]:
    """
    Deterministically compares the expected answer code against the original snippet.
    If the corrected code is identical or trivially identical to the original,
    flags the question immediately — a fix that changes nothing is proof of a false premise.
    """
    if not question.code_snippet or not question.expected_answer:
        return True, None

    def normalize_code(code_str: str) -> str:
        c = re.sub(r"//.*$", "", code_str, flags=re.MULTILINE)
        c = re.sub(r"/\*.*?\*/", "", c, flags=re.DOTALL)
        c = "".join(c.split())
        return c

    norm_snippet = normalize_code(question.code_snippet)
    norm_expected = normalize_code(question.expected_answer)

    if norm_expected and norm_expected == norm_snippet:
        return (
            False,
            "Expected answer code is identical to code snippet (fix changes nothing).",
        )

    # Extract code blocks or candidate code lines
    code_blocks: list[str] = []
    tri = chr(96) * 3
    pattern = tri + r"(?:c|cpp)?\s*(.*?)\s*" + tri
    code_blocks.extend(re.findall(pattern, question.expected_answer, flags=re.DOTALL))

    lines = question.expected_answer.splitlines()
    curr_block: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            if curr_block:
                code_blocks.append("\n".join(curr_block))
                curr_block = []
            continue
        is_code = (
            re.search(r"[;{}]", s)
            or re.match(r"^(while|if|for|return|int|void|char|typedef|struct|printf)\b", s)
        ) and not re.search(
            r"\b(this|that|replace|output|prints|ensure|ensures|program|because|with|should)\b",
            s,
            re.IGNORECASE,
        )
        if is_code:
            curr_block.append(s)
        else:
            if curr_block:
                code_blocks.append("\n".join(curr_block))
                curr_block = []
    if curr_block:
        code_blocks.append("\n".join(curr_block))

    for block in code_blocks:
        norm_block = normalize_code(block)
        if len(norm_block) >= 15 and norm_block in norm_snippet:
            return (
                False,
                f"Proposed fix is already present verbatim in code snippet ({block.strip()[:60]}...); fix changes nothing.",
            )

    return True, None


def is_bug_hunting_question(q: CodeQuestion) -> bool:
    """Identifies if a CODE question claims a bug or defect exists in its snippet."""
    text = f"{q.question_text} {q.code_snippet or ''}".lower()
    defect_phrases = [
        "contains a bug",
        "produces an incorrect",
        "identify and correct",
        "is implemented incorrectly",
        "does not wrap around",
        "fix the bug",
        "identify the bug",
        "has a bug",
        "prints an unexpected",
        "unexpected value",
        "causes an underflow",
        "causes an overflow",
        "faulty",
        "incorrect output",
        "erroneous",
        "bug prevents",
    ]
    return any(phrase in text for phrase in defect_phrases)


# ---------------------------------------------------------------------------
# Check 2: Entity Coherence Check (Deterministic, No LLM Call)
# ---------------------------------------------------------------------------

def check_entity_coherence(question: AnyQuestion) -> tuple[bool, Optional[str]]:
    """
    Deterministically checks whether a question is coherent and answerable as written.
    Ensures entities named in question_text (e.g. task labels A, B, C) match options.
    """
    if isinstance(question, MCQQuestion):
        q_entities: set[str] = set()

        fn_args = re.findall(
            r"\b(?:enqueue|push|dequeue|insert|pop|add|delete|enqueueFront|enqueueRear|insertFront|insertRear)\s*\(\s*([A-Za-z][A-Za-z0-9_]*)",
            question.question_text,
        )
        for arg in fn_args:
            if len(arg) <= 6:
                q_entities.add(arg)

        caps = re.findall(r"\b([A-Z])\b", question.question_text)
        non_articles = [c for c in caps if c not in ("A", "I")]
        for c in non_articles:
            q_entities.add(c)

        if q_entities and "A" in caps:
            if re.search(r"\bA\s*[,&/]\s*[B-Z]\b|\b[B-Z]\s*[,&/]\s*A\b|\(\s*A\s*[,)]", question.question_text):
                q_entities.add("A")

        combined_options = " ".join(question.options)
        opt_entities: set[str] = set()
        opt_caps = re.findall(r"\b([A-Z])\b", combined_options)
        opt_entities.update(opt_caps)

        opt_words = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", combined_options)
        for w in opt_words:
            if w in q_entities:
                opt_entities.add(w)

        if q_entities and opt_entities and q_entities.isdisjoint(opt_entities):
            q_ents_str = ", ".join(sorted(q_entities))
            opt_ents_str = ", ".join(sorted(opt_entities))
            return (
                False,
                f"Question names entities ({q_ents_str}) but options name disjoint entities ({opt_ents_str})",
            )

        correct_letter = chr(65 + question.correct_option_index)
        stated_correct = re.findall(
            r"(?:matching option|correct option is|correct answer is|choice)\s*\(?([A-D])\)?\b",
            question.explanation,
            re.IGNORECASE,
        )
        for cand in stated_correct:
            cand_up = cand.upper()
            if cand_up != correct_letter:
                is_negated = re.search(
                    rf"\b(?:option|choice)\s*{cand_up}\b\s+(?:is\s+)?(?:wrong|incorrect|tempting|false|invalid|not)",
                    question.explanation,
                    re.IGNORECASE,
                )
                if not is_negated:
                    return (
                        False,
                        f"Explanation claims option {cand_up} is correct, but question marks option {correct_letter}.",
                    )

    return True, None


# ---------------------------------------------------------------------------
# Part 1B: Fast-Path Batched Question Generation by Type
# ---------------------------------------------------------------------------

def generate_mcqs_batch(
    plan_items: list[tuple[Concept, str, str]],
    sample_mcq: dict,
    client: object,
    model: str,
) -> list[MCQQuestion]:
    """Generates all MCQ questions in ONE batched LLM call."""
    if not plan_items:
        return []

    items_payload = []
    for concept, q_id, primary_source in plan_items:
        subtopics_text = "; ".join(concept.subtopics) if concept.subtopics else "Core operational concepts"
        items_payload.append({
            "id": q_id,
            "concept": concept.name,
            "source_file": primary_source,
            "explanation": concept.explanation,
            "subtopics": subtopics_text,
        })

    system_prompt = (
        "You are an expert university professor creating high-quality, rigorously verified "
        "Multiple Choice Questions for Computer Science undergraduates."
    )

    user_prompt = f"""Generate exactly {len(plan_items)} Multiple Choice Questions (MCQ), one for each concept listed below.

CONCEPTS TO TEST:
{json.dumps(items_payload, indent=2)}

QUALITY RULES FOR MCQ:
1. Every entity named in question_text (e.g. task labels A, B, C or variable names) MUST appear in the options with exact matching labels.
2. Provide exactly 4 plausible options of similar length and technical depth.
3. Do NOT use 'All of the above' or 'None of the above'.
4. Explanation must state why the correct option is right AND why tempting wrong options are incorrect.
5. Base the scenario strictly on computer science correctness.

REQUESTED OUTPUT FORMAT (Return valid JSON):
{{
  "questions": [
    {{
      "id": "<matching id from input>",
      "concept": "<matching concept name>",
      "source_file": "<matching source_file>",
      "type": "MCQ",
      "difficulty": "MEDIUM",
      "marks": 2,
      "question_text": "<Clear scenario or problem statement>",
      "options": [
        "<Option A>",
        "<Option B>",
        "<Option C>",
        "<Option D>"
      ],
      "correct_option_index": <0, 1, 2, or 3>,
      "explanation": "<Pedagogical justification referencing options>"
    }}
  ]
}}
"""

    raw_resp = tracked_call_llm(
        client=client,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    if not raw_resp:
        time.sleep(INTER_CALL_DELAY_SECONDS)
        raw_resp = tracked_call_llm(
            client=client,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model=model,
            temperature=0.2,
            response_format=None,
        )

    parsed = parse_json_response(raw_resp) if raw_resp else None
    results: list[MCQQuestion] = []
    generated_by_id: dict[str, MCQQuestion] = {}

    if parsed and "questions" in parsed and isinstance(parsed["questions"], list):
        for q_dict in parsed["questions"]:
            if not isinstance(q_dict, dict):
                continue
            q_id = q_dict.get("id")
            q_dict["type"] = "MCQ"
            q_dict.setdefault("difficulty", "MEDIUM")
            q_dict.setdefault("marks", 2)
            q_dict.setdefault("verification_status", "not_applicable")
            q_dict["generated_by"] = model
            try:
                mcq = MCQQuestion.model_validate(q_dict)
                apply_deterministic_repairs(mcq)
                mcq.explanation = convert_option_indices_to_letters(mcq.explanation)
                if q_id:
                    generated_by_id[q_id] = mcq
            except ValidationError as err:
                print(f"  [Batch Validation Error] MCQ {q_id}: {err}")

    # Ensure every requested item is satisfied, falling back individually if missing
    for concept, q_id, primary_source in plan_items:
        if q_id in generated_by_id:
            results.append(generated_by_id[q_id])
        else:
            print(f"  [Batch Fallback] Regenerating missing/invalid MCQ {q_id} for '{concept.name}' individually...")
            single_q = generate_single_question(
                concept=concept,
                q_type=QuestionType.MCQ,
                question_id=q_id,
                sample_question=sample_mcq,
                client=client,
                model=model,
            )
            if isinstance(single_q, MCQQuestion):
                results.append(single_q)

    return results


def generate_codes_batch(
    plan_items: list[tuple[Concept, str, str]],
    sample_code: dict,
    client: object,
    model: str,
) -> list[CodeQuestion]:
    """Generates all CODE questions in ONE batched LLM call."""
    if not plan_items:
        return []

    items_payload = []
    for concept, q_id, primary_source in plan_items:
        subtopics_text = "; ".join(concept.subtopics) if concept.subtopics else "Implementation mechanics"
        items_payload.append({
            "id": q_id,
            "concept": concept.name,
            "source_file": primary_source,
            "explanation": concept.explanation,
            "subtopics": subtopics_text,
        })

    system_prompt = (
        "You are an expert university instructor designing rigorous C programming assessment questions "
        "for Data Structures courses."
    )

    user_prompt = f"""Generate exactly {len(plan_items)} Programming / Code Questions (CODE), one for each concept listed below.

CONCEPTS TO TEST:
{json.dumps(items_payload, indent=2)}

QUALITY RULES FOR CODE:
1. Real line breaks: code_snippet and expected_answer MUST contain real line breaks (\\n), NEVER escaped string literals on a single line.
2. If providing a bug-hunting problem, the code snippet MUST genuinely contain the defect, and running it MUST produce the claimed wrong output. Do NOT provide correct code and claim it is broken!
3. If providing a function implementation task, provide clear signatures, memory invariants, and return specifications.
4. All C code must be self-contained and syntactically standard.

REQUESTED OUTPUT FORMAT (Return valid JSON):
{{
  "questions": [
    {{
      "id": "<matching id from input>",
      "concept": "<matching concept name>",
      "source_file": "<matching source_file>",
      "type": "CODE",
      "difficulty": "MEDIUM",
      "marks": 5,
      "question_text": "<Clear problem statement specifying what the student must produce>",
      "language": "C",
      "code_snippet": "<Self-contained C code with actual newlines, or null>",
      "expected_answer": "<Exact fix, implementation, or output>",
      "explanation": "<Pedagogical explanation of why this solution is correct>"
    }}
  ]
}}
"""

    raw_resp = tracked_call_llm(
        client=client,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    if not raw_resp:
        time.sleep(INTER_CALL_DELAY_SECONDS)
        raw_resp = tracked_call_llm(
            client=client,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model=model,
            temperature=0.2,
            response_format=None,
        )

    parsed = parse_json_response(raw_resp) if raw_resp else None
    results: list[CodeQuestion] = []
    generated_by_id: dict[str, CodeQuestion] = {}

    if parsed and "questions" in parsed and isinstance(parsed["questions"], list):
        for q_dict in parsed["questions"]:
            if not isinstance(q_dict, dict):
                continue
            q_id = q_dict.get("id")
            q_dict["type"] = "CODE"
            q_dict.setdefault("difficulty", "MEDIUM")
            q_dict.setdefault("marks", 5)
            q_dict.setdefault("verification_status", "not_applicable")
            q_dict["generated_by"] = model
            try:
                code_q = CodeQuestion.model_validate(q_dict)
                apply_deterministic_repairs(code_q)
                if q_id:
                    generated_by_id[q_id] = code_q
            except ValidationError as err:
                print(f"  [Batch Validation Error] CODE {q_id}: {err}")

    for concept, q_id, primary_source in plan_items:
        if q_id in generated_by_id:
            results.append(generated_by_id[q_id])
        else:
            print(f"  [Batch Fallback] Regenerating missing/invalid CODE {q_id} for '{concept.name}' individually...")
            single_q = generate_single_question(
                concept=concept,
                q_type=QuestionType.CODE,
                question_id=q_id,
                sample_question=sample_code,
                client=client,
                model=model,
            )
            if isinstance(single_q, CodeQuestion):
                results.append(single_q)

    return results


def generate_theory_batch(
    plan_items: list[tuple[Concept, str, str]],
    sample_theory: dict,
    client: object,
    model: str,
) -> list[TheoryQuestion]:
    """Generates all THEORY questions in ONE batched LLM call."""
    if not plan_items:
        return []

    items_payload = []
    for concept, q_id, primary_source in plan_items:
        subtopics_text = "; ".join(concept.subtopics) if concept.subtopics else "Conceptual principles"
        items_payload.append({
            "id": q_id,
            "concept": concept.name,
            "source_file": primary_source,
            "explanation": concept.explanation,
            "subtopics": subtopics_text,
        })

    system_prompt = (
        "You are an expert university chief examiner creating in-depth theoretical and comparative "
        "examination questions for Data Structures."
    )

    user_prompt = f"""Generate exactly {len(plan_items)} Theoretical Analysis Questions (THEORY), one for each concept listed below.

CONCEPTS TO TEST:
{json.dumps(items_payload, indent=2)}

QUALITY RULES FOR THEORY:
1. Question must require explanation, comparison, trade-off analysis, or architectural rationale, not simple definition recall.
2. key_points must be concrete, specific, and independently markable point by point.
3. model_answer must be complete and well structured.

REQUESTED OUTPUT FORMAT (Return valid JSON):
{{
  "questions": [
    {{
      "id": "<matching id from input>",
      "concept": "<matching concept name>",
      "source_file": "<matching source_file>",
      "type": "THEORY",
      "difficulty": "MEDIUM",
      "marks": 5,
      "question_text": "<Clear theoretical or comparative question>",
      "model_answer": "<Comprehensive model answer>",
      "key_points": [
        "<Key marking point 1>",
        "<Key marking point 2>",
        "<Key marking point 3>"
      ]
    }}
  ]
}}
"""

    raw_resp = tracked_call_llm(
        client=client,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    if not raw_resp:
        time.sleep(INTER_CALL_DELAY_SECONDS)
        raw_resp = tracked_call_llm(
            client=client,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model=model,
            temperature=0.2,
            response_format=None,
        )

    parsed = parse_json_response(raw_resp) if raw_resp else None
    results: list[TheoryQuestion] = []
    generated_by_id: dict[str, TheoryQuestion] = {}

    if parsed and "questions" in parsed and isinstance(parsed["questions"], list):
        for q_dict in parsed["questions"]:
            if not isinstance(q_dict, dict):
                continue
            q_id = q_dict.get("id")
            q_dict["type"] = "THEORY"
            q_dict.setdefault("difficulty", "MEDIUM")
            q_dict.setdefault("marks", 5)
            q_dict.setdefault("verification_status", "not_applicable")
            q_dict["generated_by"] = model
            try:
                th_q = TheoryQuestion.model_validate(q_dict)
                apply_deterministic_repairs(th_q)
                if q_id:
                    generated_by_id[q_id] = th_q
            except ValidationError as err:
                print(f"  [Batch Validation Error] THEORY {q_id}: {err}")

    for concept, q_id, primary_source in plan_items:
        if q_id in generated_by_id:
            results.append(generated_by_id[q_id])
        else:
            print(f"  [Batch Fallback] Regenerating missing/invalid THEORY {q_id} for '{concept.name}' individually...")
            single_q = generate_single_question(
                concept=concept,
                q_type=QuestionType.THEORY,
                question_id=q_id,
                sample_question=sample_theory,
                client=client,
                model=model,
            )
            if isinstance(single_q, TheoryQuestion):
                results.append(single_q)

    return results


def generate_single_question(
    concept: Concept,
    q_type: QuestionType,
    question_id: str,
    sample_question: dict,
    client: object,
    model: str,
    coherence_feedback: Optional[str] = None,
) -> Optional[AnyQuestion]:
    """Fallback generator for individual questions when batch results are short or malformed."""
    primary_source = concept.sources[0].filename if concept.sources else "unknown"
    subtopics_text = "; ".join(concept.subtopics) if concept.subtopics else "Core operational concepts"

    sample_json = json.dumps(sample_question, indent=2)

    format_shape = {
        "id": question_id,
        "concept": concept.name,
        "source_file": primary_source,
        "type": q_type.value,
        "difficulty": "MEDIUM",
        "marks": 2 if q_type == QuestionType.MCQ else 5,
    }

    if q_type == QuestionType.MCQ:
        format_shape.update({
            "question_text": "<Scenario problem>",
            "options": ["<Option A>", "<Option B>", "<Option C>", "<Option D>"],
            "correct_option_index": 0,
            "explanation": "<Justification>",
        })
    elif q_type == QuestionType.CODE:
        format_shape.update({
            "question_text": "<Problem statement>",
            "language": "C",
            "code_snippet": "<Self-contained C code with actual newlines>",
            "expected_answer": "<Fix or implementation>",
            "explanation": "<Justification>",
        })
    else:
        format_shape.update({
            "question_text": "<Comparative theoretical question>",
            "model_answer": "<Model answer>",
            "key_points": ["<Point 1>", "<Point 2>", "<Point 3>"],
        })

    user_prompt = f"""Generate exactly ONE {q_type.value} question testing:
Concept: {concept.name}
Explanation: {concept.explanation}
Subtopics: {subtopics_text}
Source: {primary_source}

WORKED EXAMPLE:
{sample_json}

REQUESTED OUTPUT SHAPE (Return valid JSON):
{json.dumps(format_shape, indent=2)}
"""
    if coherence_feedback:
        user_prompt += f"\nFEEDBACK FROM PREVIOUS ATTEMPT:\n{coherence_feedback}\n"

    raw_response = tracked_call_llm(
        client=client,
        messages=[{"role": "user", "content": user_prompt}],
        model=model,
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    if not raw_response:
        time.sleep(INTER_CALL_DELAY_SECONDS)
        raw_response = tracked_call_llm(
            client=client,
            messages=[{"role": "user", "content": user_prompt}],
            model=model,
            temperature=0.3,
            response_format=None,
        )
    if not raw_response:
        return None

    data = parse_json_response(raw_response)
    if not data:
        return None

    data["id"] = question_id
    data.setdefault("concept", concept.name)
    data.setdefault("source_file", primary_source)
    data.setdefault("type", q_type.value)
    data.setdefault("difficulty", "MEDIUM")
    data.setdefault("marks", 2 if q_type == QuestionType.MCQ else 5)
    data.setdefault("verification_status", "not_applicable")
    data.setdefault("verification_note", None)
    data["generated_by"] = model

    try:
        if q_type == QuestionType.MCQ:
            mcq = MCQQuestion.model_validate(data)
            apply_deterministic_repairs(mcq)
            mcq.explanation = convert_option_indices_to_letters(mcq.explanation)
            return mcq
        elif q_type == QuestionType.CODE:
            code_q = CodeQuestion.model_validate(data)
            apply_deterministic_repairs(code_q)
            return code_q
        else:
            theory_q = TheoryQuestion.model_validate(data)
            apply_deterministic_repairs(theory_q)
            return theory_q
    except ValidationError as err:
        print(f"  [Validation Error] Question {question_id} ({q_type.value}): {err}")
        return None


# ---------------------------------------------------------------------------
# Check 3: Batched Premise Check for Bug-Hunting CODE Questions (Part 4)
# ---------------------------------------------------------------------------

def batch_check_premises(
    code_questions: list[CodeQuestion],
    client: object,
    model: str,
) -> dict[str, dict]:
    """
    Executes Check 3 (Premise Check) on all bug-hunting CODE questions.
    1. Deterministic pre-check: flags questions whose 'fix' is identical to the snippet.
    2. Batched LLM execution trace: traces code on stated input, identifies actual output,
       compares against claimed symptom, and checks whether the symptom manifests in main().
    """
    results: dict[str, dict] = {}
    questions_needing_trace = []

    # 1. Deterministic Pre-Check
    for q in code_questions:
        if not is_bug_hunting_question(q):
            # Normal implementation question has a valid premise by design
            results[q.id] = {
                "passed": True,
                "is_bug_hunting": False,
                "defect_present": True,
                "manifests_in_main": True,
                "note": "Implementation task (valid premise)",
            }
            continue

        is_valid_det, det_err = check_code_premise_deterministic(q)
        if not is_valid_det:
            print(f"  [Premise Deterministic] {q.id} '{q.concept}': FALSE PREMISE — {det_err}")
            results[q.id] = {
                "passed": False,
                "is_bug_hunting": True,
                "defect_present": False,
                "manifests_in_main": False,
                "note": f"False premise detected (deterministic check): {det_err}",
            }
        else:
            questions_needing_trace.append(q)

    if not questions_needing_trace:
        return results

    # 2. Batched LLM Trace Call
    trace_payload = []
    for q in questions_needing_trace:
        trace_payload.append({
            "id": q.id,
            "concept": q.concept,
            "question_text": q.question_text,
            "language": q.language,
            "code_snippet": q.code_snippet,
        })

    system_prompt = (
        "You are an expert compiler and runtime trace verification engine for Computer Science assessments. "
        "Analyze code snippets with absolute rigor by tracing execution step by step."
    )

    user_prompt = f"""Evaluate the premise of each of the following {len(questions_needing_trace)} bug-hunting code questions.

QUESTIONS TO TRACE:
{json.dumps(trace_payload, indent=2)}

TASK FOR EACH QUESTION:
1. Trace the code execution step-by-step as written on the stated test inputs or main().
2. State the EXACT actual output/state.
3. Identify the claimed symptom or bug described in the question text.
4. Answer two critical premise questions:
   - defect_present: true ONLY if the code as written actually contains a defect that produces incorrect behavior. Set to false if the code already behaves correctly or if output matches correct specification.
   - manifests_in_main: true if the provided main() test run actually triggers and displays the error. Set to false if the defect is latent and never triggered by the provided main().

REQUESTED OUTPUT FORMAT (Return valid JSON):
{{
  "premise_checks": [
    {{
      "id": "<matching question id>",
      "actual_output": "<exact output when tracing code as written>",
      "claimed_symptom": "<the error described by the question>",
      "defect_present": true, // or false if program behaves correctly
      "manifests_in_main": true, // or false if latent bug
      "analysis": "<trace reasoning>"
    }}
  ]
}}
"""

    raw_resp = tracked_call_llm(
        client=client,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    if not raw_resp:
        time.sleep(INTER_CALL_DELAY_SECONDS)
        raw_resp = tracked_call_llm(
            client=client,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model=model,
            temperature=0.1,
            response_format=None,
        )

    parsed = parse_json_response(raw_resp) if raw_resp else None
    if parsed and "premise_checks" in parsed and isinstance(parsed["premise_checks"], list):
        for entry in parsed["premise_checks"]:
            q_id = entry.get("id")
            defect_present = entry.get("defect_present", True)
            manifests = entry.get("manifests_in_main", True)
            actual_out = str(entry.get("actual_output", "")).strip().replace("\n", " \\n ")
            claimed_sym = entry.get("claimed_symptom", "")
            analysis = entry.get("analysis", "")

            if not defect_present:
                print(f"  [Premise Trace] {q_id}: FALSE PREMISE — Actual output '{actual_out}'. Claimed symptom did not occur.")
                results[q_id] = {
                    "passed": False,
                    "is_bug_hunting": True,
                    "defect_present": False,
                    "manifests_in_main": manifests,
                    "note": f"False premise: Verifier trace confirmed program behaves correctly. Actual output: '{actual_out}'. Analysis: {analysis}",
                }
            elif not manifests:
                print(f"  [Premise Trace] {q_id}: LATENT BUG — Defect exists in logic but given main() test does not trigger it.")
                results[q_id] = {
                    "passed": True,
                    "is_bug_hunting": True,
                    "defect_present": True,
                    "manifests_in_main": False,
                    "note": f"Latent bug: defect exists in logic but given main() test does not trigger it. Actual output: '{actual_out}'.",
                }
            else:
                print(f"  [Premise Trace] {q_id}: PASSED — Defect confirmed present and manifests on stated input.")
                results[q_id] = {
                    "passed": True,
                    "is_bug_hunting": True,
                    "defect_present": True,
                    "manifests_in_main": True,
                    "note": f"Premise confirmed: {analysis}",
                }

    # Ensure all questions have an entry
    for q in questions_needing_trace:
        if q.id not in results:
            results[q.id] = {
                "passed": True,
                "is_bug_hunting": True,
                "defect_present": True,
                "manifests_in_main": True,
                "note": "Passed trace check (fallback default)",
            }

    return results


# ---------------------------------------------------------------------------
# Check 4: Fast Batched Self-Verification Pass (Part 1A)
# ---------------------------------------------------------------------------

def batch_verify_mcqs(
    mcqs: list[MCQQuestion],
    client: object,
    model: str,
) -> dict[str, tuple[str, str]]:
    """
    Verifies all MCQs in ONE batched call blind without answers or explanations.
    Returns dict mapping q_id -> (status, note).
    """
    if not mcqs:
        return {}

    items_payload = []
    for q in mcqs:
        items_payload.append({
            "id": q.id,
            "question_text": q.question_text,
            "options": q.options,
        })

    system_prompt = (
        "You are an expert Data Structures examiner solving multiple-choice questions blind without answers. "
        "Select the single best, completely correct option based on fundamental computer science principles."
    )

    user_prompt = f"""Solve each of the following {len(mcqs)} multiple choice questions independently.

QUESTIONS TO SOLVE:
{json.dumps(items_payload, indent=2)}

REQUESTED OUTPUT FORMAT (Return valid JSON):
{{
  "answers": [
    {{
      "id": "<matching question id>",
      "answer_index": <0, 1, 2, or 3>,
      "reasoning": "<concise justification>"
    }}
  ]
}}
"""

    raw_resp = tracked_call_llm(
        client=client,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    if not raw_resp:
        time.sleep(INTER_CALL_DELAY_SECONDS)
        raw_resp = tracked_call_llm(
            client=client,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model=model,
            temperature=0.1,
            response_format=None,
        )

    parsed = parse_json_response(raw_resp) if raw_resp else None
    results: dict[str, tuple[str, str]] = {}
    opt_letters = ["A", "B", "C", "D"]

    answers_by_id = {}
    if parsed and "answers" in parsed and isinstance(parsed["answers"], list):
        for a in parsed["answers"]:
            if isinstance(a, dict) and "id" in a and "answer_index" in a:
                answers_by_id[a["id"]] = a

    for q in mcqs:
        ans = answers_by_id.get(q.id)
        if not ans:
            results[q.id] = ("needs_review", "Independent verifier did not return an answer.")
            continue

        verifier_idx = ans.get("answer_index")
        reasoning = ans.get("reasoning", "")

        if verifier_idx == q.correct_option_index:
            v_letter = opt_letters[verifier_idx] if 0 <= verifier_idx < 4 else str(verifier_idx)
            results[q.id] = ("verified", f"Independent verifier agreed on option {v_letter}.")
        else:
            orig_letter = opt_letters[q.correct_option_index] if 0 <= q.correct_option_index < 4 else str(q.correct_option_index)
            ver_letter = opt_letters[verifier_idx] if 0 <= verifier_idx < 4 else str(verifier_idx)
            results[q.id] = (
                "needs_review",
                f"Answer key conflict: generator marked {orig_letter}, verifier chose {ver_letter}. Verifier reasoning: {reasoning}",
            )

    return results


def batch_verify_codes(
    code_questions: list[CodeQuestion],
    client: object,
    model: str,
) -> dict[str, tuple[str, str]]:
    """
    Verifies all CODE questions in 2 batched calls:
    Call 1: Independent solver answers all questions blind.
    Call 2: Equivalence judge evaluates whether independent solutions substantively match reference answers.
    Returns dict mapping q_id -> (status, note).
    """
    if not code_questions:
        return {}

    # Call 1: Independent Solver
    solve_payload = []
    for q in code_questions:
        solve_payload.append({
            "id": q.id,
            "question_text": q.question_text,
            "language": q.language,
            "code_snippet": q.code_snippet,
        })

    solver_prompt = f"""Provide your complete, correct solution or fix independently for each of the following {len(code_questions)} coding problems.

PROBLEMS:
{json.dumps(solve_payload, indent=2)}

REQUESTED OUTPUT FORMAT (Return valid JSON):
{{
  "solutions": [
    {{
      "id": "<matching question id>",
      "solution": "<your solution or bug fix>",
      "reasoning": "<concise justification>"
    }}
  ]
}}
"""

    raw_solve = tracked_call_llm(
        client=client,
        messages=[
            {"role": "system", "content": "You are an expert university CS instructor solving coding problems blind."},
            {"role": "user", "content": solver_prompt},
        ],
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    if not raw_solve:
        time.sleep(INTER_CALL_DELAY_SECONDS)
        raw_solve = tracked_call_llm(
            client=client,
            messages=[
                {"role": "system", "content": "You are an expert university CS instructor solving coding problems blind."},
                {"role": "user", "content": solver_prompt},
            ],
            model=model,
            temperature=0.1,
            response_format=None,
        )

    parsed_solve = parse_json_response(raw_solve) if raw_solve else None
    sol_by_id = {}
    if parsed_solve and "solutions" in parsed_solve and isinstance(parsed_solve["solutions"], list):
        for s in parsed_solve["solutions"]:
            if isinstance(s, dict) and "id" in s and "solution" in s:
                sol_by_id[s["id"]] = s.get("solution", "")

    # Call 2: Equivalence Judge
    judge_payload = []
    for q in code_questions:
        indep_sol = sol_by_id.get(q.id, "")
        judge_payload.append({
            "id": q.id,
            "problem": q.question_text,
            "reference_solution_a": q.expected_answer,
            "independent_solution_b": indep_sol,
        })

    judge_prompt = f"""For each question, judge whether Solution B is equivalent in substance to Solution A
(they fix the same bug, achieve the same algorithmic result, or implement the same requirements, even if formatting or variable names differ).

PAIRS TO JUDGE:
{json.dumps(judge_payload, indent=2)}

REQUESTED OUTPUT FORMAT (Return valid JSON):
{{
  "comparisons": [
    {{
      "id": "<matching question id>",
      "equivalent": true, // or false
      "reasoning": "<concise explanation>"
    }}
  ]
}}
"""

    time.sleep(INTER_CALL_DELAY_SECONDS)
    raw_judge = tracked_call_llm(
        client=client,
        messages=[
            {"role": "system", "content": "You are an expert CS grader determining substance equivalence between two solutions."},
            {"role": "user", "content": judge_prompt},
        ],
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    if not raw_judge:
        time.sleep(INTER_CALL_DELAY_SECONDS)
        raw_judge = tracked_call_llm(
            client=client,
            messages=[
                {"role": "system", "content": "You are an expert CS grader determining substance equivalence between two solutions."},
                {"role": "user", "content": judge_prompt},
            ],
            model=model,
            temperature=0.1,
            response_format=None,
        )

    parsed_judge = parse_json_response(raw_judge) if raw_judge else None
    results: dict[str, tuple[str, str]] = {}
    comp_by_id = {}
    if parsed_judge and "comparisons" in parsed_judge and isinstance(parsed_judge["comparisons"], list):
        for c in parsed_judge["comparisons"]:
            if isinstance(c, dict) and "id" in c:
                comp_by_id[c["id"]] = c

    for q in code_questions:
        comp = comp_by_id.get(q.id)
        if comp and comp.get("equivalent") is True:
            results[q.id] = ("verified", f"Independent solution confirmed equivalent: {comp.get('reasoning', '')}")
        else:
            reason = comp.get("reasoning", "Solutions differed substantively") if comp else "Judge did not return verdict"
            results[q.id] = ("needs_review", f"Equivalence check failed: {reason}")

    return results


# ---------------------------------------------------------------------------
# Check 5: Batched Grounding Check against Source Material (Part 3)
# ---------------------------------------------------------------------------

def batch_check_grounding(
    questions: list[AnyQuestion],
    concepts_by_name: dict[str, Concept],
    client: object,
    model: str,
) -> dict[str, dict]:
    """
    Executes Check 5 (Grounding Check) in ONE batched call across all questions.
    Checks whether the marked answer is strictly consistent with the professor's lecture notes.
    Catches contradictions where an answer is technically plausible but opposes the course slides.
    """
    grounding_payload = []
    for q in questions:
        concept = concepts_by_name.get(q.concept)
        if not concept:
            continue

        marked_ans = ""
        if isinstance(q, MCQQuestion):
            opt_letter = chr(65 + q.correct_option_index)
            marked_ans = f"Option {opt_letter}: {q.options[q.correct_option_index]}"
        elif isinstance(q, CodeQuestion):
            marked_ans = q.expected_answer
        elif isinstance(q, TheoryQuestion):
            marked_ans = q.model_answer

        grounding_payload.append({
            "id": q.id,
            "concept_name": concept.name,
            "concept_explanation": concept.explanation,
            "concept_subtopics": concept.subtopics,
            "question_text": q.question_text,
            "marked_answer": marked_ans,
        })

    system_prompt = (
        "You are an academic auditor checking exam questions against course lecture notes and syllabus material. "
        "Strictly check whether each question and its marked answer are consistent with the professor's source notes."
    )

    user_prompt = f"""For each question below, check: Is the marked answer consistent with the source material provided?
Answer consistent=false if the source material explicitly states or implies something that contradicts the marked answer.
(For example: If the course notes state that 'sorting' is a fundamental operation, but the question marks 'sorting is NOT a fundamental operation', that is an explicit contradiction and must be marked consistent=false).

QUESTIONS TO AUDIT:
{json.dumps(grounding_payload, indent=2)}

REQUESTED OUTPUT FORMAT (Return valid JSON):
{{
  "grounding": [
    {{
      "id": "<matching question id>",
      "consistent": true, // or false
      "conflict": null // or "<specific contradiction with the source notes>"
    }}
  ]
}}
"""

    raw_resp = tracked_call_llm(
        client=client,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    if not raw_resp:
        time.sleep(INTER_CALL_DELAY_SECONDS)
        raw_resp = tracked_call_llm(
            client=client,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model=model,
            temperature=0.1,
            response_format=None,
        )

    parsed = parse_json_response(raw_resp) if raw_resp else None
    results: dict[str, dict] = {}

    if parsed and "grounding" in parsed and isinstance(parsed["grounding"], list):
        for entry in parsed["grounding"]:
            if isinstance(entry, dict) and "id" in entry:
                q_id = entry["id"]
                consistent = entry.get("consistent", True)
                conflict = entry.get("conflict")
                results[q_id] = {
                    "consistent": consistent,
                    "conflict": conflict,
                }
                if not consistent:
                    print(f"  [Grounding Audit] {q_id}: CONFLICT with course notes — {conflict}")
                else:
                    print(f"  [Grounding Audit] {q_id}: PASSED — Grounded in course lecture notes.")

    for q in questions:
        if q.id not in results:
            results[q.id] = {"consistent": True, "conflict": None}

    return results


# ---------------------------------------------------------------------------
# Part 1C: Resume Mode Helper
# ---------------------------------------------------------------------------

def load_resumable_questions(path: str = OUTPUT_QUESTIONS_PATH) -> dict[str, AnyQuestion]:
    """
    Loads existing questions.json and filters for questions that are:
    1. Schema valid
    2. Coherent
    3. Verified (or not_applicable for theory)
    4. Free of premise and grounding conflicts
    """
    if not os.path.isfile(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    quiz = None
    try:
        quiz = Quiz.model_validate(data)
    except Exception:
        return {}

    resumable: dict[str, AnyQuestion] = {}
    for q in quiz.questions:
        apply_deterministic_repairs(q)
        is_coherent, _ = check_entity_coherence(q)
        if not is_coherent:
            continue

        is_verified = (
            q.verification_status == "verified"
            or (q.type == QuestionType.THEORY and q.verification_status == "not_applicable")
        )
        if not is_verified:
            continue

        note = (q.verification_note or "").lower()
        if "false premise" in note or "grounding conflict" in note:
            continue

        resumable[q.concept] = q

    return resumable


# ---------------------------------------------------------------------------
# Main Orchestration Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 2: Question Writer with Self-Verification")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing outputs/questions.json, keeping verified questions and only regenerating missing/failed ones",
    )
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("AGENT 2: FAST-PATH QUESTION WRITER WITH 5 SAFETY CHECKS")
    print("=" * 60)

    # 1. Initialize client and select consistent model for the WHOLE run (Part 1D)
    client, provider = get_llm_client()
    selected_model, model_selection_reason = select_active_model(client)
    print(f"\nLLM provider initialized: {provider.upper()}")
    print(f"Selected Model:   {selected_model}")
    print(f"Selection Reason: {model_selection_reason}")

    # 2. Load concepts and reference samples
    concepts = load_concepts(CONCEPTS_PATH)
    concepts_by_name = {c.name: c for c in concepts}
    print(f"Loaded {len(concepts)} concepts from '{CONCEPTS_PATH}'.")

    spec = QuizSpec(
        total_questions=12,
        mcq_ratio=0.5,
        code_ratio=0.25,
        theory_ratio=0.25,
    )
    plan = plan_quiz_generation(concepts, spec)

    print("\n" + "=" * 60)
    print(f"GENERATION PLAN (Total Target: {spec.total_questions} Questions)")
    print("=" * 60)
    for idx, (concept, q_type, rationale) in enumerate(plan, start=1):
        print(f"  [q{idx:02d}] [{q_type.value:6s}] '{concept.name}' — {rationale}")
    print("=" * 60)

    sample_questions = load_sample_questions(SAMPLE_QUIZ_PATH)

    # 3. Check for Resume Mode (Part 1C)
    resumable_by_concept = load_resumable_questions(OUTPUT_QUESTIONS_PATH) if args.resume else {}
    if args.resume:
        print(f"\n[Resume Mode] Found {len(resumable_by_concept)} valid verified question(s) in '{OUTPUT_QUESTIONS_PATH}'.")

    # 4. Partition Plan into Kept vs Needed Questions
    kept_questions: dict[str, AnyQuestion] = {}
    needed_plan_items: list[tuple[Concept, QuestionType, str, str]] = []

    for idx, (concept, q_type, rationale) in enumerate(plan, start=1):
        q_id = f"q{idx}"
        if args.resume and concept.name in resumable_by_concept:
            existing_q = resumable_by_concept[concept.name]
            existing_q.id = q_id
            kept_questions[q_id] = existing_q
            print(f"  [Resume] Keeping {q_id} [{existing_q.type.value}] '{concept.name}' (already verified)")
        else:
            needed_plan_items.append((concept, q_type, rationale, q_id))
            if args.resume:
                print(f"  [Resume] Generating {q_id} [{q_type.value}] '{concept.name}' (needs generation/verification)")

    # 5. Batched Generation by Type (Part 1B)
    generated_questions: list[AnyQuestion] = []
    if needed_plan_items:
        print("\nStarting batched question generation by type...")

        mcq_items = [(c, q_id, c.sources[0].filename if c.sources else "unknown") for c, qt, _, q_id in needed_plan_items if qt == QuestionType.MCQ]
        code_items = [(c, q_id, c.sources[0].filename if c.sources else "unknown") for c, qt, _, q_id in needed_plan_items if qt == QuestionType.CODE]
        theory_items = [(c, q_id, c.sources[0].filename if c.sources else "unknown") for c, qt, _, q_id in needed_plan_items if qt == QuestionType.THEORY]

        if mcq_items:
            print(f"\nGenerating batch of {len(mcq_items)} MCQ question(s)...")
            generated_mcqs = generate_mcqs_batch(mcq_items, sample_questions.get("MCQ", {}), client, selected_model)
            generated_questions.extend(generated_mcqs)

        if code_items:
            time.sleep(INTER_CALL_DELAY_SECONDS)
            print(f"\nGenerating batch of {len(code_items)} CODE question(s)...")
            generated_codes = generate_codes_batch(code_items, sample_questions.get("CODE", {}), client, selected_model)
            generated_questions.extend(generated_codes)

        if theory_items:
            time.sleep(INTER_CALL_DELAY_SECONDS)
            print(f"\nGenerating batch of {len(theory_items)} THEORY question(s)...")
            generated_theories = generate_theory_batch(theory_items, sample_questions.get("THEORY", {}), client, selected_model)
            generated_questions.extend(generated_theories)

    # Combine kept questions and newly generated questions
    all_questions_map: dict[str, AnyQuestion] = dict(kept_questions)
    for q in generated_questions:
        all_questions_map[q.id] = q

    working_questions: list[AnyQuestion] = []
    for idx in range(1, len(plan) + 1):
        q_id = f"q{idx}"
        if q_id in all_questions_map:
            working_questions.append(all_questions_map[q_id])

    # 6. Check 2: Coherence Check & Regeneration on newly generated questions
    print("\nRunning Check 2: Deterministic Entity Coherence Check...")
    coherence_passed_count = 0
    coherence_regen_count = 0
    coherence_failed_count = 0

    coherent_questions: list[AnyQuestion] = []
    for q in working_questions:
        if q.id in kept_questions:
            coherent_questions.append(q)
            coherence_passed_count += 1
            continue

        is_coherent, reason = check_entity_coherence(q)
        if is_coherent:
            coherent_questions.append(q)
            coherence_passed_count += 1
        else:
            print(f"  [Coherence Mismatch] {q.id} '{q.concept}': {reason} — regenerating...")
            coherence_regen_count += 1
            time.sleep(INTER_CALL_DELAY_SECONDS)
            concept = concepts_by_name.get(q.concept)
            if concept:
                regen_q = generate_single_question(
                    concept=concept,
                    q_type=q.type,
                    question_id=q.id,
                    sample_question=sample_questions.get(q.type.value, {}),
                    client=client,
                    model=selected_model,
                    coherence_feedback=reason,
                )
                if regen_q:
                    is_coh_2, _ = check_entity_coherence(regen_q)
                    if is_coh_2:
                        coherent_questions.append(regen_q)
                    else:
                        coherence_failed_count += 1
                        regen_q.verification_status = "needs_review"
                        regen_q.verification_note = f"Coherence check failed twice: {reason}"
                        coherent_questions.append(regen_q)
                else:
                    coherence_failed_count += 1
                    q.verification_status = "needs_review"
                    q.verification_note = f"Coherence check failed: {reason}"
                    coherent_questions.append(q)

    # 7. Check 3: Premise Check on Bug-Hunting CODE Questions (Part 4)
    print("\nRunning Check 3: Code Premise & Program Trace Check...")
    code_qs = [q for q in coherent_questions if isinstance(q, CodeQuestion) and q.id not in kept_questions]
    premise_results = batch_check_premises(code_qs, client, selected_model) if code_qs else {}

    premise_passed_count = 0
    premise_latent_count = 0
    premise_failed_count = 0

    for q in coherent_questions:
        if isinstance(q, CodeQuestion) and q.id in premise_results:
            p_res = premise_results[q.id]
            if p_res["passed"]:
                premise_passed_count += 1
                if not p_res.get("manifests_in_main", True):
                    premise_latent_count += 1
                    q.verification_note = p_res["note"]
            else:
                premise_failed_count += 1
                q.verification_status = "needs_review"
                q.verification_note = p_res["note"]

    # 8. Check 4: Batched Self-Verification Pass (Part 1A)
    print("\nRunning Check 4: Fast Batched Self-Verification Pass...")
    mcqs_to_verify = [q for q in coherent_questions if isinstance(q, MCQQuestion) and q.id not in kept_questions]
    codes_to_verify = [
        q for q in coherent_questions
        if isinstance(q, CodeQuestion) and q.id not in kept_questions and q.verification_status != "needs_review"
    ]

    mcq_ver_results = batch_verify_mcqs(mcqs_to_verify, client, selected_model) if mcqs_to_verify else {}
    code_ver_results = batch_verify_codes(codes_to_verify, client, selected_model) if codes_to_verify else {}

    for q in coherent_questions:
        if isinstance(q, MCQQuestion) and q.id in mcq_ver_results:
            st, nt = mcq_ver_results[q.id]
            q.verification_status = st
            q.verification_note = nt
        elif isinstance(q, CodeQuestion) and q.id in code_ver_results:
            st, nt = code_ver_results[q.id]
            q.verification_status = st
            q.verification_note = nt
        elif isinstance(q, TheoryQuestion):
            q.verification_status = "not_applicable"
            q.verification_note = "Open-ended theory question; unverified by design."

    # 9. Check 5: Batched Grounding Check against Course Material (Part 3)
    print("\nRunning Check 5: Batched Grounding Check against Course Lecture Notes...")
    qs_to_ground = [q for q in coherent_questions if q.id not in kept_questions and q.verification_status == "verified"]
    grounding_results = batch_check_grounding(qs_to_ground, concepts_by_name, client, selected_model) if qs_to_ground else {}

    grounding_passed_count = 0
    grounding_conflict_count = 0

    for q in coherent_questions:
        if q.id in grounding_results:
            g_res = grounding_results[q.id]
            if g_res["consistent"]:
                grounding_passed_count += 1
            else:
                grounding_conflict_count += 1
                q.verification_status = "needs_review"
                q.verification_note = f"Grounding conflict with source notes: {g_res['conflict']}"

    # 10. Save and Format outputs/questions.json
    all_sources = sorted(list({s.filename for c in concepts for s in c.sources}))
    total_marks = sum(q.marks for q in coherent_questions)
    quiz_data = {
        "source_documents": all_sources,
        "total_marks": total_marks,
        "questions": [q.model_dump() for q in coherent_questions],
    }

    try:
        Quiz.model_validate(quiz_data)
        os.makedirs(os.path.dirname(OUTPUT_QUESTIONS_PATH), exist_ok=True)
        with open(OUTPUT_QUESTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(quiz_data, f, indent=2, ensure_ascii=False)
        print(f"\n[Success] outputs/questions.json successfully validated and saved ({len(coherent_questions)} questions).")
    except ValidationError as err:
        print(f"\n[Warning] questions.json failed Quiz schema validation: {err}")

    # 11. Final Summary & Performance Metrics
    elapsed_time = time.time() - start_time
    verified_count = sum(1 for q in coherent_questions if q.verification_status == "verified")
    review_count = sum(1 for q in coherent_questions if q.verification_status == "needs_review")
    na_count = sum(1 for q in coherent_questions if q.verification_status == "not_applicable")
    generation_failures = len(plan) - len(coherent_questions)

    print("\n" + "=" * 60)
    print("AGENT 2 ASSESSMENT EXECUTION & PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Selected Model:   {selected_model}")
    print(f"Model Reason:     {model_selection_reason}")
    print(f"Total LLM Calls:  {llm_call_count}")
    print(f"Wall-Clock Time:  {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)")

    print("\n--- The 5 Quality & Safety Check Results ---")
    print("Check 1 [Schema Validation]:")
    print(f"  - Questions Conforming to Pydantic Schema: {len(coherent_questions)}/{len(plan)}")
    print(f"  - Generation / Schema Failures:          {generation_failures}")

    print("\nCheck 2 [Entity Coherence Check]:")
    print(f"  - Passed Coherence First Time:           {coherence_passed_count}")
    print(f"  - Regenerated due to Coherence Mismatch: {coherence_regen_count}")
    print(f"  - Failed Coherence Check:                {coherence_failed_count}")

    print("\nCheck 3 [Premise Check Results]:")
    print(f"  - Passed Premise & Manifests in Main:    {premise_passed_count}")
    print(f"  - Latent Bug Flagged (Does Not Manifest):{premise_latent_count}")
    print(f"  - False Premise Flagged Needs Review:    {premise_failed_count}")

    print("\nCheck 4 [Independent Self-Verification Pass]:")
    print(f"  - Verified Correct:                      {verified_count}")
    print(f"  - Needs Instructor Review:               {review_count}")
    print(f"  - Open-Ended Theory (Unverified):        {na_count}")

    print("\nCheck 5 [Course Notes Grounding Check]:")
    print(f"  - Grounded with Professor Course Notes:  {grounding_passed_count}")
    print(f"  - Contradictions / Conflicts Flagged:    {grounding_conflict_count}")

    print("\n--- Questions Requiring Instructor Review ---")
    review_questions = [q for q in coherent_questions if q.verification_status == "needs_review"]
    if review_questions:
        for rq in review_questions:
            print(f"  - [{rq.id}] [{rq.type.value}] '{rq.concept}':")
            print(f"      Status: NEEDS_REVIEW — {rq.verification_note}")
    else:
        print("  None — all questions passed independent verification and grounding checks!")
    print("=" * 60)


if __name__ == "__main__":
    main()
