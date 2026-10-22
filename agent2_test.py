"""
agent2_test.py — Minimal Prototype for Agent 2 (Question Writer).

Loads extracted concepts from outputs/concepts.json, selects the first three,
and queries the LLM to generate exactly one high-quality, schema-valid Multiple Choice
Question (MCQ) per concept. Enforces Pydantic schema validation with automatic retry,
quality rules, and exam-paper formatting.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

from pydantic import ValidationError

from schemas import Concept, MCQQuestion
from step1_test import (
    MODEL_NAME,
    call_llm_with_retry,
    get_llm_client,
    parse_json_response,
)

CONCEPTS_PATH = "outputs/concepts.json"
SAMPLE_QUIZ_PATH = "outputs/sample_quiz.json"
INTER_CALL_DELAY_SECONDS = 3.0


def load_concepts(path: str = CONCEPTS_PATH) -> list[Concept]:
    """
    Loads and validates concepts from the concepts.json file.
    If the file is missing or contains no concepts, prints an informative message and exits.
    """
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

    try:
        return [Concept.model_validate(c) for c in raw_concepts]
    except ValidationError as err:
        print(f"\n[Error] Concepts file '{path}' failed schema validation:\n{err}")
        sys.exit(1)


def load_sample_mcq(path: str = SAMPLE_QUIZ_PATH) -> dict:
    """
    Loads the worked MCQ question example from sample_quiz.json to include in the prompt.
    Provides a hardcoded fallback if the file is missing or does not contain an MCQ.
    """
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                quiz_data = json.load(f)
                for q in quiz_data.get("questions", []):
                    if q.get("type") == "MCQ":
                        return q
        except Exception as err:
            print(f"[Warning] Could not load sample quiz from '{path}': {err}")

    # Fallback example matching MCQQuestion in schemas.py
    return {
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
        "explanation": (
            "A stack is a LIFO structure: the last element pushed onto the top is always the first "
            "to be popped off. This distinguishes it from a queue (FIFO) or an array (random access)."
        ),
    }


def build_mcq_prompt(
    concept: Concept,
    sample_mcq: dict,
    question_id: str,
    error_feedback: Optional[str] = None,
) -> list[dict]:
    """
    Constructs the prompt messages (system and user) for generating a multiple-choice question.
    Incorporates the concept's name, explanation, subtopics, source file, the sample MCQ,
    and strict pedagogical quality guidelines.
    """
    primary_source = concept.sources[0].filename if concept.sources else "unknown"
    subtopics_text = ", ".join(concept.subtopics) if concept.subtopics else "None"
    sample_example_json = json.dumps(sample_mcq, indent=2)

    system_prompt = (
        "You are an experienced university examiner setting questions for an undergraduate "
        "Data Structures course. Your questions test deep conceptual understanding and "
        "practical reasoning rather than rote vocabulary recall."
    )

    user_prompt = f"""Generate exactly ONE Multiple Choice Question (MCQ) testing the following concept:

CONCEPT DETAILS:
- Name: {concept.name}
- Explanation: {concept.explanation}
- Subtopics: {subtopics_text}
- Source Document: {primary_source}

WORKED EXAMPLE (Follow this exact schema and high standard):
{sample_example_json}

REQUESTED OUTPUT SHAPE (Return a valid JSON object matching this schema):
{{
  "id": "{question_id}",
  "concept": "{concept.name}",
  "source_file": "{primary_source}",
  "type": "MCQ",
  "difficulty": "MEDIUM",
  "marks": 2,
  "question_text": "<Clear, unambiguous question scenario or problem>",
  "options": [
    "<Plausible Option A>",
    "<Plausible Option B>",
    "<Plausible Option C>",
    "<Plausible Option D>"
  ],
  "correct_option_index": <0, 1, 2, or 3>,
  "explanation": "<Why the correct option is right AND briefly why the most tempting wrong option is wrong>"
}}

QUESTION QUALITY RULES (Strict Requirements):
1. The question must be answerable ONLY by someone who understands the concept. Do not write questions answerable from the wording of the options alone.
2. All four options must be PLAUSIBLE to someone who has not studied the topic. Wrong options should represent realistic misunderstandings, not obvious nonsense.
3. All four options should be similar in length and level of detail. A noticeably longer or more detailed option gives the answer away.
4. Do NOT use "All of the above" or "None of the above".
5. The explanation must say why the correct option is right AND briefly why the most tempting wrong option is wrong.
6. Test understanding, not vocabulary recall. Prefer "which of these scenarios requires X" or conceptual behavior analysis over "what does X stand for".
"""

    if error_feedback:
        user_prompt += f"""
PREVIOUS ATTEMPT FAILED VALIDATION:
Your previous response failed Pydantic schema validation with the following error:
{error_feedback}

Please fix the error and output a valid JSON object strictly complying with the schema and quality rules.
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_mcq_for_concept(
    concept: Concept,
    question_id: str,
    sample_mcq: dict,
    client: object,
    model: str = MODEL_NAME,
) -> Optional[MCQQuestion]:
    """
    Calls the LLM to generate an MCQ for a concept, validates the output with MCQQuestion,
    and retries once if validation fails, passing the error message back to the LLM.
    """
    primary_source = concept.sources[0].filename if concept.sources else "unknown"
    error_feedback: Optional[str] = None

    for attempt in range(1, 3):  # Attempt 1 (initial), Attempt 2 (retry on error)
        if attempt > 1:
            print(f"  -> Retrying concept '{concept.name}' with validation feedback (attempt 2/2)...")
            time.sleep(INTER_CALL_DELAY_SECONDS)

        messages = build_mcq_prompt(
            concept=concept,
            sample_mcq=sample_mcq,
            question_id=question_id,
            error_feedback=error_feedback,
        )

        raw_response = call_llm_with_retry(
            client=client,
            messages=messages,
            model=model,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        if not raw_response:
            error_feedback = "LLM returned an empty response."
            print(f"  [Attempt {attempt}] Failed: No response received from LLM.")
            continue

        parsed_data = parse_json_response(raw_response)
        if not parsed_data:
            error_feedback = "Response could not be parsed into valid JSON."
            print(f"  [Attempt {attempt}] Failed: LLM response was not valid JSON.")
            continue

        # Defensively ensure required fields are set if omitted by model
        if not parsed_data.get("source_file"):
            parsed_data["source_file"] = primary_source
        if not parsed_data.get("id"):
            parsed_data["id"] = question_id
        if not parsed_data.get("concept"):
            parsed_data["concept"] = concept.name
        if not parsed_data.get("type"):
            parsed_data["type"] = "MCQ"
        if not parsed_data.get("difficulty"):
            parsed_data["difficulty"] = "MEDIUM"
        if "marks" not in parsed_data:
            parsed_data["marks"] = 2

        # Validate against Pydantic schema
        try:
            validated_question = MCQQuestion.model_validate(parsed_data)
            return validated_question
        except ValidationError as val_err:
            error_feedback = str(val_err)
            print(f"  [Attempt {attempt}] Validation error for '{concept.name}':\n{val_err}")

    print(f"  [Failed] Could not generate a valid question for '{concept.name}' after 2 attempts.")
    return None


def print_exam_question(index: int, mcq: MCQQuestion) -> None:
    """
    Renders an MCQ in clean exam-paper format.
    """
    option_letters = ["A", "B", "C", "D"]
    correct_letter = (
        option_letters[mcq.correct_option_index]
        if 0 <= mcq.correct_option_index < len(option_letters)
        else str(mcq.correct_option_index)
    )

    print(f"\n    [{index}] Concept: {mcq.concept}")
    print(f"        Q: {mcq.question_text}")
    for i, opt in enumerate(mcq.options):
        letter = option_letters[i] if i < len(option_letters) else str(i)
        print(f"           {letter}) {opt}")
    print(f"        Correct: {correct_letter}")
    print(f"        Why: {mcq.explanation}")


def main() -> None:
    print("=" * 60)
    print("AGENT 2 MINIMAL SLICE: QUESTION GENERATOR TEST")
    print("=" * 60)

    # 1. Load concepts
    all_concepts = load_concepts(CONCEPTS_PATH)
    print(f"Loaded {len(all_concepts)} concept(s) from '{CONCEPTS_PATH}'.")

    # 2. Select first 3 concepts only
    selected_concepts = all_concepts[:3]
    print(f"Selected first {len(selected_concepts)} concept(s) for test generation:")
    for idx, c in enumerate(selected_concepts, start=1):
        sub_info = f" ({len(c.subtopics)} subtopic(s))" if c.subtopics else ""
        print(f"  {idx}. {c.name}{sub_info}")

    # 3. Load sample quiz reference
    sample_mcq = load_sample_mcq(SAMPLE_QUIZ_PATH)

    # 4. Initialize LLM client
    client, provider = get_llm_client()
    print(f"\nLLM provider initialized ({provider.upper()} / {MODEL_NAME}).")

    generated_questions: list[MCQQuestion] = []
    failed_concepts: list[str] = []

    # 5. Generate one MCQ per concept
    print("\nStarting question generation (1 question per concept, 3s delay)...")
    for idx, concept in enumerate(selected_concepts, start=1):
        print(f"\nGenerating question {idx}/{len(selected_concepts)} for '{concept.name}'...")
        question = generate_mcq_for_concept(
            concept=concept,
            question_id=f"q{idx}",
            sample_mcq=sample_mcq,
            client=client,
            model=MODEL_NAME,
        )

        if question is not None:
            generated_questions.append(question)
            print(f"  -> Successfully generated and validated question {idx}.")
        else:
            failed_concepts.append(concept.name)

        # 3-second delay between calls to respect rate limits
        if idx < len(selected_concepts):
            time.sleep(INTER_CALL_DELAY_SECONDS)

    # 6. Display generated questions in exam-paper format
    print("\n" + "=" * 60)
    print("EXAM-PAPER FORMAT QUESTIONS")
    print("=" * 60)
    if generated_questions:
        for idx, q in enumerate(generated_questions, start=1):
            print_exam_question(idx, q)
    else:
        print("No questions were generated successfully.")

    # 7. Final Summary
    total_requested = len(selected_concepts)
    total_success = len(generated_questions)
    total_failed = len(failed_concepts)

    print("\n" + "=" * 60)
    print(
        f"Summary: {total_requested} requested, "
        f"{total_success} validated successfully, "
        f"{total_failed} failed."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
