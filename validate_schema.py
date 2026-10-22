"""
validate_schema.py — Schema Validation Test for Step 2 & Step 3

Proves that the Quiz / Question Pydantic models work correctly in two ways:
  1. A valid sample quiz loads and parses without errors.
  2. Three deliberately broken inputs each raise a clear ValidationError.

Run this any time schemas.py changes to confirm the data contract is intact.
"""

import json
from pathlib import Path

from pydantic import ValidationError

from schemas import DifficultyTarget, Quiz, QuizSpec


# ---------------------------------------------------------------------------
# Helper — pretty-print a ValidationError without overwhelming the terminal.
# We show each error's location (which field failed) and the reason.
# ---------------------------------------------------------------------------

def print_validation_errors(err: ValidationError) -> None:
    for e in err.errors():
        # loc is a tuple like ('questions', 0, 'options') — join for readability
        location = " → ".join(str(part) for part in e["loc"])
        print(f"   Field : {location}")
        print(f"   Error : {e['msg']}")
        print()


# ---------------------------------------------------------------------------
# PART 1 — Load the hand-crafted sample quiz and parse it with Quiz model.
# This proves valid data passes through cleanly.
# ---------------------------------------------------------------------------

print("=" * 60)
print("PART 1: Parsing valid sample_quiz.json")
print("=" * 60)

sample_path = Path("outputs/sample_quiz.json")

try:
    raw = json.loads(sample_path.read_text(encoding="utf-8"))
    quiz = Quiz.model_validate(raw)
    print(f"✅  Success! Parsed {len(quiz.questions)} questions, "
          f"{quiz.total_marks} total marks from {len(quiz.source_documents)} source doc(s).")
    for q in quiz.questions:
        print(f"    [{q.type.value:6s}] {q.id} — {q.concept} ({q.difficulty.value}, {q.marks}m, file: {q.source_file})")
except ValidationError as e:
    # If this fires, our sample_quiz.json itself has a bug — surface it clearly
    print("❌  sample_quiz.json failed validation (fix the sample file):")
    print_validation_errors(e)


# ---------------------------------------------------------------------------
# PART 2 — Three invalid payloads that must each be rejected.
# We catch ValidationError and print the reason so it's clear the guard works.
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("PART 2: Catching expected validation errors")
print("=" * 60)

# ------------------------------------------------------------------
# Invalid case 1: MCQ with only 3 options (must have exactly 4).
# ------------------------------------------------------------------
print("\n[Test 1] MCQ with 3 options instead of 4 ...")

bad_mcq = {
    "source_documents": ["test.pdf"],
    "total_marks": 2,
    "questions": [
        {
            "id": "q1",
            "concept": "Stack (LIFO) ADT",
            "source_file": "test.pdf",
            "type": "MCQ",
            "difficulty": "EASY",
            "marks": 2,
            "question_text": "Which describes a stack?",
            "options": ["FIFO", "LIFO", "Random access"],   # only 3 — should fail
            "correct_option_index": 1,
            "explanation": "Stacks are LIFO."
        }
    ]
}

try:
    Quiz.model_validate(bad_mcq)
    print("   ❌  ERROR: Validation should have failed but didn't!")
except ValidationError as e:
    print("   ✅  Correctly rejected:")
    print_validation_errors(e)

# ------------------------------------------------------------------
# Invalid case 2: Question with marks = 0 (must be > 0).
# ------------------------------------------------------------------
print("[Test 2] Question with marks = 0 ...")

bad_marks = {
    "source_documents": ["test.pdf"],
    "total_marks": 0,
    "questions": [
        {
            "id": "q1",
            "concept": "Recursion",
            "source_file": "test.pdf",
            "type": "THEORY",
            "difficulty": "MEDIUM",
            "marks": 0,           # should fail — must be > 0
            "question_text": "Explain recursion.",
            "model_answer": "A function that calls itself.",
            "key_points": ["Base case defined", "Recursive case reduces problem"]
        }
    ]
}

try:
    Quiz.model_validate(bad_marks)
    print("   ❌  ERROR: Validation should have failed but didn't!")
except ValidationError as e:
    print("   ✅  Correctly rejected:")
    print_validation_errors(e)

# ------------------------------------------------------------------
# Invalid case 3: Difficulty set to "VERY_HARD" (not in enum).
# ------------------------------------------------------------------
print("[Test 3] Invalid difficulty value 'VERY_HARD' ...")

bad_difficulty = {
    "source_documents": ["test.pdf"],
    "total_marks": 5,
    "questions": [
        {
            "id": "q1",
            "concept": "Pointers",
            "source_file": "test.pdf",
            "type": "CODE",
            "difficulty": "VERY_HARD",   # not a valid Difficulty enum value
            "marks": 5,
            "question_text": "Write a C function that swaps two integers using pointers.",
            "language": "C",
            "code_snippet": None,
            "expected_answer": "void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }",
            "explanation": "Pointers allow the function to modify the original variables."
        }
    ]
}

try:
    Quiz.model_validate(bad_difficulty)
    print("   ❌  ERROR: Validation should have failed but didn't!")
except ValidationError as e:
    print("   ✅  Correctly rejected:")
    print_validation_errors(e)

# ------------------------------------------------------------------
# Invalid case 4: QuizSpec with ratios not summing to 1.0
# ------------------------------------------------------------------
print("[Test 4] Invalid QuizSpec ratios summing to 1.2 ...")

bad_spec = {
    "total_questions": 10,
    "mcq_ratio": 0.6,
    "code_ratio": 0.4,
    "theory_ratio": 0.2,
}

try:
    QuizSpec.model_validate(bad_spec)
    print("   ❌  ERROR: Validation should have failed but didn't!")
except ValidationError as e:
    print("   ✅  Correctly rejected:")
    print_validation_errors(e)

# ------------------------------------------------------------------
# Valid case: Default QuizSpec
# ------------------------------------------------------------------
valid_spec = QuizSpec(total_questions=12)
assert valid_spec.total_questions == 12
assert valid_spec.mcq_ratio == 0.5
print("✅  Valid QuizSpec(total_questions=12) passed correctly.")

# ------------------------------------------------------------------
# Invalid case 5: DifficultyTarget with ratios not summing to 1.0
# ------------------------------------------------------------------
print("\n[Test 5] Invalid DifficultyTarget ratios summing to 1.1 ...")

bad_target = {
    "easy_ratio": 0.4,
    "medium_ratio": 0.5,
    "hard_ratio": 0.2,
}

try:
    DifficultyTarget.model_validate(bad_target)
    print("   ❌  ERROR: Validation should have failed but didn't!")
except ValidationError as e:
    print("   ✅  Correctly rejected:")
    print_validation_errors(e)

# ------------------------------------------------------------------
# Valid case: Default DifficultyTarget
# ------------------------------------------------------------------
valid_target = DifficultyTarget()
assert valid_target.easy_ratio == 0.3
assert valid_target.medium_ratio == 0.5
assert valid_target.hard_ratio == 0.2
print("✅  Valid DifficultyTarget(0.3, 0.5, 0.2) passed correctly.")

print("=" * 60)
print("All validation tests complete.")
print("=" * 60)
