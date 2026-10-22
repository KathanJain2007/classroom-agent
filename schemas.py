"""
schemas.py — Question & Quiz Data Contract for the Multi-Agent Quiz Generator

This file defines the exact shape every question must have as it moves
between agents. Pydantic v2 enforces the rules automatically, so any
malformed LLM output fails with a clear error message here rather than
causing a mysterious crash later in the pipeline.

Agent 1 (Extractor)       produces Concept models with source traceability.
Agent 2 (Question Writer) produces questions matching these models.
Agent 3 (Examiner)        reads, modifies, and re-validates the same models.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums — only these exact values are allowed for type and difficulty.
# Using an Enum (instead of bare strings) means a typo like "Mcq" or
# "Medium" will be caught immediately rather than silently passing through.
# ---------------------------------------------------------------------------

class QuestionType(str, Enum):
    """The three question formats the pipeline supports."""
    MCQ    = "MCQ"
    CODE   = "CODE"
    THEORY = "THEORY"


class Difficulty(str, Enum):
    """Difficulty band assigned by the Question Writer agent."""
    EASY   = "EASY"
    MEDIUM = "MEDIUM"
    HARD   = "HARD"


# ---------------------------------------------------------------------------
# Concept Models — Schema for concepts extracted from documents with
# full traceability back to source files and page/block ranges.
# ---------------------------------------------------------------------------

class ConceptSource(BaseModel):
    """Identifies the file and page range where a concept was detected.

    block_start/block_end are ORIGINAL page (PDF) or paragraph (DOCX) numbers,
    translated from internal block positions via DocumentContent.block_page_map
    so a citation points at the page a reader would actually open.
    """
    filename: str = Field(..., description="Name of the source document file")
    block_start: int = Field(..., ge=1, description="Starting 1-based page/paragraph number")
    block_end: int = Field(..., ge=1, description="Ending 1-based page/paragraph number")

    @model_validator(mode="after")
    def end_not_before_start(self) -> "ConceptSource":
        """A reversed range means the page mapping went wrong upstream."""
        if self.block_end < self.block_start:
            raise ValueError(
                f"block_end ({self.block_end}) cannot be before block_start "
                f"({self.block_start}) in '{self.filename}'"
            )
        return self


class Concept(BaseModel):
    """A distinct technical educational concept extracted from the source text."""
    name: str = Field(..., min_length=1, description="Concept title/topic")
    explanation: str = Field(..., min_length=1, description="Substantive 1-2 sentence definition")
    sources: list[ConceptSource] = Field(..., min_length=1, description="At least one source reference")
    subtopics: list[str] = Field(default_factory=list, description="Names of narrower concepts merged into this one")


# ---------------------------------------------------------------------------
# Base model — fields that EVERY question type must have.
# Keeping shared fields here avoids repeating them in each subclass and
# gives Agent 3 a consistent surface to sort / filter questions.
# ---------------------------------------------------------------------------

class BaseQuestion(BaseModel):
    """Common metadata present on every question, regardless of type."""

    # Unique identifier used to reference questions across agents
    id: str

    # Name of the concept this question tests — must match a name from
    # concepts.json so we can trace which topic each question came from
    concept: str

    # Filename of the document where this concept originated (traceability)
    source_file: str

    # Which of the three question formats this is (used as the discriminator
    # so Pydantic knows which subclass to parse the dict into)
    type: QuestionType

    # Difficulty band — controls how questions are distributed in a quiz
    difficulty: Difficulty

    # Point value for this question; must be at least 1
    marks: int = Field(..., gt=0, description="Marks must be greater than 0")

    # The actual question text shown to the student; must not be blank
    question_text: str = Field(..., min_length=1)

    # Verification status from Agent 2 self-verification pass.
    # The schema validates structure but not correctness. This pass catches
    # wrong answer keys before professors review them.
    verification_status: Literal["verified", "needs_review", "not_applicable"] = Field(
        default="not_applicable",
        description="One of 'verified', 'needs_review', or 'not_applicable'",
    )

    # Recorded explanation, alternative answer, or reasoning on disagreement
    verification_note: Optional[str] = Field(
        default=None,
        description="Notes on verification result (e.g. what the verifier answered on disagreement)",
    )

    # Model that generated this question, for traceability and quality tracking
    generated_by: Optional[str] = Field(
        default=None,
        description="The LLM model name that generated this question",
    )

    @model_validator(mode="after")
    def question_text_not_whitespace(self) -> "BaseQuestion":
        """Reject question_text that is only spaces/newlines — min_length=1
        alone would accept a string like '   '."""
        if not self.question_text.strip():
            raise ValueError("question_text must not be blank or whitespace only")
        return self


# ---------------------------------------------------------------------------
# MCQQuestion — a multiple-choice question with exactly 4 options.
# The correct_option_index tells Agent 3 which option is the answer without
# having to compare strings, which avoids subtle whitespace mismatches.
# ---------------------------------------------------------------------------

class MCQQuestion(BaseQuestion):
    """A multiple-choice question. Must have exactly 4 options."""

    # Literal type tag — Pydantic uses this to route to MCQQuestion in the union
    type: Literal[QuestionType.MCQ] = QuestionType.MCQ

    # Exactly 4 answer choices — enforced by min_length and max_length
    options: list[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Must contain exactly 4 options",
    )

    # Index (0–3) of the correct option in the options list
    correct_option_index: int = Field(
        ...,
        ge=0,
        le=3,
        description="Must be 0, 1, 2, or 3",
    )

    # Explains why the correct answer is right — used in student feedback
    explanation: str

    @model_validator(mode="after")
    def index_within_range(self) -> "MCQQuestion":
        """Double-check the index is valid even if options length changes."""
        if self.correct_option_index >= len(self.options):
            raise ValueError(
                f"correct_option_index {self.correct_option_index} is out of "
                f"range for {len(self.options)} options"
            )
        return self


# ---------------------------------------------------------------------------
# CodeQuestion — asks the student to write, fix, or trace a code snippet.
# Having language as a separate field lets Agent 3 filter by language and
# lets the frontend apply syntax highlighting correctly.
# ---------------------------------------------------------------------------

class CodeQuestion(BaseQuestion):
    """A programming question. May include a starter or buggy code snippet."""

    type: Literal[QuestionType.CODE] = QuestionType.CODE

    # Programming language (e.g. "C", "Python", "Java")
    language: str

    # Optional: a snippet shown to the student (starter code, buggy code, etc.)
    # None means the student writes everything from scratch
    code_snippet: Optional[str] = None

    # The correct output, implementation, or fix that a model answer would show
    expected_answer: str

    # Explains the solution so students can learn from mistakes
    explanation: str


# ---------------------------------------------------------------------------
# TheoryQuestion — an open-ended written question.
# key_points gives Agent 3 (and eventually a human examiner) a checklist of
# what a full-mark answer must contain, making marking objective.
# ---------------------------------------------------------------------------

class TheoryQuestion(BaseQuestion):
    """An open-ended written question with a model answer and marking rubric."""

    type: Literal[QuestionType.THEORY] = QuestionType.THEORY

    # A complete example answer at the expected level of detail
    model_answer: str

    # At least 2 bullet points an examiner checks when awarding marks.
    # Minimum of 2 is enforced so every theory question has a real rubric.
    key_points: list[str] = Field(
        ...,
        min_length=2,
        description="At least 2 marking points are required",
    )


# ---------------------------------------------------------------------------
# Discriminated union — when Pydantic sees a dict with "type": "MCQ" it
# automatically parses it as MCQQuestion, "CODE" → CodeQuestion, etc.
# This is more precise than a plain Union because the type field drives
# which model is used, rather than Pydantic trying each model in order.
# ---------------------------------------------------------------------------

# Annotated with the discriminator so Quiz.questions gets the right subtype
AnyQuestion = Annotated[
    Union[MCQQuestion, CodeQuestion, TheoryQuestion],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Quiz — the top-level container passed between agents and written to disk.
# The total_marks validator acts as a sanity check: if Agent 2 sets the
# field wrong the error surfaces here rather than in the exam frontend.
# ---------------------------------------------------------------------------

class Quiz(BaseModel):
    """
    The complete assessment. Holds all questions produced by Agent 2 and
    refined by Agent 3. Written to outputs/quiz.json at the end of the
    pipeline.
    """

    # Filenames of source documents so any downstream tool knows the origin
    source_documents: list[str] = Field(..., min_length=1, description="List of source document names")

    # Claimed total marks — must equal the sum of individual question marks
    total_marks: int

    # The questions themselves, each parsed into the correct subtype
    questions: list[AnyQuestion]

    # Balance notes comparing actual difficulty distribution against target
    balance_note: Optional[str] = Field(
        default=None,
        description="Examiner notes on difficulty balance and quota shortfalls",
    )

    @model_validator(mode="after")
    def total_marks_must_match_sum(self) -> "Quiz":
        """Ensure total_marks equals the actual sum of question marks.
        Catches copy-paste mistakes where the header and questions disagree."""
        actual = sum(q.marks for q in self.questions)
        if self.total_marks != actual:
            raise ValueError(
                f"total_marks is {self.total_marks} but the questions sum to "
                f"{actual}. Update total_marks or fix individual question marks."
            )
        return self


# ---------------------------------------------------------------------------
# QuizSpec — configuration specification for generating an assessment.
# ---------------------------------------------------------------------------

class QuizSpec(BaseModel):
    """
    Specification for a quiz generation run.
    Defines total question volume and distribution across question formats.
    """
    total_questions: int = Field(..., gt=0, description="Total number of questions to generate (must be > 0)")
    mcq_ratio: float = Field(default=0.5, ge=0.0, le=1.0, description="Proportion of MCQ questions")
    code_ratio: float = Field(default=0.25, ge=0.0, le=1.0, description="Proportion of Code questions")
    theory_ratio: float = Field(default=0.25, ge=0.0, le=1.0, description="Proportion of Theory questions")

    @model_validator(mode="after")
    def ratios_must_sum_to_one(self) -> "QuizSpec":
        total_ratio = self.mcq_ratio + self.code_ratio + self.theory_ratio
        if abs(total_ratio - 1.0) > 1e-3:
            raise ValueError(
                f"Question ratios must sum to 1.0 (got {total_ratio:.4f}: "
                f"MCQ={self.mcq_ratio}, CODE={self.code_ratio}, THEORY={self.theory_ratio})"
            )
        return self


# ---------------------------------------------------------------------------
# DifficultyTarget — Target difficulty balance for assessment assembly.
# ---------------------------------------------------------------------------

class DifficultyTarget(BaseModel):
    """
    Target difficulty distribution across the assessment.
    Used by Agent 3 to compare actual vs planned cognitive load.
    """
    easy_ratio: float = Field(default=0.3, ge=0.0, le=1.0, description="Target proportion of EASY questions")
    medium_ratio: float = Field(default=0.5, ge=0.0, le=1.0, description="Target proportion of MEDIUM questions")
    hard_ratio: float = Field(default=0.2, ge=0.0, le=1.0, description="Target proportion of HARD questions")

    @model_validator(mode="after")
    def ratios_must_sum_to_one(self) -> "DifficultyTarget":
        total_ratio = self.easy_ratio + self.medium_ratio + self.hard_ratio
        if abs(total_ratio - 1.0) > 1e-3:
            raise ValueError(
                f"Difficulty ratios must sum to 1.0 (got {total_ratio:.4f}: "
                f"EASY={self.easy_ratio}, MEDIUM={self.medium_ratio}, HARD={self.hard_ratio})"
            )
        return self


