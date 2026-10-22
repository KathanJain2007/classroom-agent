"""
pipeline_state.py — LangGraph State Schema for the Multi-Agent Quiz Generator

Defines the shared state flowing across all 20 nodes of the LangGraph state machine.
Enforces strict typing and data contracts across Agent 1 (Extraction), Agent 2 (Question Writing
and Verification), and Agent 3 (Assessment Assembly and Calibration).
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from extractors import Chunk, DocumentContent
from schemas import (
    AnyQuestion,
    Concept,
    DifficultyTarget,
    Quiz,
    QuizSpec,
)


class PipelineState(TypedDict, total=False):
    """
    Complete state dictionary that flows between all nodes in the LangGraph pipeline.

    Attributes:
        input_paths: Paths to source PDF or DOCX documents to process.
        documents: Extracted and cleaned DocumentContent objects.
        concepts: Extracted, deduplicated, and semantically merged concepts.
        quiz_spec: Quiz configuration defining target question count and type ratios.
        generation_plan: Mapping of concepts to question types and pedagogical rationales.
        questions: Generated questions moving through the verification and calibration checks.
        failed_question_ids: Question IDs flagged by any safety check requiring regeneration.
        check_results: Detailed audit results from each of the 5 distinct safety checks.
        regeneration_attempts: Dictionary tracking retry count per question ID (bounded to < 2).
        difficulty_target: Target difficulty distribution (EASY / MEDIUM / HARD).
        final_quiz: The assembled, renumbered, and validated Quiz instance.
        errors: Non-fatal warnings and diagnostic messages accumulated across execution.
        llm_calls: Running counter of total LLM calls across all graph nodes.
        started_at: Unix timestamp recorded at the start of pipeline execution.

        chunks: Document chunks with block traceability for per-chunk extraction.
        chunk_results: Raw concept lists extracted from individual chunks before merging.
        coherence_feedback: Specific entity mismatch reasons keyed by question ID.
        breakdowns: Itemized mark breakdown criteria keyed by question ID.
        mismatch_reports: Diagnostics from mark breakdown criteria alignment checks.
        breakdown_mismatch_detected: Flag indicating whether criteria alignment failed on non-MCQs.
        breakdown_retried: Boolean guard bounding the breakdown regeneration conditional edge to 1 retry.
        balance_note: Honest summary comparing target vs actual cognitive difficulty.
        target_counts: Expected question count per difficulty band.
        actual_counts: Actual question count per difficulty band.
        selected_model: Active LLM model name determined during startup probe.
        model_selection_reason: Justification recorded during startup model probe.
    """

    # Core required fields specified in design contract
    input_paths: list[str]
    documents: list[DocumentContent]
    concepts: list[Concept]
    quiz_spec: QuizSpec
    generation_plan: list[Any]
    questions: list[AnyQuestion]
    failed_question_ids: list[str]
    check_results: dict[str, Any]
    regeneration_attempts: dict[str, int]
    difficulty_target: DifficultyTarget
    final_quiz: Optional[Quiz]
    errors: list[str]
    llm_calls: int
    started_at: float

    # Intermediate artifacts and edge routing guards
    chunks: list[tuple[DocumentContent, list[Chunk]]]
    chunk_results: list[list[Concept]]
    coherence_feedback: dict[str, str]
    breakdowns: dict[str, list[dict]]
    mismatch_reports: list[tuple[str, str]]
    breakdown_mismatch_detected: bool
    breakdown_retried: bool
    breakdown_run_count: int
    balance_note: str
    target_counts: dict[str, int]
    actual_counts: dict[str, int]
    selected_model: str
    model_selection_reason: str
