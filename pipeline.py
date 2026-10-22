"""
pipeline.py — LangGraph State Machine for Multi-Agent Classroom Quiz Generator

Wraps and orchestrates existing logic across:
- Agent 1 (Extraction & Deduplication: step1_test.py + extractors.py)
- Agent 2 (Batched Generation & 5 Safety Checks: agent2.py)
- Agent 3 (Assessment Assembly, Balancing & Calibration: agent3.py)

** ARCHITECTURAL CONSTRAINT: WRAP, DO NOT REWRITE **
Every existing function is called as-is. Thin adapters bridge graph state
to existing agent function signatures without altering any core logic.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

# Agent 1 imports
import extractors
import step1_test
from extractors import Chunk, DocumentContent, build_chunks, extract_document
from step1_test import (
    MODEL_NAME,
    collect_file_paths,
    extract_concepts_from_chunk,
    get_llm_client,
    merge_concepts_string_pass,
    print_chunking_summary,
    report_quality,
    save_and_display_final_concepts,
    select_active_model,
    semantic_merge_concepts,
)

# Agent 2 imports
import agent2
from agent2 import (
    INTER_CALL_DELAY_SECONDS,
    SAMPLE_QUIZ_PATH,
    apply_deterministic_repairs,
    batch_check_grounding,
    batch_check_premises,
    batch_verify_codes,
    batch_verify_mcqs,
    check_entity_coherence,
    generate_codes_batch,
    generate_mcqs_batch,
    generate_single_question,
    generate_theory_batch,
    load_sample_questions,
    plan_quiz_generation,
)

# Agent 3 imports
import agent3
from agent3 import (
    FINAL_QUIZ_PATH,
    MARKING_SCHEME_JSON_PATH,
    MARKING_SCHEME_TXT_PATH,
    MARKS_TABLE,
    QUESTION_PAPER_TXT_PATH,
    assign_difficulty,
    batch_assess_difficulties,
    batch_generate_mark_breakdowns,
    convert_option_indices_to_letters,
    evaluate_paper_balance,
    format_marking_scheme_text,
    format_question_paper_text,
    generate_single_mark_breakdown,
    order_questions_for_paper,
)

# Shared Schema & State
from pipeline_state import PipelineState
from schemas import (
    AnyQuestion,
    CodeQuestion,
    Concept,
    Difficulty,
    DifficultyTarget,
    MCQQuestion,
    QuestionType,
    Quiz,
    QuizSpec,
    TheoryQuestion,
)


# ---------------------------------------------------------------------------
# Centralized LLM Call Counter & Provider Cache
# ---------------------------------------------------------------------------
_GLOBAL_LLM_CALL_COUNT = 0
_CACHED_CLIENT: Optional[object] = None
_CACHED_PROVIDER: Optional[str] = None
_ACTIVE_MODEL: Optional[str] = None
_MODEL_SELECTION_REASON: Optional[str] = None


def get_llm_call_count() -> int:
    """Returns total LLM calls recorded across all graph nodes."""
    return _GLOBAL_LLM_CALL_COUNT


def increment_llm_call_count() -> int:
    """Increments the central LLM call counter by 1."""
    global _GLOBAL_LLM_CALL_COUNT
    _GLOBAL_LLM_CALL_COUNT += 1
    return _GLOBAL_LLM_CALL_COUNT


# Instrument step1_test.call_llm_with_retry to track all LLM calls uniformly
_original_call_llm_with_retry = step1_test.call_llm_with_retry


def _instrumented_call_llm(*args, **kwargs) -> Optional[str]:
    increment_llm_call_count()
    return _original_call_llm_with_retry(*args, **kwargs)


step1_test.call_llm_with_retry = _instrumented_call_llm
agent2.call_llm_with_retry = _instrumented_call_llm
agent3.call_llm_with_retry = _instrumented_call_llm


def get_shared_client() -> tuple[object, str, str, str]:
    """
    Initializes or returns cached LLM client, provider name, and active model.
    Ensures model selection probe runs exactly once per execution.
    """
    global _CACHED_CLIENT, _CACHED_PROVIDER, _ACTIVE_MODEL, _MODEL_SELECTION_REASON
    if _CACHED_CLIENT is None:
        _CACHED_CLIENT, _CACHED_PROVIDER = get_llm_client()
        _ACTIVE_MODEL, _MODEL_SELECTION_REASON = select_active_model(_CACHED_CLIENT)
    return _CACHED_CLIENT, _CACHED_PROVIDER, _ACTIVE_MODEL, _MODEL_SELECTION_REASON


# ---------------------------------------------------------------------------
# Human-Readable Graph Node Identifiers
# ---------------------------------------------------------------------------
NODE_EXTRACT_DOCS = "Agent 1 - Extract Documents"
NODE_CHUNK_DOCS = "Agent 1 - Chunk Documents"
NODE_EXTRACT_CONCEPTS = "Agent 1 - Extract Concepts"
NODE_MERGE_CONCEPTS = "Agent 1 - Merge Concepts"
NODE_SAVE_CONCEPTS = "Agent 1 - Save Concepts"

NODE_PLAN_GENERATION = "Agent 2 - Build Generation Plan"
NODE_GENERATE_QUESTIONS = "Agent 2 - Generate Questions"
NODE_CHECK_SCHEMA = "Check 1 - Schema Validation"
NODE_CHECK_COHERENCE = "Check 2 - Entity Coherence"
NODE_CHECK_PREMISE = "Check 3 - Premise Trace"
NODE_CHECK_VERIFICATION = "Check 4 - Self Verification"
NODE_CHECK_GROUNDING = "Check 5 - Notes Grounding"
NODE_REGENERATE_FAILED = "Agent 2 - Regenerate Failed"
NODE_SAVE_QUESTIONS = "Agent 2 - Save Questions"

NODE_ASSESS_DIFFICULTY = "Agent 3 - Assess Difficulty"
NODE_EVALUATE_BALANCE = "Agent 3 - Evaluate Balance"
NODE_ALLOCATE_MARKS = "Agent 3 - Allocate Marks"
NODE_GENERATE_BREAKDOWNS = "Agent 3 - Generate Breakdowns"
NODE_ASSEMBLE_PAPER = "Agent 3 - Assemble Paper"
NODE_SAVE_OUTPUTS = "Agent 3 - Save Outputs"


# ===========================================================================
# AGENT 1 NODES: Extraction & Concept Normalization
# ===========================================================================

def extract_documents_node(state: PipelineState) -> dict:
    """Agent 1 Node 1: Resolves file/folder paths and extracts raw DocumentContent."""
    print(f"\n>>> Running [{NODE_EXTRACT_DOCS}]...")
    input_paths = state.get("input_paths", [])
    resolved_paths = collect_file_paths(input_paths)

    successful_docs: list[DocumentContent] = []
    errors: list[str] = list(state.get("errors", []))

    os.makedirs("outputs", exist_ok=True)
    for path in resolved_paths:
        try:
            doc = extract_document(str(path))
            if not doc.blocks:
                msg = f"Document contains no extractable text: {path.name}"
                errors.append(msg)
                print(f"  [Warning] Skipping '{path.name}': {msg}")
                continue
            successful_docs.append(doc)
            report_quality(doc)

            cleaned_file_path = f"outputs/cleaned_{doc.filename}.txt"
            with open(cleaned_file_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(doc.blocks))
            print(f"  Saved cleaned text to: {cleaned_file_path}")
        except Exception as exc:
            err_msg = f"Failed to extract '{path.name}': {exc}"
            errors.append(err_msg)
            print(f"  [Warning] {err_msg}")

    return {
        "documents": successful_docs,
        "errors": errors,
        "llm_calls": get_llm_call_count(),
    }


def chunk_documents_node(state: PipelineState) -> dict:
    """Agent 1 Node 2: Partitions documents into isolated, overlapping chunks."""
    print(f"\n>>> Running [{NODE_CHUNK_DOCS}]...")
    docs = state.get("documents", [])
    doc_chunk_plan: list[tuple[DocumentContent, list[Chunk]]] = []

    for doc in docs:
        # page_map translates chunk ranges into real source page numbers, so
        # concept citations point at the page a professor would actually open.
        chunks = build_chunks(doc.blocks, page_map=doc.block_page_map)
        doc_chunk_plan.append((doc, chunks))
        print_chunking_summary(chunks, doc.filename)

    return {
        "chunks": doc_chunk_plan,
        "llm_calls": get_llm_call_count(),
    }


def extract_concepts_node(state: PipelineState) -> dict:
    """Agent 1 Node 3: Executes per-chunk LLM concept extraction."""
    print(f"\n>>> Running [{NODE_EXTRACT_CONCEPTS}]...")
    client, provider, model, _ = get_shared_client()
    doc_chunk_plan = state.get("chunks", [])

    total_chunks = sum(len(chunks) for _, chunks in doc_chunk_plan)
    chunk_results: list[list[Concept]] = []
    global_chunk_idx = 0

    for doc, chunks in doc_chunk_plan:
        for chunk in chunks:
            global_chunk_idx += 1
            extracted = extract_concepts_from_chunk(
                chunk=chunk,
                doc=doc,
                client=client,
                provider=provider,
                global_chunk_num=global_chunk_idx,
                total_global_chunks=total_chunks,
            )
            if extracted:
                chunk_results.append(extracted)

    return {
        "chunk_results": chunk_results,
        "llm_calls": get_llm_call_count(),
    }


def merge_concepts_node(state: PipelineState) -> dict:
    """Agent 1 Node 4: Performs Pass 1 (string normalization) and Pass 2 (semantic merge)."""
    print(f"\n>>> Running [{NODE_MERGE_CONCEPTS}]...")
    client, provider, model, _ = get_shared_client()
    chunk_results = state.get("chunk_results", [])

    # Pass 1: Deduplication via string normalization
    pass1_concepts = merge_concepts_string_pass(chunk_results)
    print(f"  Pass 1 (String Normalization): Reduced to {len(pass1_concepts)} unique concepts.")

    # Pass 2: Semantic Deduplication with structured output
    final_concepts, group_merges = semantic_merge_concepts(
        concepts=pass1_concepts,
        client=client,
        provider=provider,
    )
    print(f"  Pass 2 (Semantic Merge): Finalized {len(final_concepts)} educational concepts.")

    # Record a concept shortfall here rather than in the routing function:
    # LangGraph discards state written by a router, so a warning raised there
    # would print once and then vanish from the final summary.
    errors = list(state.get("errors", []))
    spec = state.get("quiz_spec", QuizSpec(total_questions=12))
    needed = spec.total_questions
    actual = len(final_concepts)
    if actual < needed:
        warning_msg = (
            f"Concept Shortfall: Extracted {actual} concepts, but QuizSpec requested {needed}. "
            f"Some concepts will be reused across questions."
        )
        print(f"  [Warning] {warning_msg}")
        errors.append(warning_msg)

    return {
        "concepts": final_concepts,
        "errors": errors,
        "llm_calls": get_llm_call_count(),
    }


def save_concepts_node(state: PipelineState) -> dict:
    """Agent 1 Node 5: Persists concepts to outputs/concepts.json and prints summary."""
    print(f"\n>>> Running [{NODE_SAVE_CONCEPTS}]...")
    concepts = state.get("concepts", [])
    save_and_display_final_concepts(concepts, output_path="outputs/concepts.json")
    return {"llm_calls": get_llm_call_count()}


# ===========================================================================
# AGENT 2 NODES: Batched Question Generation & 5 Safety Checks
# ===========================================================================

def build_generation_plan_node(state: PipelineState) -> dict:
    """Agent 2 Node 6: Selects concepts and assigns question formats per QuizSpec."""
    print(f"\n>>> Running [{NODE_PLAN_GENERATION}]...")
    concepts = state.get("concepts", [])
    quiz_spec = state.get("quiz_spec", QuizSpec(total_questions=12))

    plan = plan_quiz_generation(concepts, quiz_spec)

    print("\n" + "=" * 60)
    print(f"GENERATION PLAN (Total Target: {quiz_spec.total_questions} Questions)")
    print("=" * 60)
    for idx, (concept, q_type, rationale) in enumerate(plan, start=1):
        print(f"  [q{idx:02d}] [{q_type.value:6s}] '{concept.name}' — {rationale}")
    print("=" * 60)

    return {
        "generation_plan": plan,
        "llm_calls": get_llm_call_count(),
    }


def generate_questions_node(state: PipelineState) -> dict:
    """Agent 2 Node 7: Batched question generation by type (MCQ, CODE, THEORY)."""
    print(f"\n>>> Running [{NODE_GENERATE_QUESTIONS}]...")
    client, provider, model, _ = get_shared_client()
    plan = state.get("generation_plan", [])
    sample_questions = load_sample_questions(SAMPLE_QUIZ_PATH)

    needed_plan_items: list[tuple[Concept, QuestionType, str, str]] = []
    for idx, (concept, q_type, rationale) in enumerate(plan, start=1):
        needed_plan_items.append((concept, q_type, rationale, f"q{idx}"))

    mcq_items = [
        (c, q_id, c.sources[0].filename if c.sources else "unknown")
        for c, qt, _, q_id in needed_plan_items
        if qt == QuestionType.MCQ
    ]
    code_items = [
        (c, q_id, c.sources[0].filename if c.sources else "unknown")
        for c, qt, _, q_id in needed_plan_items
        if qt == QuestionType.CODE
    ]
    theory_items = [
        (c, q_id, c.sources[0].filename if c.sources else "unknown")
        for c, qt, _, q_id in needed_plan_items
        if qt == QuestionType.THEORY
    ]

    generated_questions: list[AnyQuestion] = []

    if mcq_items:
        print(f"  Generating batch of {len(mcq_items)} MCQ question(s)...")
        generated_mcqs = generate_mcqs_batch(mcq_items, sample_questions.get("MCQ", {}), client, model)
        generated_questions.extend(generated_mcqs)

    if code_items:
        time.sleep(INTER_CALL_DELAY_SECONDS)
        print(f"  Generating batch of {len(code_items)} CODE question(s)...")
        generated_codes = generate_codes_batch(code_items, sample_questions.get("CODE", {}), client, model)
        generated_questions.extend(generated_codes)

    if theory_items:
        time.sleep(INTER_CALL_DELAY_SECONDS)
        print(f"  Generating batch of {len(theory_items)} THEORY question(s)...")
        generated_theories = generate_theory_batch(theory_items, sample_questions.get("THEORY", {}), client, model)
        generated_questions.extend(generated_theories)

    # Order questions according to original plan IDs
    q_by_id = {q.id: q for q in generated_questions}
    working_questions: list[AnyQuestion] = []
    for idx in range(1, len(plan) + 1):
        qid = f"q{idx}"
        if qid in q_by_id:
            working_questions.append(q_by_id[qid])

    return {
        "questions": working_questions,
        "failed_question_ids": [],
        "check_results": {},
        "regeneration_attempts": {},
        "coherence_feedback": {},
        "llm_calls": get_llm_call_count(),
    }


def check_schema_node(state: PipelineState) -> dict:
    """Agent 2 Node 8: Check 1 [Schema Validation] using Pydantic models."""
    print(f"\n>>> Running [{NODE_CHECK_SCHEMA}]...")
    questions = state.get("questions", [])
    attempts = state.get("regeneration_attempts", {})
    check_results = dict(state.get("check_results", {}))

    failed_ids: list[str] = []
    passed_count = 0

    for q in questions:
        try:
            apply_deterministic_repairs(q)
            # Validate through model validator
            if isinstance(q, MCQQuestion):
                MCQQuestion.model_validate(q.model_dump())
            elif isinstance(q, CodeQuestion):
                CodeQuestion.model_validate(q.model_dump())
            elif isinstance(q, TheoryQuestion):
                TheoryQuestion.model_validate(q.model_dump())
            passed_count += 1
        except Exception as exc:
            print(f"  [Schema Failure] {q.id} '{q.concept}': {exc}")
            if attempts.get(q.id, 0) >= 2:
                q.verification_status = "needs_review"
                q.verification_note = f"Schema validation failed after 2 attempts: {exc}"
            else:
                failed_ids.append(q.id)

    check_results["schema"] = {
        "total": len(questions),
        "passed": passed_count,
        "failed": len(failed_ids),
    }

    return {
        "failed_question_ids": failed_ids,
        "check_results": check_results,
        "llm_calls": get_llm_call_count(),
    }


def check_coherence_node(state: PipelineState) -> dict:
    """Agent 2 Node 9: Check 2 [Deterministic Entity Coherence Check]."""
    print(f"\n>>> Running [{NODE_CHECK_COHERENCE}]...")
    questions = state.get("questions", [])
    attempts = state.get("regeneration_attempts", {})
    feedback_map = dict(state.get("coherence_feedback", {}))
    check_results = dict(state.get("check_results", {}))

    failed_ids: list[str] = []
    passed_count = 0
    flagged_review_count = 0

    for q in questions:
        is_coherent, reason = check_entity_coherence(q)
        if is_coherent:
            passed_count += 1
        else:
            print(f"  [Coherence Mismatch] {q.id} '{q.concept}': {reason}")
            feedback_map[q.id] = reason
            if attempts.get(q.id, 0) >= 2:
                q.verification_status = "needs_review"
                q.verification_note = f"Coherence check failed after max attempts: {reason}"
                flagged_review_count += 1
            else:
                failed_ids.append(q.id)

    check_results["coherence"] = {
        "passed": passed_count,
        "flagged_for_regen": len(failed_ids),
        "needs_review": flagged_review_count,
    }

    return {
        "failed_question_ids": failed_ids,
        "coherence_feedback": feedback_map,
        "check_results": check_results,
        "llm_calls": get_llm_call_count(),
    }


def check_premise_node(state: PipelineState) -> dict:
    """Agent 2 Node 10: Check 3 [Code Premise & Trace Check] on CODE questions."""
    print(f"\n>>> Running [{NODE_CHECK_PREMISE}]...")
    client, provider, model, _ = get_shared_client()
    questions = state.get("questions", [])
    attempts = state.get("regeneration_attempts", {})
    check_results = dict(state.get("check_results", {}))

    code_qs = [q for q in questions if isinstance(q, CodeQuestion)]
    premise_results = batch_check_premises(code_qs, client, model) if code_qs else {}

    failed_ids: list[str] = []
    passed_count = 0
    latent_count = 0
    review_count = 0

    for q in code_qs:
        p_res = premise_results.get(q.id)
        if not p_res:
            continue

        if p_res["passed"]:
            passed_count += 1
            if not p_res.get("manifests_in_main", True):
                latent_count += 1
                q.verification_note = p_res["note"]
        else:
            print(f"  [False Premise] {q.id} '{q.concept}': {p_res['note']}")
            if attempts.get(q.id, 0) >= 2:
                q.verification_status = "needs_review"
                q.verification_note = p_res["note"]
                review_count += 1
            else:
                failed_ids.append(q.id)

    check_results["premise"] = {
        "passed": passed_count,
        "latent_bugs": latent_count,
        "flagged_for_regen": len(failed_ids),
        "needs_review": review_count,
    }

    return {
        "failed_question_ids": failed_ids,
        "check_results": check_results,
        "llm_calls": get_llm_call_count(),
    }


def check_verification_node(state: PipelineState) -> dict:
    """Agent 2 Node 11: Check 4 [Batched Independent Self-Verification Pass]."""
    print(f"\n>>> Running [{NODE_CHECK_VERIFICATION}]...")
    client, provider, model, _ = get_shared_client()
    questions = state.get("questions", [])
    check_results = dict(state.get("check_results", {}))

    mcqs_to_verify = [q for q in questions if isinstance(q, MCQQuestion)]
    codes_to_verify = [
        q for q in questions
        if isinstance(q, CodeQuestion) and q.verification_status != "needs_review"
    ]

    mcq_ver_results = batch_verify_mcqs(mcqs_to_verify, client, model) if mcqs_to_verify else {}
    code_ver_results = batch_verify_codes(codes_to_verify, client, model) if codes_to_verify else {}

    verified_count = 0
    review_count = 0
    na_count = 0

    for q in questions:
        if isinstance(q, MCQQuestion) and q.id in mcq_ver_results:
            st, nt = mcq_ver_results[q.id]
            q.verification_status = st
            q.verification_note = nt
            if st == "verified":
                verified_count += 1
            else:
                review_count += 1
        elif isinstance(q, CodeQuestion) and q.id in code_ver_results:
            st, nt = code_ver_results[q.id]
            q.verification_status = st
            q.verification_note = nt
            if st == "verified":
                verified_count += 1
            else:
                review_count += 1
        elif isinstance(q, TheoryQuestion):
            q.verification_status = "not_applicable"
            q.verification_note = "Open-ended theory question; unverified by design."
            na_count += 1

    check_results["verification"] = {
        "verified": verified_count,
        "needs_review": review_count,
        "unverified_theory": na_count,
    }

    return {
        "failed_question_ids": [],
        "check_results": check_results,
        "llm_calls": get_llm_call_count(),
    }


def check_grounding_node(state: PipelineState) -> dict:
    """Agent 2 Node 12: Check 5 [Batched Grounding Check against Course Material]."""
    print(f"\n>>> Running [{NODE_CHECK_GROUNDING}]...")
    client, provider, model, _ = get_shared_client()
    questions = state.get("questions", [])
    concepts = state.get("concepts", [])
    concepts_by_name = {c.name: c for c in concepts}
    check_results = dict(state.get("check_results", {}))

    qs_to_ground = [q for q in questions if q.verification_status == "verified"]
    grounding_results = batch_check_grounding(qs_to_ground, concepts_by_name, client, model) if qs_to_ground else {}

    grounding_passed = 0
    grounding_conflicts = 0

    for q in questions:
        if q.id in grounding_results:
            g_res = grounding_results[q.id]
            if g_res["consistent"]:
                grounding_passed += 1
            else:
                grounding_conflicts += 1
                q.verification_status = "needs_review"
                q.verification_note = f"Grounding conflict with source notes: {g_res['conflict']}"

    check_results["grounding"] = {
        "passed": grounding_passed,
        "conflicts_flagged": grounding_conflicts,
    }

    return {
        "failed_question_ids": [],
        "check_results": check_results,
        "llm_calls": get_llm_call_count(),
    }


def regenerate_failed_node(state: PipelineState) -> dict:
    """
    Agent 2 Node 13: Targeted regenerator for failed question IDs.
    Increments attempt counter per ID and replaces only failed questions.
    """
    print(f"\n>>> Running [{NODE_REGENERATE_FAILED}]...")
    client, provider, model, _ = get_shared_client()
    failed_ids = state.get("failed_question_ids", [])
    attempts = dict(state.get("regeneration_attempts", {}))
    questions = list(state.get("questions", []))
    concepts = state.get("concepts", [])
    concepts_by_name = {c.name: c for c in concepts}
    coherence_feedback = state.get("coherence_feedback", {})
    sample_questions = load_sample_questions(SAMPLE_QUIZ_PATH)

    questions_by_id = {q.id: q for q in questions}

    for qid in failed_ids:
        attempts[qid] = attempts.get(qid, 0) + 1
        orig_q = questions_by_id.get(qid)
        if not orig_q:
            continue

        concept = concepts_by_name.get(orig_q.concept)
        if not concept:
            continue

        print(f"  Targeted regeneration for {qid} [{orig_q.type.value}] '{orig_q.concept}' (attempt {attempts[qid]}/2)...")
        feedback = coherence_feedback.get(qid)
        sample_q = sample_questions.get(orig_q.type.value, {})

        time.sleep(INTER_CALL_DELAY_SECONDS)
        regen_q = generate_single_question(
            concept=concept,
            q_type=orig_q.type,
            question_id=qid,
            sample_question=sample_q,
            client=client,
            model=model,
            coherence_feedback=feedback,
        )

        if regen_q:
            questions_by_id[qid] = regen_q
            print(f"  -> Successfully regenerated {qid} for '{concept.name}'.")
        else:
            print(f"  [Warning] Single regeneration returned None for {qid}. Retaining previous question object.")

    # Reassemble the list by each question's OWN id, preserving original order.
    #
    # The previous version looked up f"q{position}", which silently assumed every
    # question's id matched its position in the list. Any deviation — a gap from a
    # failed generation, or zero-padded ids like "q01" — made every lookup miss and
    # discarded all regenerated questions without warning.
    updated_questions = [questions_by_id.get(q.id, q) for q in questions]

    return {
        "questions": updated_questions,
        "failed_question_ids": [],
        "regeneration_attempts": attempts,
        "llm_calls": get_llm_call_count(),
    }


def save_questions_node(state: PipelineState) -> dict:
    """Agent 2 Node 14: Validates Quiz schema and writes outputs/questions.json."""
    print(f"\n>>> Running [{NODE_SAVE_QUESTIONS}]...")
    questions = state.get("questions", [])
    concepts = state.get("concepts", [])

    all_sources = sorted(list({s.filename for c in concepts for s in c.sources}))
    total_marks = sum(q.marks for q in questions)
    quiz_data = {
        "source_documents": all_sources,
        "total_marks": total_marks,
        "questions": [q.model_dump() for q in questions],
    }

    Quiz.model_validate(quiz_data)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/questions.json", "w", encoding="utf-8") as f:
        json.dump(quiz_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(questions)} validated questions to outputs/questions.json")

    return {"llm_calls": get_llm_call_count()}


# ===========================================================================
# AGENT 3 NODES: Assessment Assembly, Balancing & Calibration
# ===========================================================================

def assess_difficulty_node(state: PipelineState) -> dict:
    """Agent 3 Node 15: Batched cognitive difficulty assessment across all questions."""
    print(f"\n>>> Running [{NODE_ASSESS_DIFFICULTY}]...")
    client, provider, model, _ = get_shared_client()
    questions = list(state.get("questions", []))

    # Defensive ID and MCQ letter normalization
    seen_ids: set[str] = set()
    for idx, q in enumerate(questions, start=1):
        if not q.id or q.id in seen_ids:
            q.id = f"q{idx}"
        seen_ids.add(q.id)
        if isinstance(q, MCQQuestion):
            q.explanation = convert_option_indices_to_letters(q.explanation)

    batch_assess_difficulties(questions, client, model)

    print("\n" + "=" * 60)
    print("DIFFICULTY ASSIGNMENTS WITH REASONING")
    print("=" * 60)
    for q in questions:
        diff_enum, reason = assign_difficulty(q)
        q.difficulty = diff_enum
        print(f"  [{q.id}] [{q.type.value:6s}] Assessed: {diff_enum.value:<6s} — {q.concept}")
        print(f"        Reasoning: {reason}")
    print("=" * 60)

    return {
        "questions": questions,
        "llm_calls": get_llm_call_count(),
    }


def evaluate_balance_node(state: PipelineState) -> dict:
    """Agent 3 Node 16: Evaluates target vs actual balance with honest shortfall reporting."""
    print(f"\n>>> Running [{NODE_EVALUATE_BALANCE}]...")
    questions = state.get("questions", [])
    target = state.get("difficulty_target", DifficultyTarget(easy_ratio=0.3, medium_ratio=0.5, hard_ratio=0.2))

    balance_note, target_counts, actual_counts = evaluate_paper_balance(questions, target)

    print("\nPAPER COGNITIVE BALANCE EVALUATION:")
    print(f"  - Target Distribution : {target_counts['EASY']} Easy / {target_counts['MEDIUM']} Medium / {target_counts['HARD']} Hard")
    print(f"  - Actual Distribution : {actual_counts['EASY']} Easy / {actual_counts['MEDIUM']} Medium / {actual_counts['HARD']} Hard")
    print(f"  - Status Report       : {balance_note}")

    return {
        "balance_note": balance_note,
        "target_counts": target_counts,
        "actual_counts": actual_counts,
        "llm_calls": get_llm_call_count(),
    }


def allocate_marks_node(state: PipelineState) -> dict:
    """Agent 3 Node 17: Deterministic mark allocation via MARKS_TABLE lookup."""
    print(f"\n>>> Running [{NODE_ALLOCATE_MARKS}]...")
    questions = list(state.get("questions", []))

    for q in questions:
        key = (q.type, q.difficulty)
        q.marks = MARKS_TABLE.get(key, 2)

    total_marks = sum(q.marks for q in questions)
    print(f"  Computed total assessment marks: {total_marks} Marks.")

    return {
        "questions": questions,
        "llm_calls": get_llm_call_count(),
    }


def generate_breakdowns_node(state: PipelineState) -> dict:
    """Agent 3 Node 18: Generates criteria-level mark breakdowns for CODE and THEORY questions."""
    print(f"\n>>> Running [{NODE_GENERATE_BREAKDOWNS}]...")
    client, provider, model, _ = get_shared_client()
    questions = state.get("questions", [])
    code_and_theory_questions = [q for q in questions if isinstance(q, (CodeQuestion, TheoryQuestion))]

    # How many times this node has executed in the current run (1 on first pass).
    run_count = int(state.get("breakdown_run_count", 0)) + 1

    time.sleep(INTER_CALL_DELAY_SECONDS)
    print(f"  Generating criteria mark breakdowns for {len(code_and_theory_questions)} CODE & THEORY questions...")
    breakdowns, mismatch_reports = batch_generate_mark_breakdowns(code_and_theory_questions, client, model)

    has_mismatch = bool(mismatch_reports)

    # Record here — in the NODE — that the single permitted retry is now spent.
    #
    # WHY THIS LIVES IN THE NODE, NOT THE ROUTER: a conditional-edge router is a
    # pure routing function. Anything it writes into `state` is discarded, so a
    # router cannot mark its own retry as consumed. With the guard in the router,
    # a persistent mismatch routed back here on every pass — burning one LLM call
    # each time — until LangGraph's recursion limit aborted the entire run.
    # Returning the flag from the node is what actually bounds the loop.
    return {
        "breakdowns": breakdowns,
        "mismatch_reports": mismatch_reports,
        "breakdown_mismatch_detected": has_mismatch,
        "breakdown_run_count": run_count,
        "breakdown_retried": run_count >= 2,
        "llm_calls": get_llm_call_count(),
    }


def assemble_paper_node(state: PipelineState) -> dict:
    """Agent 3 Node 19: Orders questions pedagogically and constructs final Quiz object."""
    print(f"\n>>> Running [{NODE_ASSEMBLE_PAPER}]...")
    questions = state.get("questions", [])
    concepts = state.get("concepts", [])
    balance_note = state.get("balance_note", "")

    all_sources = sorted(list({s.filename for c in concepts for s in c.sources}))
    ordered_questions = order_questions_for_paper(questions)
    total_marks = sum(q.marks for q in ordered_questions)

    final_quiz = Quiz(
        source_documents=all_sources if all_sources else ["unknown"],
        total_marks=total_marks,
        questions=ordered_questions,
        balance_note=balance_note,
    )

    return {
        "final_quiz": final_quiz,
        "questions": ordered_questions,
        "llm_calls": get_llm_call_count(),
    }


def save_outputs_node(state: PipelineState) -> dict:
    """Agent 3 Node 20: Writes all four required output files."""
    print(f"\n>>> Running [{NODE_SAVE_OUTPUTS}]...")
    final_quiz = state.get("final_quiz")
    ordered_questions = state.get("questions", [])
    breakdowns = state.get("breakdowns", {})
    balance_note = state.get("balance_note", "")

    if not final_quiz:
        raise ValueError("Cannot save outputs: final_quiz has not been constructed.")

    os.makedirs("outputs", exist_ok=True)

    # 1. outputs/final_quiz.json
    with open(FINAL_QUIZ_PATH, "w", encoding="utf-8") as f:
        json.dump(final_quiz.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"  [Saved] Final Quiz object written to: {FINAL_QUIZ_PATH}")

    # 2. outputs/marking_scheme.json
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
    print(f"  [Saved] Structured Marking Scheme written to: {MARKING_SCHEME_JSON_PATH}")

    # 3. outputs/question_paper.txt
    question_paper_text = format_question_paper_text(final_quiz, ordered_questions)
    with open(QUESTION_PAPER_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(question_paper_text)
    print(f"  [Saved] Student-facing Question Paper written to: {QUESTION_PAPER_TXT_PATH}")

    # 4. outputs/marking_scheme.txt
    marking_scheme_text = format_marking_scheme_text(final_quiz, ordered_questions, breakdowns)
    with open(MARKING_SCHEME_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(marking_scheme_text)
    print(f"  [Saved] Examiner-facing Marking Scheme written to: {MARKING_SCHEME_TXT_PATH}")

    return {"llm_calls": get_llm_call_count()}


# ===========================================================================
# PART C: CONDITIONAL EDGES — RETRY TOPOLOGY & ROUTING LOGIC
# ===========================================================================
#
# WHY CONDITIONAL EDGES MATTER:
# A linear sequential script cannot express "re-run the check chain only on
# the questions that failed, up to a bounded number of attempts." In a linear
# script, retry loops are either nested imperatively inside node logic or all
# questions are redundantly regenerated.
#
# The LangGraph state machine makes the retry topology explicit, inspectable,
# and deterministic:
# 1. Any safety check (Schema, Coherence, Premise, Verification, Grounding) can
#    divert questions that fail into 'regenerate_failed', which loops back to
#    'check_schema' to re-validate the regenerated item through all 5 checks.
# 2. Each question ID is bounded to at most 2 regeneration attempts. When an ID
#    hits attempt 2, it is marked verification_status="needs_review" and allowed
#    to proceed, ensuring the graph cannot enter an infinite loop.
# 3. Mark breakdown alignment mismatches trigger at most 1 targeted retry loop.
# 4. Concept shortfalls are flagged into errors without halting pipeline flow.
# ===========================================================================

def make_safety_check_router(next_node: str):
    """
    Factory creating a conditional edge router for Checks 8-12.
    Routes to 'regenerate_failed' if any question ID failed and has attempts < 2.
    Otherwise routes forward to next_node.
    """
    def router(state: PipelineState) -> str:
        failed_ids = state.get("failed_question_ids", [])
        attempts = state.get("regeneration_attempts", {})

        to_regen = [qid for qid in failed_ids if attempts.get(qid, 0) < 2]
        if to_regen:
            print(
                f"\n[Conditional Edge] {len(to_regen)} question(s) ({to_regen}) failed check. "
                f"Routing to '{NODE_REGENERATE_FAILED}' for targeted regeneration."
            )
            return NODE_REGENERATE_FAILED

        return next_node

    return router


def concept_shortfall_router(state: PipelineState) -> str:
    """
    Conditional edge after merge_concepts.

    Routing decision ONLY — the shortfall warning itself is recorded by
    merge_concepts_node, because a router's writes to `state` are discarded by
    LangGraph and would never reach the final summary.
    """
    return NODE_SAVE_CONCEPTS


def breakdown_retry_router(state: PipelineState) -> str:
    """
    Conditional edge after generate_breakdowns:
    If a mark-breakdown mismatch was detected and not yet retried,
    routes back to generate_breakdowns once. Otherwise proceeds to assemble_paper.
    """
    mismatch_detected = state.get("breakdown_mismatch_detected", False)
    # Set by generate_breakdowns_node, not here — see the comment in that node.
    already_retried = state.get("breakdown_retried", False)

    if mismatch_detected and not already_retried:
        print("\n[Conditional Edge] Mark breakdown criteria mismatch detected. Routing back for 1 retry...")
        return NODE_GENERATE_BREAKDOWNS

    if mismatch_detected and already_retried:
        print(
            "\n[Conditional Edge] Mark breakdown mismatch persists after retry. "
            "Proceeding; affected questions are flagged for manual review."
        )

    return NODE_ASSEMBLE_PAPER


# ===========================================================================
# Graph Construction & Compilation
# ===========================================================================

def build_pipeline_graph(checkpointer: Optional[Any] = None):
    """
    Constructs the complete 20-node LangGraph state machine with conditional
    edges for question regeneration, criteria alignment, and concept shortfall.
    """
    builder = StateGraph(PipelineState)

    # 1. Register all 20 nodes with human-readable titles
    builder.add_node(NODE_EXTRACT_DOCS, extract_documents_node)
    builder.add_node(NODE_CHUNK_DOCS, chunk_documents_node)
    builder.add_node(NODE_EXTRACT_CONCEPTS, extract_concepts_node)
    builder.add_node(NODE_MERGE_CONCEPTS, merge_concepts_node)
    builder.add_node(NODE_SAVE_CONCEPTS, save_concepts_node)

    builder.add_node(NODE_PLAN_GENERATION, build_generation_plan_node)
    builder.add_node(NODE_GENERATE_QUESTIONS, generate_questions_node)
    builder.add_node(NODE_CHECK_SCHEMA, check_schema_node)
    builder.add_node(NODE_CHECK_COHERENCE, check_coherence_node)
    builder.add_node(NODE_CHECK_PREMISE, check_premise_node)
    builder.add_node(NODE_CHECK_VERIFICATION, check_verification_node)
    builder.add_node(NODE_CHECK_GROUNDING, check_grounding_node)
    builder.add_node(NODE_REGENERATE_FAILED, regenerate_failed_node)
    builder.add_node(NODE_SAVE_QUESTIONS, save_questions_node)

    builder.add_node(NODE_ASSESS_DIFFICULTY, assess_difficulty_node)
    builder.add_node(NODE_EVALUATE_BALANCE, evaluate_balance_node)
    builder.add_node(NODE_ALLOCATE_MARKS, allocate_marks_node)
    builder.add_node(NODE_GENERATE_BREAKDOWNS, generate_breakdowns_node)
    builder.add_node(NODE_ASSEMBLE_PAPER, assemble_paper_node)
    builder.add_node(NODE_SAVE_OUTPUTS, save_outputs_node)

    # 2. Agent 1 Linear Edges & Concept Shortfall Conditional Edge
    builder.add_edge(START, NODE_EXTRACT_DOCS)
    builder.add_edge(NODE_EXTRACT_DOCS, NODE_CHUNK_DOCS)
    builder.add_edge(NODE_CHUNK_DOCS, NODE_EXTRACT_CONCEPTS)
    builder.add_edge(NODE_EXTRACT_CONCEPTS, NODE_MERGE_CONCEPTS)

    builder.add_conditional_edges(
        NODE_MERGE_CONCEPTS,
        concept_shortfall_router,
        [NODE_SAVE_CONCEPTS],
    )
    builder.add_edge(NODE_SAVE_CONCEPTS, NODE_PLAN_GENERATION)

    # 3. Agent 2 Generation & 5 Safety Checks with Regeneration Loops
    builder.add_edge(NODE_PLAN_GENERATION, NODE_GENERATE_QUESTIONS)
    builder.add_edge(NODE_GENERATE_QUESTIONS, NODE_CHECK_SCHEMA)

    # Conditional Edges for Checks 8-12
    builder.add_conditional_edges(
        NODE_CHECK_SCHEMA,
        make_safety_check_router(NODE_CHECK_COHERENCE),
        [NODE_REGENERATE_FAILED, NODE_CHECK_COHERENCE],
    )
    builder.add_conditional_edges(
        NODE_CHECK_COHERENCE,
        make_safety_check_router(NODE_CHECK_PREMISE),
        [NODE_REGENERATE_FAILED, NODE_CHECK_PREMISE],
    )
    builder.add_conditional_edges(
        NODE_CHECK_PREMISE,
        make_safety_check_router(NODE_CHECK_VERIFICATION),
        [NODE_REGENERATE_FAILED, NODE_CHECK_VERIFICATION],
    )
    builder.add_conditional_edges(
        NODE_CHECK_VERIFICATION,
        make_safety_check_router(NODE_CHECK_GROUNDING),
        [NODE_REGENERATE_FAILED, NODE_CHECK_GROUNDING],
    )
    builder.add_conditional_edges(
        NODE_CHECK_GROUNDING,
        make_safety_check_router(NODE_SAVE_QUESTIONS),
        [NODE_REGENERATE_FAILED, NODE_SAVE_QUESTIONS],
    )

    # Regeneration routes back to Check 1 to re-run full check chain on regenerated items
    builder.add_edge(NODE_REGENERATE_FAILED, NODE_CHECK_SCHEMA)

    # 4. Agent 2 -> Agent 3 Handoff
    builder.add_edge(NODE_SAVE_QUESTIONS, NODE_ASSESS_DIFFICULTY)

    # 5. Agent 3 Linear Flow, Breakdown Retry Edge & Output
    builder.add_edge(NODE_ASSESS_DIFFICULTY, NODE_EVALUATE_BALANCE)
    builder.add_edge(NODE_EVALUATE_BALANCE, NODE_ALLOCATE_MARKS)
    builder.add_edge(NODE_ALLOCATE_MARKS, NODE_GENERATE_BREAKDOWNS)

    builder.add_conditional_edges(
        NODE_GENERATE_BREAKDOWNS,
        breakdown_retry_router,
        [NODE_GENERATE_BREAKDOWNS, NODE_ASSEMBLE_PAPER],
    )
    builder.add_edge(NODE_ASSEMBLE_PAPER, NODE_SAVE_OUTPUTS)
    builder.add_edge(NODE_SAVE_OUTPUTS, END)

    return builder.compile(checkpointer=checkpointer)
