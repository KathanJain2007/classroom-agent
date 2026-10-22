"""
main.py — Single Entry Point for the Multi-Agent Automated Classroom Quiz Generator.

Executes the LangGraph state machine orchestrating:
- Agent 1: Document extraction, cleaning, chunking, and two-pass concept deduplication.
- Agent 2: Batched question generation with 5 distinct quality checks and bounded retry loops.
- Agent 3: Cognitive difficulty assessment, balance reporting, deterministic marks allocation,
           and question paper/marking scheme formatting.

Usage examples:
    python main.py sample_pdfs/DS-UNIT-1.pdf sample_pdfs/DS-UNIT-2.pdf "sample_pdfs/Double Ended Queue.docx"
    python main.py sample_pdfs/
    python main.py --resume
    python main.py --questions 6
    python main.py --visualize
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver

import pipeline
from pipeline import build_pipeline_graph, get_llm_call_count, get_shared_client
from pipeline_state import PipelineState
from schemas import Difficulty, DifficultyTarget, QuizSpec

CHECKPOINT_DB_PATH = "outputs/checkpoints.db"

# File recording the thread id of the most recent run, so --resume knows which
# checkpoint to continue.
LAST_THREAD_FILE = "outputs/.last_thread_id"


def new_thread_id() -> str:
    """Creates a unique thread id for a fresh run.

    A FIXED thread id is wrong here. LangGraph treats a thread as one ongoing
    execution: starting a 'fresh' run on a thread whose previous run stopped
    part-way makes the graph CONTINUE from that point instead of starting over,
    silently reusing the earlier run's concepts and questions. Giving every fresh
    run its own thread guarantees a clean start, while --resume reads the id
    below to pick up exactly where the last run left off.
    """
    return f"run_{int(time.time())}"


def read_last_thread_id() -> Optional[str]:
    """Returns the thread id of the most recent run, or None if there isn't one."""
    try:
        with open(LAST_THREAD_FILE, "r", encoding="utf-8") as f:
            value = f.read().strip()
            return value or None
    except OSError:
        return None


def write_last_thread_id(thread_id: str) -> None:
    """Records the current thread id so a later --resume can find it."""
    try:
        os.makedirs("outputs", exist_ok=True)
        with open(LAST_THREAD_FILE, "w", encoding="utf-8") as f:
            f.write(thread_id)
    except OSError as exc:
        print(f"[Notice] Could not record thread id for --resume: {exc}")


def handle_visualization(graph) -> None:
    """Renders the LangGraph topology to outputs/pipeline_graph.png with text fallback."""
    os.makedirs("outputs", exist_ok=True)
    png_path = "outputs/pipeline_graph.png"
    mmd_path = "outputs/pipeline_graph.mmd"

    print("\n" + "=" * 60)
    print("GRAPH TOPOLOGY VISUALIZATION")
    print("=" * 60)

    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        print(f"[Saved] Visual graph diagram generated: {png_path} ({len(png_bytes):,} bytes)")
    except Exception as exc:
        print(f"[Notice] PNG rendering unavailable ({exc}). Exporting Mermaid text diagram...")
        try:
            mermaid_src = graph.get_graph().draw_mermaid()
            with open(mmd_path, "w", encoding="utf-8") as f:
                f.write(mermaid_src)
            print(f"[Saved] Mermaid definition written to: {mmd_path}")
            print("\nMermaid Source:\n" + mermaid_src)
        except Exception as err2:
            print(f"[Warning] Could not extract Mermaid graph: {err2}")

    print("=" * 60 + "\n")


def print_combined_summary(state: PipelineState, total_time: float) -> None:
    """Prints the comprehensive final summary across all three agents."""
    questions = state.get("questions", [])
    concepts = state.get("concepts", [])
    check_results = state.get("check_results", {})
    balance_note = state.get("balance_note", "No balance evaluation recorded.")
    actual_counts = state.get("actual_counts", {})
    target_counts = state.get("target_counts", {})
    total_marks = sum(q.marks for q in questions) if questions else 0
    total_calls = state.get("llm_calls", get_llm_call_count())
    selected_model = state.get("selected_model", "Unknown")
    model_reason = state.get("model_selection_reason", "")

    # Category breakdowns
    marks_by_type: dict[str, int] = {}
    for q in questions:
        marks_by_type[q.type.value] = marks_by_type.get(q.type.value, 0) + q.marks

    marks_by_difficulty: dict[str, int] = {}
    for q in questions:
        marks_by_difficulty[q.difficulty.value] = marks_by_difficulty.get(q.difficulty.value, 0) + q.marks

    represented_concepts = {q.concept for q in questions}
    total_concept_count = len(concepts) if concepts else len(represented_concepts)
    concept_pct = (len(represented_concepts) / total_concept_count) * 100 if total_concept_count else 100.0

    print("\n" + "=" * 70)
    print("MULTI-AGENT CLASSROOM ASSESSMENT GENERATOR — COMBINED FINAL SUMMARY")
    print("=" * 70)
    print(f"Active LLM Model       : {selected_model}")
    if model_reason:
        print(f"Selection Justification: {model_reason}")
    print(f"Total Wall-Clock Time  : {total_time:.1f}s ({total_time / 60:.2f} min)")
    print(f"Total LLM Calls        : {total_calls}")
    print(f"Total Questions        : {len(questions)}")
    print(f"Total Assessment Marks : {total_marks} Marks")
    print(f"Cognitive Balance      : {balance_note}")

    print("\n--- The 5 Safety & Quality Checks Summary ---")
    sch = check_results.get("schema", {})
    print(f"Check 1 [Schema Validation]     : {sch.get('passed', len(questions))}/{sch.get('total', len(questions))} passed schema")
    coh = check_results.get("coherence", {})
    print(f"Check 2 [Entity Coherence]      : {coh.get('passed', 0)} passed, {coh.get('flagged_for_regen', 0)} regenerated, {coh.get('needs_review', 0)} flagged review")
    prem = check_results.get("premise", {})
    print(f"Check 3 [Premise & Trace Check] : {prem.get('passed', 0)} passed, {prem.get('latent_bugs', 0)} latent bugs, {prem.get('needs_review', 0)} false premises")
    ver = check_results.get("verification", {})
    print(f"Check 4 [Self-Verification Pass]: {ver.get('verified', 0)} verified, {ver.get('needs_review', 0)} needs review, {ver.get('unverified_theory', 0)} open theory")
    grd = check_results.get("grounding", {})
    print(f"Check 5 [Notes Grounding Check] : {grd.get('passed', 0)} consistent, {grd.get('conflicts_flagged', 0)} conflicts flagged")

    print("\n--- Marks Allocation by Question Type ---")
    for t_name in ["MCQ", "CODE", "THEORY"]:
        print(f"  - {t_name:6s}: {marks_by_type.get(t_name, 0):2d} Marks")

    print("\n--- Marks Allocation by Difficulty Band ---")
    for d_name in ["EASY", "MEDIUM", "HARD"]:
        cnt = actual_counts.get(d_name, sum(1 for q in questions if q.difficulty.value == d_name))
        m = marks_by_difficulty.get(d_name, 0)
        print(f"  - {d_name:6s}: {cnt} Questions, {m:2d} Marks")

    print(f"\nSyllabus Concept Coverage: {len(represented_concepts)}/{total_concept_count} Concepts ({concept_pct:.1f}%)")

    needs_review = [q for q in questions if q.verification_status == "needs_review"]
    print(f"\nQuestions Requiring Instructor Review ({len(needs_review)}):")
    if needs_review:
        for rq in needs_review:
            print(f"  - [{rq.id}] [{rq.type.value}] '{rq.concept}':")
            print(f"      Reason: {rq.verification_note}")
    else:
        print("  None — all questions passed independent verification or are open-ended theory questions.")

    errors = state.get("errors", [])
    if errors:
        print(f"\nAccumulated Non-Fatal Pipeline Notices ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")

    print("\n--- Generated Assessment Deliverables ---")
    print(f"  1. outputs/concepts.json        (Agent 1 Extracted & Merged Concepts)")
    print(f"  2. outputs/questions.json       (Agent 2 Verified Question Pool)")
    print(f"  3. outputs/final_quiz.json      (Agent 3 Master Assessment Quiz Model)")
    print(f"  4. outputs/marking_scheme.json  (Agent 3 Itemized Criteria Marking Data)")
    print(f"  5. outputs/question_paper.txt   (Student-Facing Examination Paper)")
    print(f"  6. outputs/marking_scheme.txt   (Examiner-Facing Full Marking Scheme)")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Automated Classroom Quiz & Assessment Generator (LangGraph Orchestration)"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Paths to PDF/DOCX files or folders containing course documents",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution from the last completed LangGraph checkpoint in outputs/checkpoints.db",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=12,
        help="Target total questions to generate (default: 12)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render LangGraph topology diagram to outputs/pipeline_graph.png and print Mermaid source",
    )
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # 1. Checkpoint setup
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_pipeline_graph(checkpointer=checkpointer)

    # Handle --visualize flag
    if args.visualize:
        handle_visualization(graph)
        if not args.paths and not args.resume:
            sys.exit(0)

    # Validate inputs
    if not args.paths and not args.resume:
        print("\n[Error] No input files or directories specified.")
        print("Usage examples:")
        print("  python main.py sample_pdfs/DS-UNIT-1.pdf sample_pdfs/DS-UNIT-2.pdf \"sample_pdfs/Double Ended Queue.docx\"")
        print("  python main.py sample_pdfs/")
        print("  python main.py --resume")
        print("  python main.py --visualize")
        sys.exit(1)

    # Resuming continues the previous run's thread; a fresh run gets its own, so
    # it can never inherit a half-finished graph from an earlier failed attempt.
    if args.resume:
        thread_id = read_last_thread_id()
        if not thread_id:
            print(f"\n[Error] Cannot resume: no previous run recorded in '{LAST_THREAD_FILE}'.")
            print("Please execute a normal run first before attempting to resume.")
            sys.exit(1)
    else:
        thread_id = new_thread_id()
        write_last_thread_id(thread_id)

    config = {"configurable": {"thread_id": thread_id}}
    start_time = time.time()

    print("=" * 60)
    print("MULTI-AGENT QUIZ GENERATOR: LANGGRAPH STATE MACHINE")
    print("=" * 60)

    # Probe and initialize client
    client, provider, selected_model, model_reason = get_shared_client()
    print(f"LLM Provider    : {provider.upper()}")
    print(f"Selected Model  : {selected_model}")
    print(f"Selection Reason: {model_reason}")
    print("=" * 60)

    # Determine whether we are starting fresh or resuming
    if args.resume:
        st = graph.get_state(config)
        if not st or not st.values:
            print(f"\n[Error] Cannot resume: no checkpoint found for thread '{thread_id}' in '{CHECKPOINT_DB_PATH}'.")
            print("Please execute a normal run first before attempting to resume.")
            sys.exit(1)
        next_nodes = st.next
        if not next_nodes:
            print(f"\n[Info] Pipeline already completed all nodes in the checkpoint.")
            print_combined_summary(st.values, time.time() - start_time)
            sys.exit(0)
        print(f"\n[Resume Mode] Resuming from checkpoint. Next node to execute: {next_nodes}")
        initial_input = None
    else:
        # Fresh run: Clear previous thread state if any
        initial_input: PipelineState = {
            "input_paths": args.paths,
            "documents": [],
            "concepts": [],
            "quiz_spec": QuizSpec(
                total_questions=args.questions,
                mcq_ratio=0.5,
                code_ratio=0.25,
                theory_ratio=0.25,
            ),
            "generation_plan": [],
            "questions": [],
            "failed_question_ids": [],
            "check_results": {},
            "regeneration_attempts": {},
            "difficulty_target": DifficultyTarget(
                easy_ratio=0.3,
                medium_ratio=0.5,
                hard_ratio=0.2,
            ),
            "final_quiz": None,
            "errors": [],
            "llm_calls": 0,
            "started_at": start_time,
            "selected_model": selected_model,
            "model_selection_reason": model_reason,
            "chunks": [],
            "chunk_results": [],
            "coherence_feedback": {},
            "breakdowns": {},
            "mismatch_reports": [],
            "breakdown_mismatch_detected": False,
            "breakdown_retried": False,
            "breakdown_run_count": 0,
            "balance_note": "",
            "target_counts": {},
            "actual_counts": {},
        }

    # Execute stream
    print("\nStarting execution through LangGraph state machine...\n")
    try:
        for event in graph.stream(initial_input, config=config, stream_mode="updates"):
            for node_name, state_update in event.items():
                current_calls = get_llm_call_count()
                print(f"[NODE COMPLETED] {node_name} | Running LLM Calls: {current_calls}")
    except KeyboardInterrupt:
        print("\n[Interrupted] Execution paused by user. Checkpoint saved; use --resume to continue.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[Error] Pipeline execution failed at a node: {exc}")
        print("State has been checkpointed. Use 'python main.py --resume' to re-run from the last saved state.")
        raise exc

    # Retrieve final state
    final_checkpoint = graph.get_state(config)
    final_state = final_checkpoint.values
    elapsed_time = time.time() - start_time

    # Print combined final summary
    print_combined_summary(final_state, elapsed_time)


if __name__ == "__main__":
    main()
