"""
Step 1/3B Orchestrator: Multi-Agent Automated Classroom Quiz Generator
Extracts text across multiple documents and formats (PDF, DOCX), performs per-document
cleaning, doc_kind detection, and isolated chunking, queries LLM per chunk with rate
limiting, and performs global deduplication and cross-document concept tracking.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from extractors import Chunk, DocumentContent, extract_document, build_chunks
from schemas import Concept, ConceptSource

# ---------------------------------------------------------------------------
# LLM Configuration
# Model name kept as a constant at the top for easy editing.
# For OpenAI: e.g. "gpt-4o-mini", "gpt-4o"
# For Groq:   e.g. "openai/gpt-oss-120b", "llama-3.3-70b-versatile"
# ---------------------------------------------------------------------------
MODEL_PRIORITY = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
MODEL_NAME = "openai/gpt-oss-120b"


def select_active_model(client: object) -> tuple[str, str]:
    """
    On startup, tests each model in MODEL_PRIORITY with a tiny test call.
    If a model is rate limited with wait > 60s, or fails/not found, moves to the next.
    Returns (selected_model_name, selection_reason).
    Uses that single model for the entire run.
    """
    reasons: list[str] = []
    for model_name in MODEL_PRIORITY:
        try:
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            reason = f"Verified responsive on startup (prior attempts: {'; '.join(reasons) if reasons else 'none'})"
            return model_name, reason
        except Exception as err:
            err_msg = str(err)
            wait = parse_retry_after(err_msg)
            if wait and wait > 60.0:
                reasons.append(f"{model_name} rate limited (wait {wait:.0f}s > 60s)")
            elif "not_found" in err_msg.lower() or "does not exist" in err_msg.lower():
                reasons.append(f"{model_name} not found on account")
            else:
                reasons.append(f"{model_name} error: {err_msg[:60]}")

    fallback = MODEL_PRIORITY[-1]
    return fallback, f"Fallback to {fallback} after checks: {'; '.join(reasons)}"

# ---------------------------------------------------------------------------
# Document Type Hints
# ---------------------------------------------------------------------------
SLIDES_HINT = (
    "The text comes from lecture slides, so points can be terse, fragmentary bullets. "
    "Infer the full concept each bullet points to, and use your own subject knowledge "
    "to write complete, substantive explanations that would make sense to a student."
)

PROSE_HINT = (
    "The text is comprehensive textbook or technical prose. "
    "Synthesize the key technical topics covered into clear, self-contained educational concepts."
)

CODE_HINT = (
    "This text is a source code listing or implementation file. "
    "Extract the core data-structure concepts and algorithms the code DEMONSTRATES "
    "(such as 'Double-Ended Queue (Deque)'), not the mechanics of individual functions, "
    "variable names, syntax, or language-specific boilerplate."
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def get_llm_client() -> tuple[object, str]:
    """
    Loads environment variables and initializes either an OpenAI or Groq client.
    Never prints or logs the API key value.

    Returns:
        tuple of (client_instance, provider_name)
    """
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    # If OPENAI_API_KEY is present, use OpenAI
    if openai_key and openai_key.strip():
        try:
            from openai import OpenAI
            return OpenAI(api_key=openai_key.strip()), "openai"
        except ImportError:
            raise RuntimeError("The 'openai' package is required but not installed.")

    # Alternatively, if GROQ_API_KEY is present, use Groq
    if groq_key and groq_key.strip():
        try:
            from groq import Groq
            return Groq(api_key=groq_key.strip()), "groq"
        except ImportError:
            raise RuntimeError("The 'groq' package is required but not installed.")

    raise ValueError(
        "Missing API key! Neither OPENAI_API_KEY nor GROQ_API_KEY was found in .env."
    )


def parse_retry_after(err_str: str) -> Optional[float]:
    """
    Parses the suggested wait time out of an API error message (e.g. 'try again in 22.5ms'
    or 'retry after 3.5s'). Returns seconds as float, or None if not found.
    """
    match = re.search(
        r"(?:try again in|retry after|retry-after)[^\d]*([\d.]+)\s*(ms|milliseconds|s|sec|seconds|m|minutes)?",
        err_str,
        re.IGNORECASE,
    )
    if match:
        val = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        if unit.startswith("ms"):
            seconds = val / 1000.0
        elif unit.startswith("m") and not unit.startswith("ms"):
            seconds = val * 60.0
        else:
            seconds = val
        # Add a buffer of 0.5s with a minimum wait of 1.0s to ensure token bucket refill
        return max(1.0, seconds + 0.5)
    return None


def call_llm_with_retry(
    client: object,
    messages: list[dict],
    model: str = MODEL_NAME,
    temperature: float = 0.2,
    max_retries: int = 4,
    response_format: Optional[dict] = None,
) -> Optional[str]:
    """
    Executes a chat completion call with automatic retry and backoff on HTTP 429 / Rate Limit errors.
    Parses the suggested wait time from the error message when present, or uses exponential backoff
    (2s, 4s, 8s, 16s). Retries up to max_retries times before giving up.
    Gracefully catches and falls back if response_format is rejected by the provider.
    """
    backoff_schedule = [2.0, 4.0, 8.0, 16.0]
    current_response_format = response_format

    for attempt in range(max_retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if current_response_format is not None:
                kwargs["response_format"] = current_response_format

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as err:
            err_msg = str(err)
            # If provider rejects response_format, fall back and retry without it
            if current_response_format is not None and any(
                term in err_msg.lower() for term in (
                    "response_format",
                    "json_object",
                    "unsupported parameter",
                    "json_validate_failed",
                    "failed to validate json",
                )
            ):
                print(f"  [Warning] response_format parameter rejected ({err_msg[:80]}...). Continuing without response_format...")
                current_response_format = None
                continue

            is_429 = (
                getattr(err, "status_code", None) == 429
                or "429" in err_msg
                or "rate limit" in err_msg.lower()
                or "rate_limit_exceeded" in err_msg.lower()
            )

            if is_429 and attempt < max_retries:
                retry_num = attempt + 1
                parsed_wait = parse_retry_after(err_msg)
                if parsed_wait is not None:
                    wait_time = min(parsed_wait, 60.0)
                else:
                    wait_time = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]

                print(f"  Rate limited, retrying in {wait_time:.1f}s (attempt {retry_num}/{max_retries})...")
                time.sleep(wait_time)
                continue
            else:
                if is_429:
                    print(f"  [Error] Rate limit retries exhausted ({max_retries}/{max_retries}).")
                else:
                    print(f"  [Error] LLM call failed with error: {type(err).__name__}: {err_msg}")
                return None

    return None


def parse_json_response(raw: str) -> Optional[dict]:
    """
    Defensively parses a JSON object from an LLM response string.
    Tries 5 strategies in order:
    1. Direct json.loads(raw)
    2. Strip markdown fences (```json and ```), then json.loads
    3. Find the first '{' and the last '}', take that substring, then json.loads
    4. On that substring, remove trailing commas before '}' or ']' with regex, then json.loads
    5. Returns None if all fail.
    """
    if not raw or not isinstance(raw, str):
        return None

    cleaned = raw.strip()

    # 1. Direct json.loads
    try:
        val = json.loads(cleaned)
        if isinstance(val, dict):
            return val
    except Exception:
        pass

    # 2. Strip markdown code fences
    no_fences = cleaned
    if no_fences.startswith("```"):
        no_fences = re.sub(r"^```(?:json)?\s*", "", no_fences)
        no_fences = re.sub(r"\s*```$", "", no_fences).strip()
        try:
            val = json.loads(no_fences)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

    # 3. Find first '{' and last '}'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_substr = cleaned[first_brace : last_brace + 1]
        try:
            val = json.loads(json_substr)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

        # 4. Remove trailing commas before '}' or ']'
        fixed_commas = re.sub(r",\s*([\]}])", r"\1", json_substr)
        try:
            val = json.loads(fixed_commas)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

    # 5. Return None if all strategies fail
    return None


QUALIFIERS = {
    "algorithm",
    "implementation",
    "implementations",
    "adt",
    "solution",
    "solutions",
    "technique",
    "techniques",
    "operation",
    "operations",
    "variant",
    "variants",
    "type",
    "types",
}


def concept_broadness_score(name: str) -> tuple[int, int]:
    """
    Ranks concept names to select the broadest concept as the survivor:
    1. Names without qualifiers (e.g. 'Algorithm', 'Implementation', 'ADT', 'Operations') rank higher.
    2. Shorter names rank higher (e.g. 'Recursion' is broader than 'Recursion and Types of Recursion').
    """
    words = re.findall(r"\b\w+\b", name.lower())
    qualifier_count = sum(1 for w in words if w in QUALIFIERS)
    return (qualifier_count, len(name))


def collect_file_paths(raw_args: list[str]) -> list[Path]:
    """
    Resolves input arguments into a list of file paths.
    If an argument is a directory, expands to non-recursive supported files in that folder.
    Skips unsupported extensions with a clear warning.
    """
    collected: list[Path] = []

    for arg in raw_args:
        p = Path(arg)
        if not p.exists():
            print(f"[Warning] Path does not exist, skipping: '{arg}'")
            continue

        if p.is_dir():
            dir_files = [f for f in sorted(p.iterdir()) if f.is_file() and not f.name.startswith(".")]
            print(f"\nScanning directory '{p}': found {len(dir_files)} direct file(s).")
            for f in dir_files:
                if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    collected.append(f)
                else:
                    print(f"  [Skip] Unsupported extension for file: '{f.name}' (supported: .pdf, .docx)")
        elif p.is_file():
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                collected.append(p)
            else:
                print(f"[Warning] Skipping '{p.name}': unsupported extension '{p.suffix}' (supported: .pdf, .docx)")

    return collected


def report_quality(doc: DocumentContent) -> None:
    """
    Prints an extraction quality summary and displays a snippet of the cleaned text.

    Args:
        doc: Extracted DocumentContent object.
    """
    all_text = "\n\n".join(doc.blocks)
    char_count = len(all_text)
    word_count = len(all_text.split())
    empty_count = len(doc.empty_block_indices)

    print("\n" + "=" * 60)
    print("DOCUMENT EXTRACTION QUALITY REPORT")
    print("=" * 60)
    print(f"File Name:        {doc.filename}")
    print(f"Total Blocks:     {doc.block_count}")
    print(f"Empty Blocks:     {empty_count}")
    print(f"Detected Kind:    {doc.doc_kind.upper()}")
    print(f"Character Count:  {char_count:,}")
    print(f"Word Count:       {word_count:,}")

    if doc.empty_block_indices:
        # Show first 15 empty indices if there are many to prevent flooding
        if len(doc.empty_block_indices) <= 15:
            nums_str = ", ".join(str(n) for n in doc.empty_block_indices)
        else:
            nums_str = ", ".join(str(n) for n in doc.empty_block_indices[:15]) + f"... ({len(doc.empty_block_indices)} total)"
        print(f"\nEmpty blocks (no extractable text): {nums_str}")
        print(
            "  Note: In PDFs, these are typically image/diagram slides; in DOCX, empty spacing lines. "
            "OCR / vision-based extraction is out of scope for this step."
        )

    # Check if more than ~30% of blocks are empty
    if doc.block_count > 0:
        empty_ratio = empty_count / doc.block_count
        if empty_ratio > 0.30:
            print("\n[WARNING] More than 30% of blocks are empty "
                  f"({empty_count}/{doc.block_count} = {empty_ratio:.1%}).")

    print("\n--- FIRST 500 CHARACTERS OF CLEANED TEXT ---")
    snippet = all_text[:500] if all_text else "[No text extracted]"
    print(snippet)
    if len(all_text) > 500:
        print("...\n[Preview truncated at 500 characters]")
    print("=" * 60 + "\n")


def print_chunking_summary(chunks: list[Chunk], filename: str) -> None:
    """Prints a concise breakdown of document chunking."""
    print("=" * 60)
    print(f"CHUNKING SUMMARY FOR: {filename}")
    print("=" * 60)
    print(f"Total Chunks Generated: {len(chunks)}")
    for i, c in enumerate(chunks, start=1):
        print(f"  Chunk {i}: {c.word_count} words | Covers blocks {c.block_start} to {c.block_end}")
    print("=" * 60 + "\n")


def extract_concepts_from_chunk(
    chunk: Chunk,
    doc: DocumentContent,
    client: object,
    provider: str,
    global_chunk_num: int,
    total_global_chunks: int,
) -> list[Concept]:
    """
    Invokes the LLM on a single chunk to extract 5-8 technical concepts.
    Adapts system prompt based on doc.doc_kind. Returns empty list on failure.
    """
    if doc.doc_kind == "code":
        hint = CODE_HINT
    elif doc.doc_kind == "slides":
        hint = SLIDES_HINT
    else:
        hint = PROSE_HINT
    system_prompt = f"You are an expert educator. {hint}"

    user_prompt = (
        "List UP TO 8 of the most important TECHNICAL CONCEPTS found in this text section. "
        "Return fewer if the chunk genuinely contains fewer — do not pad the list to reach a number.\n\n"
        "Granularity & Scope Guidelines:\n"
        "1. Extract concepts a student would be EXAMINED on — the standard topics that appear in a syllabus or exam paper (e.g. 'Double-Ended Queue (Deque)', 'Recursion', 'Stack (LIFO) ADT', 'Priority Queue').\n"
        "2. Prefer one substantial, cohesive concept over several narrow sub-topics or algorithms. For example, return 'Recursion' covering its various forms (direct, indirect, tail), rather than separate entries for each; return 'Double-Ended Queue (Deque)' rather than separate entries for each insert/delete function.\n"
        "3. Do NOT extract individual functions or routines as separate concepts (e.g. do NOT return 'insert_front', 'delete_rear', or 'Deletion from front'). Group them under the parent data structure or operations topic.\n"
        "4. Do NOT extract incidental implementation details from code listings: specific variable names, #define constants, temporary pointer variables, or language mechanics used only as a means to an end.\n"
        "5. Do NOT extract document structure: section headings, course outcomes, references, syllabus items, or introductory meta-text.\n"
        "6. Each explanation must be one or two complete, substantive sentences that would make sense to a student encountering the concept for the first time — do NOT simply restate bullets; use your subject knowledge to elaborate.\n\n"
        "Return ONLY valid JSON in this shape: "
        "{\"concepts\": [{\"name\": \"...\", \"explanation\": \"...\"}]}\n\n"
        f"Text:\n{chunk.text}"
    )

    print(f"Calling LLM ({provider} / {MODEL_NAME}) for Chunk {global_chunk_num}/{total_global_chunks} "
          f"[{doc.filename}: {chunk.word_count} words, blocks {chunk.block_start}-{chunk.block_end}]...")

    content = call_llm_with_retry(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=MODEL_NAME,
        temperature=0.2,
    )

    if not content:
        print(f"  [Warning] Chunk {global_chunk_num} ({doc.filename}): No response from LLM. Skipping.")
        return []

    # Clean code fences
    raw_json = content.strip()
    if raw_json.startswith("```"):
        raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json)
        raw_json = re.sub(r"\s*```$", "", raw_json)
        raw_json = raw_json.strip()

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as decode_err:
        print(f"  [Warning] Chunk {global_chunk_num} ({doc.filename}): Response could not be parsed as JSON ({decode_err}). Skipping.")
        return []

    if not isinstance(data, dict) or "concepts" not in data or not isinstance(data["concepts"], list):
        print(f"  [Warning] Chunk {global_chunk_num} ({doc.filename}): Unexpected JSON structure. Skipping.")
        return []

    source = ConceptSource(
        filename=doc.filename,
        block_start=chunk.block_start,
        block_end=chunk.block_end,
    )

    results: list[Concept] = []
    for item in data["concepts"]:
        if isinstance(item, dict) and "name" in item and "explanation" in item:
            name = str(item["name"]).strip()
            explanation = str(item["explanation"]).strip()
            if name and explanation:
                results.append(Concept(name=name, explanation=explanation, sources=[source]))

    return results


def normalize_concept_name(name: str) -> str:
    """
    Normalizes a concept name for deduplication comparison ONLY.

    Replaces ASCII hyphens, unicode dashes/hyphens (e.g. non-breaking hyphen U+2011,
    en-dash, em-dash), and underscores with spaces, removes other punctuation,
    converts to lowercase, and collapses extra whitespace.

    Example:
        'Double-ended Queue (Deque)' and 'Double Ended Queue (Deque)'
        both normalize to 'double ended queue deque', ensuring they correctly
        merge into one concept across documents instead of duplicating.
    """
    # Replace all hyphen variants (ASCII hyphen, unicode dashes/hyphens U+2010-U+2015, minus U+2212) and underscores with spaces
    normalized = re.sub(r"[\u2010-\u2015\u2212\-_]", " ", name.lower())
    # Strip remaining punctuation characters
    normalized = re.sub(r"[^\w\s]", "", normalized)
    # Collapse multiple consecutive whitespace characters and trim
    return re.sub(r"\s+", " ", normalized).strip()


def merge_concepts_string_pass(chunk_results: list[list[Concept]]) -> list[Concept]:
    """
    Pass 1 — Cheap string-based deduplication pass.
    Catches exact and near-exact repeats (ignoring casing, punctuation, hyphens, and whitespace),
    reduces duplicate noise for the subsequent semantic merge, and costs zero LLM tokens.
    """
    merged_map: dict[str, Concept] = {}

    for concepts in chunk_results:
        for c in concepts:
            norm_key = normalize_concept_name(c.name)
            if norm_key not in merged_map:
                merged_map[norm_key] = Concept(
                    name=c.name,
                    explanation=c.explanation,
                    sources=list(c.sources),
                    subtopics=list(c.subtopics),
                )
            else:
                existing = merged_map[norm_key]
                # Keep longer explanation
                if len(c.explanation) > len(existing.explanation):
                    existing.explanation = c.explanation
                # Union sources avoiding duplicate records
                for src in c.sources:
                    if not any(
                        s.filename == src.filename
                        and s.block_start == src.block_start
                        and s.block_end == src.block_end
                        for s in existing.sources
                    ):
                        existing.sources.append(src)
                # Union subtopics
                for sub in c.subtopics:
                    if sub not in existing.subtopics:
                        existing.subtopics.append(sub)

    return list(merged_map.values())


def semantic_merge_concepts(
    concepts: list[Concept],
    client: object,
    provider: str,
) -> tuple[list[Concept], list[tuple[str, list[str]]]]:
    """
    Pass 2 — Semantic merge using an LLM with defensive parsing and structured output.

    Sends numbered concept names to the LLM to identify groups that refer
    to the same underlying concept or a broad concept with its specific sub-cases/examples.
    The LLM designates the 'parent' (broadest concept) and 'members' (narrower concepts to fold).

    Defensive layers:
    - Step 1: Raw response printed always (on success and failure) under clear headers.
    - Step 2: Uses Groq structured output mode (response_format={"type": "json_object"}).
    - Step 3: Minimal, clean prompt asking only for concept grouping integers with parent/members hierarchy.
    - Step 4: Multi-stage defensive JSON parser (parse_json_response).
    - Step 5: Up to 2 retries on JSON parse failure with error-correcting prompt.
    - Step 6: Strict validation of parent/member indices, deduplication, and parent-as-survivor selection.

    Returns:
        tuple of (final_concepts_list, list_of_group_merges) where each group merge is (parent_name, [member_names])
    """
    if len(concepts) <= 1:
        return concepts, []

    # Numbered names list (1-indexed, names ONLY — no explanations, no sources)
    numbered_names = "\n".join(f"{i}. {c.name}" for i, c in enumerate(concepts, start=1))

    # STEP 3: Minimal prompt requesting JSON object with parent and members
    system_prompt = (
        "You are a curriculum expert. You return only JSON objects. "
        "You never write explanations, comments, or any text outside the JSON object."
    )

    base_user_prompt = (
        "Below is a numbered list of concept names extracted from course materials.\n\n"
        f"{numbered_names}\n\n"
        "Grouping rules:\n"
        "- Group numbers that refer to the SAME concept worded differently.\n"
        "- Group a broad concept with its specific examples, sub-cases, or operations, where the specific ones would not be set as separate exam topics on their own.\n"
        "  For each group, specify 'parent' as the index of the broadest concept — the one that would appear as a topic on a syllabus — and 'members' as an array of the narrower concept indices folded into it.\n"
        "  If concepts 3, 7 and 12 are 'Recursion', 'Tail Recursion' and 'Recursive Factorial', the parent is 3 because 'Recursion' is the syllabus topic and the others are specific cases of it.\n"
        "- Concepts describing the internal mechanics of a code listing — index bookkeeping, boundary checks, variable initialization, individual helper functions — should be grouped under the data structure or implementation they belong to. They are not standalone exam topics.\n"
        "- Do NOT group different data structures (keep Stack, Queue ADT, Circular Queue, Priority Queue, and Double-Ended Queue distinct).\n"
        "- Do NOT group a general data structure ADT (like Stack ADT) with specific application algorithms (like expression conversion or evaluation).\n"
        "- Do NOT group two implementations being compared (e.g. Array-Based vs Linked-List Queue).\n"
        "- When unsure, do not group.\n\n"
        "Target: approximately 14 to 18 final consolidated concepts.\n\n"
        'Return a JSON object with a single key "groups" whose value is an array of objects of the form: '
        '{"parent": <int>, "members": [<int>, ...]}. Example: {"groups": [{"parent": 9, "members": [5, 10]}]}. '
        'If nothing should be grouped, return {"groups": []}. Output nothing except this JSON object.'
    )

    print(f"\nRunning Pass 2: Semantic Deduplication LLM call ({provider} / {MODEL_NAME}) across {len(concepts)} concepts...")

    # STEP 5: Retry on parse failure (up to 2 retries -> 3 attempts total)
    max_parse_retries = 2
    user_prompt = base_user_prompt
    raw_response: Optional[str] = None
    parsed_data: Optional[dict] = None

    for attempt in range(1, max_parse_retries + 2):
        if attempt > 1:
            print(f"\n  Retrying semantic merge due to parse failure (attempt {attempt}/{max_parse_retries + 1})...")

        # STEP 2: Use response_format={"type": "json_object"}
        raw_response = call_llm_with_retry(
            client=client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=MODEL_NAME,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        # STEP 1: PRINT THE RAW RESPONSE, ALWAYS
        print("\n--- RAW SEMANTIC MERGE RESPONSE ---")
        if raw_response:
            print(raw_response[:2000])
            if len(raw_response) > 2000:
                print(f"... [Truncated {len(raw_response) - 2000} characters]")
        else:
            print("[Empty or None Response]")
        print("--- END RAW RESPONSE ---")

        if not raw_response:
            continue

        # STEP 4: Parse defensively
        parsed_data = parse_json_response(raw_response)
        if parsed_data and "groups" in parsed_data and isinstance(parsed_data["groups"], list):
            break

        # Append retry clarification for next attempt
        user_prompt = (
            base_user_prompt
            + '\n\nYour previous reply could not be parsed as JSON. '
            + 'Return ONLY a JSON object of the form {"groups": [{"parent": 9, "members": [5, 10]}]} with no other text.'
        )

    if not parsed_data or "groups" not in parsed_data or not isinstance(parsed_data["groups"], list):
        print(f"\n[Warning] Semantic deduplication failed after {max_parse_retries + 1} attempt(s). Falling back to Pass 1 results.")
        return concepts, []

    # STEP 6: Validate the groups before applying
    raw_groups = parsed_data["groups"]
    n_concepts = len(concepts)
    seen_indices: set[int] = set()
    validated_groups: list[dict[str, any]] = []

    for grp_idx, item in enumerate(raw_groups, start=1):
        if not isinstance(item, dict):
            print(f"  [Warning] Discarded group #{grp_idx}: expected dict with 'parent' and 'members', got {type(item).__name__}")
            continue

        parent = item.get("parent")
        members = item.get("members")

        if not isinstance(parent, int) or not (1 <= parent <= n_concepts):
            print(f"  [Warning] Discarded group #{grp_idx}: invalid parent index '{parent}' (must be 1..{n_concepts})")
            continue

        if not isinstance(members, list):
            print(f"  [Warning] Discarded group #{grp_idx}: 'members' must be a list, got {type(members).__name__}")
            continue

        if parent in seen_indices:
            print(f"  [Warning] Discarded group #{grp_idx}: parent {parent} ('{concepts[parent - 1].name}') was already used in an earlier group")
            continue

        valid_members: list[int] = []
        for m in members:
            if not isinstance(m, int):
                print(f"  [Warning] Discarded non-integer member '{m}' in group #{grp_idx}")
                continue
            if not (1 <= m <= n_concepts):
                print(f"  [Warning] Discarded out-of-range member index {m} in group #{grp_idx}")
                continue
            if m == parent:
                print(f"  [Warning] Discarded self-reference: parent index {parent} ('{concepts[parent - 1].name}') cannot be in its own members")
                continue
            if m in seen_indices:
                print(f"  [Warning] Discarded member index {m} ('{concepts[m - 1].name}') in group #{grp_idx}: already used in an earlier group")
                continue

            valid_members.append(m)

        if not valid_members:
            print(f"  [Warning] Discarded group #{grp_idx}: no valid members remaining for parent {parent} ('{concepts[parent - 1].name}')")
            continue

        seen_indices.add(parent)
        for m in valid_members:
            seen_indices.add(m)

        validated_groups.append({"parent": parent, "members": valid_members})

    # Apply validated groups
    merged_away_indices: set[int] = set()
    survivors_map: dict[int, Concept] = {}
    group_merge_records: list[tuple[str, list[str]]] = []

    for grp in validated_groups:
        p_idx = grp["parent"]
        m_indices = grp["members"]

        parent_concept = concepts[p_idx - 1]
        survivor = Concept(
            name=parent_concept.name,
            explanation=parent_concept.explanation,  # unmodified!
            sources=list(parent_concept.sources),
            subtopics=list(parent_concept.subtopics),
        )

        merged_member_names: list[str] = []

        for m_idx in m_indices:
            merged_away_indices.add(m_idx)
            member_concept = concepts[m_idx - 1]
            merged_member_names.append(member_concept.name)

            # Union sources into survivor
            for src in member_concept.sources:
                if not any(
                    s.filename == src.filename
                    and s.block_start == src.block_start
                    and s.block_end == src.block_end
                    for s in survivor.sources
                ):
                    survivor.sources.append(src)

            # Append member's name and its subtopics to survivor's subtopics list
            if member_concept.name not in survivor.subtopics:
                survivor.subtopics.append(member_concept.name)
            for sub in member_concept.subtopics:
                if sub not in survivor.subtopics:
                    survivor.subtopics.append(sub)

        survivors_map[p_idx] = survivor
        group_merge_records.append((parent_concept.name, merged_member_names))

    # Construct final concept list preserving document order
    final_list: list[Concept] = []
    for i in range(1, n_concepts + 1):
        if i in merged_away_indices:
            continue
        if i in survivors_map:
            final_list.append(survivors_map[i])
        else:
            final_list.append(concepts[i - 1])

    return final_list, group_merge_records


def save_and_display_final_concepts(concepts: list[Concept], output_path: str = "outputs/concepts.json") -> int:
    """
    Saves merged concepts to JSON conforming to schema and displays them in terminal.
    Highlights cross-document concepts, outputs readable explanations and subtopics on their own line.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    serializable = {
        "concepts": [c.model_dump() for c in concepts]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"\nConcepts successfully saved to: {output_path}")
    print("=" * 60)
    print(f"FINAL MERGED CONCEPTS (Total: {len(concepts)})")
    print("=" * 60)

    cross_doc_count = 0
    for idx, item in enumerate(concepts, start=1):
        unique_files = {s.filename for s in item.sources}
        is_cross_doc = len(unique_files) > 1
        if is_cross_doc:
            cross_doc_count += 1

        tag = " [CROSS-DOCUMENT]" if is_cross_doc else ""
        sources_str = ", ".join(f"{s.filename} [p.{s.block_start}-{s.block_end}]" for s in item.sources)
        print(f"\n{idx}. {item.name}{tag}")
        print(f"   {item.explanation}")
        if item.subtopics:
            subtopics_str = ", ".join(item.subtopics)
            print(f"   Subtopics: {subtopics_str}")
        print(f"   Sources: {sources_str}")

    print("\n" + "=" * 60)
    print(f"CROSS-DOCUMENT CONCEPTS FOUND: {cross_doc_count} concept(s) appear across multiple files.")
    print("=" * 60)
    return cross_doc_count


def main() -> None:
    """Main orchestrator: accepts multiple files/folders, extracts, chunks, merges."""
    if len(sys.argv) < 2:
        print("\nUsage: python step1_test.py <path_or_dir_1> [path_or_dir_2 ...]")
        print("Example: python step1_test.py sample_pdfs/DS-UNIT-1.pdf sample_pdfs/DS-UNIT-2.pdf \"sample_pdfs/Double Ended Queue.docx\"")
        print("Example: python step1_test.py sample_pdfs/\n")
        sys.exit(1)

    file_paths = collect_file_paths(sys.argv[1:])
    if not file_paths:
        print("\n[Error] No valid files found to process. Aborting.")
        sys.exit(1)

    print(f"\nFound {len(file_paths)} document(s) to process:")
    for fp in file_paths:
        print(f"  - {fp}")

    # 1. Extraction layer for each document
    successful_docs: list[DocumentContent] = []
    skipped_docs: list[tuple[str, str]] = []

    for path in file_paths:
        try:
            doc = extract_document(str(path))
            if not doc.blocks:
                skipped_docs.append((path.name, "No readable text content found"))
                print(f"[Warning] Skipping '{path.name}': document contains no extractable text.")
                continue
            successful_docs.append(doc)
            report_quality(doc)

            # Save individual cleaned text file
            os.makedirs("outputs", exist_ok=True)
            cleaned_file_path = f"outputs/cleaned_{doc.filename}.txt"
            with open(cleaned_file_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(doc.blocks))
            print(f"Saved cleaned text to: {cleaned_file_path}\n")

        except Exception as e:
            skipped_docs.append((path.name, str(e)))
            print(f"[Warning] Failed to extract '{path.name}': {e}. Skipping.")

    if not successful_docs:
        print("\n[Error] No documents could be successfully extracted. Aborting concept extraction.")
        sys.exit(1)

    # 2. Build chunks per document (NEVER across documents)
    doc_chunk_plan: list[tuple[DocumentContent, list[Chunk]]] = []
    total_chunks_count = 0
    for doc in successful_docs:
        # page_map translates chunk ranges into real source page numbers, so
        # concept citations point at the page a professor would actually open.
        chunks = build_chunks(
            doc.blocks,
            target_words=1200,
            overlap_words=150,
            page_map=doc.block_page_map,
        )
        doc_chunk_plan.append((doc, chunks))
        total_chunks_count += len(chunks)
        print_chunking_summary(chunks, doc.filename)

    # 3. LLM Client initialization
    try:
        client, provider = get_llm_client()
    except (ValueError, RuntimeError) as e:
        print(f"\n[Configuration Error] {e}")
        sys.exit(1)

    # 4. Process all chunks with running progress and rate-limit delay
    all_chunk_results: list[list[Concept]] = []
    doc_concept_counts: dict[str, int] = {}
    global_chunk_index = 0

    print(f"\nStarting concept extraction across {len(successful_docs)} document(s) ({total_chunks_count} total chunks)...")

    for doc, chunks in doc_chunk_plan:
        doc_concepts_extracted = 0
        for chunk in chunks:
            global_chunk_index += 1
            concepts = extract_concepts_from_chunk(
                chunk=chunk,
                doc=doc,
                client=client,
                provider=provider,
                global_chunk_num=global_chunk_index,
                total_global_chunks=total_chunks_count,
            )
            print(f"  -> Chunk {global_chunk_index}/{total_chunks_count} produced {len(concepts)} concepts.")
            all_chunk_results.append(concepts)
            doc_concepts_extracted += len(concepts)

            # 3-second delay between calls to stay well within free-tier rate limits
            if global_chunk_index < total_chunks_count:
                time.sleep(3.0)

        doc_concept_counts[doc.filename] = doc_concepts_extracted

    # 5. Global Merge & Deduplication (Two-Pass Architecture)
    total_raw_concepts = sum(len(c) for c in all_chunk_results)

    # Pass 1: String-based exact/near-exact normalization merge
    pass1_concepts = merge_concepts_string_pass(all_chunk_results)

    # 5-second pause before semantic LLM call to replenish TPM rate limit tokens
    time.sleep(5.0)

    # Pass 2: LLM semantic deduplication
    final_concepts, group_merges = semantic_merge_concepts(pass1_concepts, client, provider)

    # 6. Reporting: Per-document, combined, and semantic deduplication summaries
    print("\n" + "=" * 60)
    print("PER-DOCUMENT PROCESSING SUMMARY")
    print("=" * 60)
    for doc, chunks in doc_chunk_plan:
        concepts_count = doc_concept_counts.get(doc.filename, 0)
        print(f"  File: {doc.filename}")
        print(f"    - Detected Kind: {doc.doc_kind.upper()}")
        print(f"    - Total Blocks:  {doc.block_count} (Empty: {len(doc.empty_block_indices)})")
        print(f"    - Chunks:        {len(chunks)}")
        print(f"    - Raw Concepts:  {concepts_count}")
        print()

    print("=" * 60)
    print("COMBINED PROCESSING & DEDUPLICATION SUMMARY")
    print("=" * 60)
    print(f"Total Documents Processed: {len(successful_docs)}")
    if skipped_docs:
        print(f"Documents Skipped:        {len(skipped_docs)}")
        for name, reason in skipped_docs:
            print(f"  - {name}: {reason}")
    else:
        print("Documents Skipped:        0")
    print(f"Total Raw Concepts:       {total_raw_concepts}")
    print(f"Concepts After Pass 1:    {len(pass1_concepts)} (string merge)")
    print(f"Concepts After Pass 2:    {len(final_concepts)} (semantic merge)")

    if group_merges:
        total_merged = sum(len(m) for _, m in group_merges)
        print(f"\nSemantic Merges Applied in Pass 2 ({len(group_merges)} group(s), {total_merged} concept(s) merged):")
        for parent_name, member_names in group_merges:
            members_str = ", ".join(f'"{m}"' for m in member_names)
            print(f'  "{parent_name}" <- {members_str}')
    else:
        print("\nSemantic Merges Applied in Pass 2: None (all concepts were considered distinct)")

    print("=" * 60)

    # 7. Save & Display Final Concepts
    save_and_display_final_concepts(final_concepts, "outputs/concepts.json")

    # 8. Distinct Exam Topics Verification
    print("\n" + "=" * 60)
    print("EXAM TOPIC SEPARATION VERIFICATION")
    print("=" * 60)
    final_names_lower = [c.name.lower() for c in final_concepts]
    has_queue_adt = any("queue adt" in n or ("queue" in n and "circular" not in n and "priority" not in n and "deque" not in n and "double" not in n) for n in final_names_lower)
    has_circular = any("circular queue" in n for n in final_names_lower)
    has_priority = any("priority queue" in n for n in final_names_lower)
    has_deque = any("deque" in n or "double-ended queue" in n or "double ended queue" in n for n in final_names_lower)

    print(f"  - Queue ADT (Linear Queue) present: {'YES' if has_queue_adt else 'NO'}")
    print(f"  - Circular Queue present:          {'YES' if has_circular else 'NO'}")
    print(f"  - Priority Queue present:          {'YES' if has_priority else 'NO'}")
    print(f"  - Double-Ended Queue present:      {'YES' if has_deque else 'NO'}")

    distinct_count = sum([has_queue_adt, has_circular, has_priority, has_deque])
    print(f"\nDistinct Queue Topics Preserved: {distinct_count}/4 distinct queue variations.")
    print("=" * 60)
    print("\nStep 3B completed successfully!")


if __name__ == "__main__":
    main()
