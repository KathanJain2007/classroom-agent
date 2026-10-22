"""
extractors.py — Extraction & Chunking Layer for Multi-Agent Quiz Generator

Provides a unified interface for reading documents into structured content blocks,
cleaning extraction artifacts, detecting document types (slides vs prose),
and chunking content with overlap for complete coverage during concept extraction.
Supports PDF and DOCX formats.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pypdf
import docx
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.table import Table


@dataclass
class DocumentContent:
    """Standardized representation of an extracted document."""
    filename: str
    blocks: list[str]                  # Cleaned text blocks (non-empty pages/paragraphs) in document order
    block_count: int                   # Total blocks found (including empty ones)
    empty_block_indices: list[int]     # 1-based indices of empty blocks/pages
    doc_kind: str                      # 'slides', 'prose', 'code', or 'unknown'

    # Maps each entry in `blocks` back to its ORIGINAL 1-based page/paragraph
    # number in the source file.
    #
    # WHY THIS EXISTS: empty pages are dropped from `blocks`, so position i in
    # `blocks` is NOT page i+1 whenever an earlier page was empty. Without this
    # map, a citation like "DS-UNIT-1.pdf [p.36-47]" points at the wrong pages
    # (off by the number of empty pages before it), which silently breaks the
    # traceability a professor relies on to check a generated question.
    block_page_map: list[int] = field(default_factory=list)


@dataclass
class Chunk:
    """A bounded segment of document text with block-level traceability."""
    text: str
    block_start: int                   # 1-based starting block/page index
    block_end: int                     # 1-based ending block/page index
    word_count: int


def clean_text(text: str) -> str:
    """
    Cleans raw document text by fixing common extraction artifacts:
    - Stripping non-printable control characters
    - Rejoining hyphenated words split across line breaks
    - Collapsing repeated horizontal whitespace
    - Normalizing excessive newlines

    Args:
        text: Raw text string.

    Returns:
        Cleaned, normalized text string.
    """
    if not text:
        return ""

    # 1. Remove non-printable control characters (preserve standard \n, \t, \r)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 2. Rejoin words split by a hyphen at the end of a line (e.g. "exam-\nple" -> "example")
    cleaned = re.sub(r"(\b\w+)-\s*\n\s*(\w+\b)", r"\1\2", cleaned)

    # 3. Collapse multiple horizontal spaces and tabs into a single space
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # 4. Clean trailing/leading spaces on each line
    lines = [line.strip() for line in cleaned.splitlines()]

    # 5. Collapse 3 or more consecutive blank lines into 2
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


# Markers that reliably indicate source code rather than prose.
#
# Deliberately EXCLUDES a bare ";" and a bare "return": ordinary academic prose
# uses semicolons freely and discusses "return values", so including them
# misclassified normal textbook chapters as code and fed them the wrong
# extraction prompt. Every marker below is rare in prose.
CODE_MARKER_REGEX = re.compile(
    r"#include"
    r"|#define"
    r"|[{}]"                                               # braces
    r"|\b(?:int|void|char|float|double|struct)\s+\w+\s*[(\[;]"  # typed declarations
    r"|\b(?:printf|scanf|malloc|calloc|realloc|free)\s*\("  # C stdlib calls
    r"|;\s*$",                                             # statement-terminating semicolon
    re.MULTILINE,
)


def detect_doc_kind(blocks: list[str]) -> str:
    """
    Detects whether a document is source code, a slide deck, or prose.

    Checks:
    - 'code': if >= 25% of non-empty blocks contain C/programming syntax markers
              (#include, #define, {, }, ;, int/void/char/float/double... (, printf, return)
    - 'slides': if average words per block < 60
    - 'prose': if average words per block >= 60
    - 'unknown': if blocks is empty

    Args:
        blocks: List of non-empty text blocks.

    Returns:
        'code', 'slides', 'prose', or 'unknown'.
    """
    if not blocks:
        return "unknown"

    code_blocks = sum(1 for b in blocks if CODE_MARKER_REGEX.search(b))
    if (code_blocks / len(blocks)) >= 0.25:
        return "code"

    total_words = sum(len(b.split()) for b in blocks)
    avg_words = total_words / len(blocks)

    if avg_words < 60:
        return "slides"
    return "prose"


def _extract_pdf(path: Path) -> DocumentContent:
    """Extracts and cleans text blocks from a PDF file using pypdf."""
    reader = pypdf.PdfReader(str(path))
    block_count = len(reader.pages)
    blocks: list[str] = []
    empty_block_indices: list[int] = []
    block_page_map: list[int] = []

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)
        if not cleaned:
            empty_block_indices.append(page_num)
        else:
            blocks.append(cleaned)
            # Remember the real page number this block came from, so citations
            # point at the page a professor would actually turn to.
            block_page_map.append(page_num)

    kind = detect_doc_kind(blocks)

    return DocumentContent(
        filename=path.name,
        blocks=blocks,
        block_count=block_count,
        empty_block_indices=empty_block_indices,
        doc_kind=kind,
        block_page_map=block_page_map,
    )


def _extract_docx(path: Path) -> DocumentContent:
    """
    Extracts and cleans text blocks from a DOCX file using python-docx.
    Iterates paragraphs and tables in document order.
    Table rows are formatted by joining cells with ' | '.
    Empty blocks are tracked in empty_block_indices.
    """
    doc = docx.Document(str(path))
    blocks: list[str] = []
    empty_block_indices: list[int] = []
    block_page_map: list[int] = []
    raw_block_counter = 0

    for child in doc.element.body:
        if isinstance(child, CT_P):
            p = Paragraph(child, doc)
            raw_block_counter += 1
            cleaned = clean_text(p.text)
            if not cleaned:
                empty_block_indices.append(raw_block_counter)
            else:
                blocks.append(cleaned)
                block_page_map.append(raw_block_counter)
        elif isinstance(child, CT_Tbl):
            t = Table(child, doc)
            raw_block_counter += 1
            table_lines: list[str] = []
            for row in t.rows:
                row_cells = [cell.text.replace("\n", " ").strip() for cell in row.cells]
                # Join cells with " | "
                row_str = " | ".join(row_cells)
                if row_str.strip():
                    table_lines.append(row_str)
            table_text = "\n".join(table_lines)
            cleaned = clean_text(table_text)
            if not cleaned:
                empty_block_indices.append(raw_block_counter)
            else:
                blocks.append(cleaned)
                block_page_map.append(raw_block_counter)

    kind = detect_doc_kind(blocks)

    return DocumentContent(
        filename=path.name,
        blocks=blocks,
        block_count=raw_block_counter,
        empty_block_indices=empty_block_indices,
        doc_kind=kind,
        block_page_map=block_page_map,
    )


def extract_document(path: str) -> DocumentContent:
    """
    Single entry point for document extraction.
    Hides file formats behind a uniform DocumentContent return type.

    Args:
        path: Path to the target file.

    Returns:
        DocumentContent with cleaned blocks, empty page metadata, and doc_kind.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: '{path}'")

    suffix = path_obj.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path_obj)
    elif suffix == ".docx":
        return _extract_docx(path_obj)
    elif suffix in (".pptx", ".txt", ".md"):
        raise NotImplementedError(
            f"Format '{suffix}' is planned for future releases, but is not yet supported."
        )
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported: .pdf, .docx")


def build_chunks(
    blocks: list[str],
    target_words: int = 1200,
    overlap_words: int = 150,
    page_map: Optional[list[int]] = None,
) -> list[Chunk]:
    """
    Groups document blocks into sequential, overlapping Chunks without cutting blocks.

    IMPORTANT: Chunking operates on the blocks of ONE document only. Never chunk
    across documents — a chunk spanning the end of one file and the start of another
    produces nonsense concepts.

    Args:
        blocks: Cleaned text blocks belonging to a single document.
        target_words: Target word limit before closing a chunk.
        overlap_words: Approximate word count to carry over into the next chunk.
        page_map: Optional DocumentContent.block_page_map. When supplied, the
            returned block_start/block_end are translated into ORIGINAL page
            numbers instead of positions in the filtered `blocks` list. Pass this
            whenever the result will be shown to a human or stored as a citation,
            otherwise the numbers are wrong for any document containing empty pages.

    Returns:
        List of Chunk objects covering the whole document.
    """
    if not blocks:
        return []

    def to_page(block_index: int) -> int:
        """Translate a 1-based index into `blocks` to its original page number."""
        if not page_map:
            return block_index
        # Clamp defensively so a short/missing map can never raise.
        safe = min(max(block_index, 1), len(page_map))
        return page_map[safe - 1]

    # Never carry more than half a chunk forward as overlap. Without this cap a
    # single block larger than target_words would be re-sent in every subsequent
    # chunk, multiplying token cost on dense documents.
    max_overlap_words = max(overlap_words, target_words // 2)

    chunks: list[Chunk] = []
    current_blocks: list[str] = []
    current_word_count = 0
    chunk_start_idx = 1  # 1-based block index

    for i, block in enumerate(blocks, start=1):
        block_words = len(block.split())

        # If adding this block exceeds target_words and we already have blocks, close chunk
        if current_blocks and (current_word_count + block_words > target_words):
            chunk_text = "\n\n".join(current_blocks)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    block_start=to_page(chunk_start_idx),
                    block_end=to_page(i - 1),
                    word_count=current_word_count,
                )
            )

            # Build overlap from the trailing end of current_blocks
            overlap_blocks: list[str] = []
            overlap_count = 0
            # Walk backwards through current_blocks
            for b in reversed(current_blocks):
                b_words = len(b.split())
                # Stop before the overlap grows past half a chunk — otherwise an
                # oversized block gets dragged into every later chunk. If even the
                # first trailing block is too large, carry no overlap at all; a
                # block that big cannot straddle a boundary anyway.
                if (overlap_count + b_words) > max_overlap_words:
                    break
                overlap_blocks.insert(0, b)
                overlap_count += b_words
                if overlap_count >= overlap_words:
                    break

            # Calculate the starting block index for the next chunk
            overlap_block_count = len(overlap_blocks)
            chunk_start_idx = i - overlap_block_count
            current_blocks = list(overlap_blocks)
            current_word_count = overlap_count

        current_blocks.append(block)
        current_word_count += block_words

    # Append remaining blocks as the final chunk
    if current_blocks:
        chunk_text = "\n\n".join(current_blocks)
        chunks.append(
            Chunk(
                text=chunk_text,
                block_start=to_page(chunk_start_idx),
                block_end=to_page(len(blocks)),
                word_count=current_word_count,
            )
        )

    return chunks
