from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_EMBEDDING_MODEL,
    SEMANTIC_THRESHOLD,
)

# ─── Module-level lazy singleton for the sentence encoder ────────────────────
_encoder = None

# ── Network/hub exception helper ─────────────────────────────────────────────
# Canonical definition lives in src/_network_exceptions.py; re-export here so
# existing internal callers (e.g. tests that patch m1._get_network_exceptions)
# continue to work without modification.
from src._network_exceptions import get_network_exceptions as _get_network_exceptions


def _get_encoder():
    """Lazily load the sentence-transformer encoder (singleton)."""
    global _encoder
    if _encoder is None:
        _net_excs = _get_network_exceptions()
        try:
            from sentence_transformers import SentenceTransformer

            _encoder = SentenceTransformer(SEMANTIC_EMBEDDING_MODEL)
        except _net_excs:
            # Network or hub may be unavailable; rely on cached model
            from sentence_transformers import SentenceTransformer

            _encoder = SentenceTransformer(
                SEMANTIC_EMBEDDING_MODEL, local_files_only=True
            )
    return _encoder


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold for splitting. New chunk starts
                   when similarity between consecutive sentences drops below this.
        metadata: Additional metadata to attach to each chunk.

    Returns:
        list[Chunk] with strategy="semantic" in metadata.
    """
    metadata = metadata or {}

    # ── 1. Sentence segmentation ──────────────────────────────────────────
    # Split on blank lines (paragraph boundaries) first, then on sentence
    # boundaries within each paragraph.  This avoids blindly splitting on
    # every newline (which would break markdown lists / code blocks).
    paragraphs = re.split(r"\n\s*\n", text)
    sentences: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Split on sentence-ending punctuation followed by whitespace
        parts = re.split(r"(?<=[.!?])\s+", para)
        for s in parts:
            s = s.strip()
            if s:
                sentences.append(s)

    # Handle edge cases
    if not sentences:
        return []

    # Single sentence: return as-is
    if len(sentences) == 1:
        return [Chunk(
            text=sentences[0],
            metadata={
                **metadata,
                "strategy": "semantic",
                "chunk_index": 0,
                "sentence_count": 1,
                "group_index": 0,
            },
        )]

    # ── 2. Encode sentences ────────────────────────────────────────────────
    encoder = _get_encoder()
    try:
        embeddings = encoder.encode(sentences, show_progress_bar=False)
    except _get_network_exceptions():  # noqa: BLE001
        # Fallback: sentence-boundary grouping if encoding fails (network or hub)
        return _semantic_fallback(sentences, metadata)

    # ── 3. Cosine similarity between neighbours; group while sim >= threshold ─
    groups: list[list[int]] = []
    groups.append([0])
    for i in range(1, len(sentences)):
        sim = _cosine(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            groups.append([i])
        else:
            groups[-1].append(i)

    # ── 4. Hard-cap: if a group exceeds MAX_CHUNK_CHARS, split it further ───
    MAX_CHUNK_CHARS = 1500
    final_groups: list[list[int]] = []
    for g in groups:
        group_text = " ".join(sentences[j] for j in g)
        if len(group_text) <= MAX_CHUNK_CHARS:
            final_groups.append(g)
        else:
            # Split the oversized group at sentence boundaries
            subgroup: list[int] = []
            subgroup_len = 0
            for idx in g:
                s_len = len(sentences[idx])
                if subgroup and subgroup_len + s_len > MAX_CHUNK_CHARS:
                    final_groups.append(subgroup)
                    subgroup = [idx]
                    subgroup_len = s_len
                else:
                    subgroup.append(idx)
                    subgroup_len += s_len
            if subgroup:
                final_groups.append(subgroup)

    # ── 5. Build Chunk objects ──────────────────────────────────────────────
    chunks: list[Chunk] = []
    for gi, group in enumerate(final_groups):
        chunk_text = " ".join(sentences[j] for j in group)
        chunks.append(Chunk(
            text=chunk_text,
            metadata={
                **metadata,
                "strategy": "semantic",
                "chunk_index": gi,
                "sentence_count": len(group),
                "group_index": gi,
            },
        ))

    return chunks


def _cosine(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = float(_norm(a))
    norm_b = float(_norm(b))
    denom = norm_a * norm_b + 1e-9
    return float(a.dot(b)) / denom


def _norm(v):
    """L2 norm of a vector (works for numpy arrays and array-likes)."""
    return (sum(x * x for x in v)) ** 0.5


def _semantic_fallback(sentences: list[str], metadata: dict) -> list[Chunk]:
    """
    Deterministic fallback when sentence encoding fails.
    Groups sentences by sentence-boundary accumulation (target ~1000 chars),
    still labelled as strategy="semantic".
    """
    TARGET = 1000
    chunks: list[Chunk] = []
    group: list[str] = []
    group_len = 0

    for s in sentences:
        s_len = len(s)
        if group and group_len + s_len > TARGET:
            chunks.append(Chunk(
                text=" ".join(group),
                metadata={
                    **metadata,
                    "strategy": "semantic",
                    "chunk_index": len(chunks),
                    "sentence_count": len(group),
                    "group_index": len(chunks),
                    "_fallback": True,
                },
            ))
            group = [s]
            group_len = s_len
        else:
            group.append(s)
            group_len += s_len

    if group:
        chunks.append(Chunk(
            text=" ".join(group),
            metadata={
                **metadata,
                "strategy": "semantic",
                "chunk_index": len(chunks),
                "sentence_count": len(group),
                "group_index": len(chunks),
                "_fallback": True,
            },
        ))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Maximum character length of each parent chunk.
        child_size: Maximum character length of each child chunk.
        metadata: Additional metadata to attach.

    Returns:
        (parents, children) — each child carries a ``parent_id`` that exists in parents.
    """
    metadata = metadata or {}

    # Empty input
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    # ── 1. Build parent chunks ─────────────────────────────────────────────
    parents: list[Chunk] = []
    current_paras: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len <= parent_size:
            current_paras.append(para)
            current_len += para_len + 2  # account for "\n\n"
        else:
            if current_paras:
                _flush_parent(current_paras, parents, metadata)
                current_paras = []
                current_len = 0
            if para_len > parent_size:
                # Oversized single paragraph — safe-split it into chunks ≤ parent_size
                sub_chunks = _safe_split(para, parent_size)
                for sub in sub_chunks:
                    _flush_parent([sub], parents, metadata)
                current_paras = []
                current_len = 0
            else:
                current_paras.append(para)
                current_len = para_len

    if current_paras:
        _flush_parent(current_paras, parents, metadata)

    # Drop any empty parents that slipped through
    parents = [p for p in parents if p.text.strip()]

    # ── 2. Build child chunks from each parent ────────────────────────────
    children: list[Chunk] = []
    for p_idx, parent in enumerate(parents):
        pid = parent.metadata.get("parent_id", f"parent_{p_idx}")
        sub_chunks = _safe_split(parent.text, child_size)
        for c_idx, sc in enumerate(sub_chunks):
            children.append(Chunk(
                text=sc.strip(),
                metadata={
                    **metadata,
                    "chunk_type": "child",
                    "parent_id": pid,
                    "child_index": c_idx,
                    "chunk_index": len(children),
                },
                parent_id=pid,
            ))

    return (parents, children)


def _flush_parent(paragraphs: list[str], parents: list[Chunk], metadata: dict) -> None:
    """Append a parent chunk built from accumulated paragraphs."""
    pid = f"parent_{len(parents)}"
    text = "\n\n".join(paragraphs)
    parents.append(Chunk(
        text=text,
        metadata={
            **metadata,
            "chunk_type": "parent",
            "parent_id": pid,
            "chunk_index": len(parents),
        },
        parent_id=pid,
    ))


def _safe_split(text: str, max_size: int) -> list[str]:
    """
    Split ``text`` into substrings no longer than ``max_size``.
    Strategy: sentence-aware → paragraph-aware → hard truncate.
    """
    # 1. Try sentence-level split
    sentences: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        parts = re.split(r"(?<=[.!?])\s+", para)
        for s in parts:
            if s.strip():
                sentences.append(s.strip())

    if sentences:
        result: list[str] = []
        current: list[str] = []
        current_len = 0
        for s in sentences:
            s_len = len(s)
            if current and current_len + s_len > max_size:
                result.append(" ".join(current))
                current = [s]
                current_len = s_len
            else:
                current.append(s)
                current_len += s_len
        if current:
            result.append(" ".join(current))

        # If all pieces fit, return
        if all(len(r) <= max_size for r in result):
            return result

    # 2. Paragraph-level split
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paras:
        result = []
        current = []
        current_len = 0
        for p in paras:
            p_len = len(p)
            if current and current_len + p_len > max_size:
                result.append("\n\n".join(current))
                current = [p]
                current_len = p_len
            else:
                current.append(p)
                current_len += p_len
        if current:
            result.append("\n\n".join(current))
        if all(len(r) <= max_size for r in result):
            return result

    # 3. Hard split every max_size chars (last resort)
    return [text[i:i + max_size] for i in range(0, len(text), max_size)]


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse Markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Input Markdown text.
        metadata: Additional metadata to attach.

    Returns:
        list[Chunk] with ``strategy="structure"`` and ``section`` in metadata.
    """
    metadata = metadata or {}

    if not text.strip():
        return []

    # ── 1. Split keeping header lines ─────────────────────────────────────
    # re.split with a capturing group keeps the matched header lines.
    parts = re.split(r"(?m)(^(#{1,6})\s+.+$)", text)
    # parts layout: [preamble, header, content_between, header, content_between, ...]

    chunks: list[Chunk] = []

    # Re-assemble into (header, body_lines) pairs
    # parts[0] = text before first header (or empty)
    # then (header, body) triples follow
    sections: list[tuple[str, str, int]] = []  # (header, body, level)

    # Re-assemble into (header, body, level) triples
    sections: list[tuple[str, str, int]] = []  # (header, body, level)

    current_header = ""
    current_level = 0
    current_body_parts: list[str] = []

    def _flush_section():
        nonlocal current_header, current_body_parts, current_level
        body = "\n".join(current_body_parts).strip()
        if current_header or body:
            sections.append((current_header, body, current_level))
        current_header = ""
        current_level = 0
        current_body_parts = []

    # Walk through parts: alternating (header_line, text_between)
    # parts[0] = preamble (text before first header)
    i = 0
    if parts[i].strip():
        current_body_parts.append(parts[i].strip())
    i += 1

    while i < len(parts):
        hdr = parts[i].strip()       # captured header text, e.g. "## Nghỉ phép năm"
        level = len(hdr) - len(hdr.lstrip("#"))
        content = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2

        # Flush the section accumulated so far
        _flush_section()
        current_header = hdr
        current_level = level
        if content.strip():
            current_body_parts.append(content.strip())

    # Flush the last section (may be empty if doc ended on a header)
    _flush_section()

    # ── 2. Build chunks from sections ────────────────────────────────────
    #
    # Determine the default "section" name for header-less content
    # We look at the passed metadata for a "source" key as fallback.
    source_fallback = metadata.get("source", "preamble")

    # We must track fenced-code state to avoid splitting mid-code-block
    def _is_code_line(line: str) -> bool:
        return line.strip().startswith("```") or line.strip().startswith("|")

    def _safe_boundary(lines: list[str]) -> list[list[str]]:
        """
        Split a list of lines into sub-lists at safe paragraph boundaries,
        never inside a fenced code block or markdown table.
        """
        result: list[list[str]] = []
        current: list[str] = []
        in_code_block = False
        for line in lines:
            stripped = line.strip()
            # Toggle fenced code block state
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                current.append(line)
                continue
            if in_code_block:
                current.append(line)
                continue

            # Table row
            if stripped.startswith("|"):
                current.append(line)
                continue

            # Empty line = paragraph boundary
            if stripped == "":
                if current:
                    result.append(current)
                    current = []
                continue

            current.append(line)

        if current:
            result.append(current)

        return result

    for header, body, level in sections:
        # Determine section name
        section_name = header if header else source_fallback

        if not body.strip():
            # Header only, no body — include the header as the chunk
            chunks.append(Chunk(
                text=header,
                metadata={
                    **metadata,
                    "strategy": "structure",
                    "section": section_name,
                    "level": level,
                    "chunk_index": len(chunks),
                },
            ))
            continue

        lines = body.split("\n")
        # Split at safe boundaries
        sub_blocks = _safe_boundary(lines)

        if len(sub_blocks) <= 1:
            # Single block — whole section is one chunk
            chunk_text = (header + "\n\n" + body).strip() if header else body.strip()
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "strategy": "structure",
                    "section": section_name,
                    "level": level,
                    "chunk_index": len(chunks),
                },
            ))
        else:
            # Multiple blocks — each gets the header prepended
            for block in sub_blocks:
                block_text = "\n".join(block)
                chunk_text = (header + "\n\n" + block_text).strip() if header else block_text.strip()
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        **metadata,
                        "strategy": "structure",
                        "section": section_name,
                        "level": level,
                        "chunk_index": len(chunks),
                    },
                ))

    # Handle preamble-only documents (no headers at all)
    if not chunks:
        chunks.append(Chunk(
            text=text.strip(),
            metadata={
                **metadata,
                "strategy": "structure",
                "section": source_fallback,
                "level": 0,
                "chunk_index": 0,
            },
        ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
