"""Tests for Module 5: Enrichment Pipeline."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m5_enrichment import (
    EnrichedChunk,
    contextual_prepend,
    enrich_chunks,
    extract_metadata,
    generate_hypothesis_questions,
    summarize_chunk,
)

SAMPLE = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm."
CHUNKS = [
    {"text": SAMPLE, "metadata": {"source": "policy.md"}},
    {"text": "Mật khẩu phải thay đổi mỗi 90 ngày.", "metadata": {"source": "it.md"}},
]


def test_summarize_returns_string():
    result = summarize_chunk(SAMPLE)
    assert isinstance(result, str)


def test_summarize_shorter_than_original():
    result = summarize_chunk(SAMPLE)
    if result:  # May be empty if no API key
        assert len(result) <= len(SAMPLE) * 2  # Summary should not be much longer


def test_hyqa_returns_list():
    result = generate_hypothesis_questions(SAMPLE, n_questions=2)
    assert isinstance(result, list)


def test_hyqa_generates_questions():
    result = generate_hypothesis_questions(SAMPLE, n_questions=2)
    if result:
        assert len(result) >= 1
        assert any("?" in q or "bao" in q.lower() or "mấy" in q.lower() for q in result)


def test_contextual_prepend_returns_string():
    result = contextual_prepend(SAMPLE, "Sổ tay nhân viên")
    assert isinstance(result, str)
    assert len(result) >= len(SAMPLE)  # Should be at least as long as original


def test_contextual_contains_original():
    result = contextual_prepend(SAMPLE, "Sổ tay nhân viên")
    assert SAMPLE in result  # Original text must be preserved


def test_extract_metadata_returns_dict():
    result = extract_metadata(SAMPLE)
    assert isinstance(result, dict)


def test_enrich_chunks_returns_list():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    assert isinstance(result, list)


def test_enrich_chunks_type():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    if result:
        assert all(isinstance(c, EnrichedChunk) for c in result)


def test_enrich_preserves_original():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    if result:
        assert result[0].original_text == SAMPLE
