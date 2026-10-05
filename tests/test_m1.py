"""Tests for Module 1: Advanced Chunking."""
import os
import sys
import unittest.mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m1_chunking import (
    Chunk,
    _get_network_exceptions,
    chunk_basic,
    chunk_hierarchical,
    chunk_semantic,
    chunk_structure_aware,
    compare_strategies,
    load_documents,
)

TEXT = """# Nghỉ phép

## Nghỉ phép năm

Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm.
Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên.

## Nghỉ phép không lương

Nhân viên có thể xin nghỉ phép không lương tối đa 30 ngày mỗi năm.
Đơn xin nghỉ phải được Giám đốc bộ phận phê duyệt.

## Nghỉ ốm

Cần nộp giấy xác nhận y tế trong vòng 3 ngày làm việc."""


# --- Baseline (đã implement sẵn) ---

def test_basic_returns_chunks():
    assert len(chunk_basic(TEXT)) > 0

def test_basic_type():
    assert all(isinstance(c, Chunk) for c in chunk_basic(TEXT))


# --- Semantic Chunking ---

def test_semantic_returns_chunks():
    result = chunk_semantic(TEXT, threshold=0.5)
    assert len(result) > 0, "Semantic chunking should return chunks"

def test_semantic_type():
    assert all(isinstance(c, Chunk) for c in chunk_semantic(TEXT, 0.5))

def test_semantic_groups_by_topic():
    """Semantic should produce fewer chunks than basic (groups related sentences)."""
    basic = chunk_basic(TEXT, chunk_size=100)
    semantic = chunk_semantic(TEXT, threshold=0.5)
    assert len(semantic) <= len(basic) + 2  # Allow some tolerance


# --- Hierarchical Chunking ---

def test_hierarchical_returns_both():
    parents, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    assert len(parents) > 0, "Should return parents"
    assert len(children) > 0, "Should return children"

def test_hierarchical_children_have_parent_id():
    _, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    for c in children:
        assert c.parent_id is not None, "Each child must have parent_id"

def test_hierarchical_valid_parent_ids():
    parents, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    parent_ids = {p.metadata.get("parent_id") for p in parents}
    for c in children:
        assert c.parent_id in parent_ids, f"Child parent_id '{c.parent_id}' not in parents"

def test_hierarchical_children_smaller():
    parents, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    avg_p = sum(len(p.text) for p in parents) / max(len(parents), 1)
    avg_c = sum(len(c.text) for c in children) / max(len(children), 1)
    assert avg_c < avg_p, "Children should be smaller than parents"


# --- Structure-Aware Chunking ---

def test_structure_returns_chunks():
    result = chunk_structure_aware(TEXT)
    assert len(result) > 0, "Structure-aware should return chunks"

def test_structure_preserves_headers():
    result = chunk_structure_aware(TEXT)
    texts = " ".join(c.text for c in result)
    assert "Nghỉ phép năm" in texts, "Should preserve section headers"

def test_structure_has_section_metadata():
    result = chunk_structure_aware(TEXT)
    if result:
        assert any("section" in c.metadata for c in result), "Should have section in metadata"


# --- Compare ---

def test_compare_all_strategies():
    docs = load_documents()
    if docs:
        r = compare_strategies(docs)
        for key in ["basic", "semantic", "hierarchical", "structure"]:
            assert key in r, f"Missing strategy: {key}"


# --- Regression: network exception handling ---

def test_network_exceptions_includes_httpx():
    """_get_network_exceptions() must include httpx.HTTPError (not just OSError).

    httpx.HTTPError (and its subclasses like ProxyError) do NOT inherit OSError
    in httpx 0.28.x, so they must be explicitly included in the tuple.
    """
    excs = _get_network_exceptions()
    assert OSError in excs, "OSError must be in network exceptions"
    httpx = pytest.importorskip("httpx")
    assert httpx.HTTPError in excs, "httpx.HTTPError must be in network exceptions"


def test_init_time_httpx_failure_uses_local_retry():
    """_get_encoder() must retry with local_files_only=True when init raises httpx.ProxyError.

    When SentenceTransformer.__init__ raises httpx.ProxyError (e.g. corporate proxy
    blocking model download), the fixed code must NOT propagate the exception — it
    must retry with local_files_only=True.  A reverted implementation that only
    catches OSError would let httpx.ProxyError escape and crash.
    """
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("sentence_transformers")

    import src.m1_chunking as m1

    m1._encoder = None  # reset singleton

    proxy_error = httpx.ProxyError("Corporate proxy blocked download", request=None)

    # Track calls so we can assert the retry happened
    call_args: list[dict] = []

    def fake_init(model_name, **kwargs):
        call_args.append(kwargs)
        if len(call_args) == 1:
            # First call: simulate the proxy blocking the download
            raise proxy_error
        # Second call (retry): return a mock encoder
        mock_enc = unittest.mock.MagicMock()
        mock_enc.encode.return_value = None
        return mock_enc

    with unittest.mock.patch(
        "sentence_transformers.SentenceTransformer",
        side_effect=fake_init,
    ):
        enc = m1._get_encoder()

    # Fixed code: second call must have been made with local_files_only=True
    assert len(call_args) == 2, (
        f"Expected 2 calls to SentenceTransformer (init + retry), got {len(call_args)}. "
        f"The retry path was not taken."
    )
    assert call_args[1].get("local_files_only") is True, (
        f"Expected second call to use local_files_only=True, got: {call_args[1]}"
    )
    assert enc is not None, "Encoder must be returned from retry path"


def test_encode_time_httpx_failure_takes_fallback():
    """chunk_semantic must not propagate httpx.ProxyError raised at encode time.

    If the model loads fine but encode() raises httpx.ProxyError (e.g. network blip
    mid-batch), the sentence-boundary fallback must be returned — the exception must
    NOT escape to the caller.  A reverted implementation that only catches OSError
    in the encode try/except would let httpx.ProxyError propagate.
    """
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("sentence_transformers")

    import src.m1_chunking as m1

    m1._encoder = None  # reset singleton

    # A mock encoder that loads OK but raises httpx.ProxyError on encode
    fake_encoder = unittest.mock.MagicMock()
    fake_encoder.encode.side_effect = httpx.ProxyError(
        "Connection reset during encoding", request=None
    )

    with unittest.mock.patch.object(m1, "_get_encoder", return_value=fake_encoder):
        result = chunk_semantic(TEXT, threshold=0.5)

    # Must return the fallback path (chunks with _fallback=True)
    assert isinstance(result, list), "Must return a list"
    assert len(result) > 0, "Must return at least one chunk from fallback"
    assert all(isinstance(c, m1.Chunk) for c in result), "All items must be Chunk"
    # Fallback chunks must have the _fallback marker in metadata
    assert all(c.metadata.get("_fallback") is True for c in result), (
        "All chunks must be from the sentence-boundary fallback (_fallback=True)"
    )
