"""Tests for Module 3: Reranking."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m3_rerank import CrossEncoderReranker, RerankResult, benchmark_reranker

Q = "Nhân viên được nghỉ phép bao nhiêu ngày?"
DOCS = [
    {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
    {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
    {"text": "VPN dùng WireGuard AES-256.", "score": 0.6, "metadata": {}},
]

def test_rerank_returns():
    r = CrossEncoderReranker().rerank(Q, DOCS, top_k=2)
    assert len(r) > 0 and len(r) <= 2

def test_rerank_type():
    assert all(isinstance(x, RerankResult) for x in CrossEncoderReranker().rerank(Q, DOCS))

def test_rerank_sorted():
    r = CrossEncoderReranker().rerank(Q, DOCS)
    if len(r) >= 2:
        assert r[0].rerank_score >= r[1].rerank_score

def test_rerank_relevant_first():
    r = CrossEncoderReranker().rerank(Q, DOCS)
    if r:
        assert "nghỉ" in r[0].text.lower() or "12" in r[0].text

def test_benchmark_stats():
    stats = benchmark_reranker(CrossEncoderReranker(), Q, DOCS, n_runs=2)
    assert "avg_ms" in stats and "min_ms" in stats and "max_ms" in stats
