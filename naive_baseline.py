"""
Basic RAG Baseline — Chạy TRƯỚC để có scores so sánh.
=====================================================
Basic = paragraph chunking + dense-only search (không hybrid, không rerank, không enrichment).
Đây là RAG đã học ở buổi trước — hôm nay production pipeline sẽ cải thiến từng bước.

Cố ý GIỮ NGUYÊN mức độ "naive" để phép so sánh là công bằng:
  - chunking: paragraph (chunk_basic) — không phân tích ngữ nghĩa/cấu trúc
  - search:   dense-only — không BM25, không RRF
  - answer:   prompt 1 câu đơn giản (không grounding mạnh, không xử lý phiên bản tài liệu)
  - KHÔNG rerank, KHÔNG enrichment
"""

import math
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EMBEDDING_MODEL, NAIVE_COLLECTION
from src.llm_client import chat, groq_available, provider_info
from src.m1_chunking import chunk_basic, load_documents
from src.m2_search import DenseSearch
from src.m4_eval import evaluate_ragas, load_test_set, save_report

# Prompt naive — cố tình đơn giản, đây là baseline chứ không phải production.
NAIVE_SYSTEM_PROMPT = "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"


def main():
    print("=" * 60)
    print("BASIC RAG BASELINE")
    print("(paragraph chunking + dense-only, no rerank, no enrichment)")
    print("=" * 60)

    t0 = time.time()
    docs = load_documents()
    chunks = []
    for doc in docs:
        for c in chunk_basic(doc["text"], metadata=doc["metadata"]):
            chunks.append({"text": c.text, "metadata": c.metadata})
    print(f"  {len(chunks)} basic paragraph chunks")

    t_index = time.time()
    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)
    index_ms = (time.time() - t_index) * 1000

    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []
    has_key = groq_available()

    for i, item in enumerate(test_set):
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        contexts = [r.text for r in results]

        answer = None
        if has_key and contexts:
            answer = chat(
                NAIVE_SYSTEM_PROMPT,
                "Context:\n" + "\n\n".join(contexts) + f"\n\nCâu hỏi: {item['question']}",
                max_tokens=400,
            )
        if not answer:
            # Không có key hoặc call lỗi → baseline rơi về trích nguyên văn passage đầu.
            answer = contexts[0] if contexts else "Không tìm thấy."

        answers.append(answer)
        questions.append(item["question"])
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i + 1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t_eval = time.time()
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    eval_ms = (time.time() - t_eval) * 1000

    print("\nBASIC BASELINE SCORES")
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        raw = results.get(m)
        # None/NaN = judge unavailable (rate-limit/API error), NOT a real 0.0.
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            print(f"  {m}: n/a (judge unavailable)")
            continue
        print(f"  {m}: {float(raw):.4f}")
    print(f"  status: {results.get('status', 'unknown')}")
    if results.get("status") == "degraded":
        cov = results.get("metric_coverage", {})
        bad = ", ".join(
            f"{k} {v['n_measured']}/{v['n_total']}"
            for k, v in cov.items() if v.get("coverage", 1.0) < 1.0
        )
        print(f"  ⚠️  DEGRADED — not fully measured: {bad}")

    save_report(
        results,
        [],
        path="reports/naive_baseline_report.json",
        latency_breakdown_ms={
            "document_loading_and_chunking": (t_index - t0) * 1000,
            "indexing": index_ms,
            "retrieval_and_answer_generation": max(
                (t_eval - t_index - eval_ms) * 1000, 0.0
            ),
            "ragas_evaluation": eval_ms,
        },
        provider_info={
            **provider_info(),
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": None,
        },
    )
    print("\nDone! Now implement advanced modules and run: python main.py")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total: {time.time() - start:.1f}s")
