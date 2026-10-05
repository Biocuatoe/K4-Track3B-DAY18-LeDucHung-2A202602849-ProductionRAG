from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5.

Thứ tự chạy (đúng theo đề bài):
    load documents (M1) → hierarchical chunking (M1) → enrichment (M5)
    → BM25 + Dense indexing (M2) → hybrid retrieval (M2) → reranking (M3)
    → grounded answer generation (Groq) → RAGAS evaluation (M4)
    → failure analysis (M4) → report generation

Nguyên tắc quan trọng (Lab 18 §12.1):
    * Retrieval representation = enriched_text (giàu ngữ nghĩa, bridge vocabulary).
    * Answer context            = original source text (sạch, giảm noise, tăng
      faithfulness). Ta luôn giữ lại original_text trong metadata để phục hồi.
"""

import math
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMBEDDING_MODEL, HYBRID_TOP_K, RERANK_TOP_K
from src.llm_client import chat, provider_info
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, failure_analysis, load_test_set, save_report
from src.m5_enrichment import enrich_chunks

# ── Latency instrumentation ──────────────────────────────


class LatencyTracker:
    """Thu thập thời gian thực của từng bước (không fabricate)."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    def time(self, name: str):
        tracker = self

        class _Timer:
            def __enter__(self_inner):
                self_inner._t0 = time.perf_counter()
                return self_inner

            def __exit__(self_inner, *exc):
                elapsed = (time.perf_counter() - self_inner._t0) * 1000.0
                # Cộng dồn nếu một bước chạy lại nhiều lần (ví dụ answer generation).
                tracker.stages[name] = tracker.stages.get(name, 0.0) + elapsed
                return False

        return _Timer()

    def as_dict(self) -> dict:
        return {k: round(v, 2) for k, v in self.stages.items()}


# ── Grounded answer generation (Groq) ─────────────────────

GROUNDED_SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu nội bộ của công ty.

NGUYÊN TẮC BẮT BUỘC:
1. Trả lời bằng TIẾNG VIỆT.
2. CHỈ được dùng thông tin trong phần CONTEXT được cung cấp. Tuyệt đối không dùng
   kiến thức bên ngoài và không bịa thêm sự kiện, con số hay quy định.
3. Nếu CONTEXT không đủ thông tin để trả lời, phải nói rõ ràng:
   "Không tìm thấy thông tin trong tài liệu đã cung cấp." — KHÔNG được đoán.
4. PHÂN BIỆT CHÍNH SÁCH CŨ VÀ CHÍNH SÁCH HIỆN HÀNH:
   - Tài liệu có thể chứa nhiều phiên bản (v1.0/v2.0, 2023/2024).
   - Khi có nhiều phiên bản cùng nói về một quy định, hãy ưu tiên phiên bản
     HIỆN HÀNH (phiên bản mới hơn, ngày hiệu lực muộn hơn, hoặc văn bản ghi rõ
     "thay thế" phiên bản cũ).
   - Nếu cả hai phiên bản cùng xuất hiện trong context, hãy nêu rõ: quy định
     hiện hành là gì, và quy định cũ là gì (đã bị thay thế).
5. GIỮ NGUYÊN CON SỐ và đơn vị đúng như trong tài liệu (ví dụ: 200.000.000 VNĐ,
   15 ngày, 120 ngày, 2%/tháng). Không làm tròn, không quy đổi, không ước lượng.
6. CÂU HỎI PHỦ ĐỊNH — trả lời đúng nghĩa "KHÔNG":
   - Câu hỏi dạng "Có phải...?", "... có được ... không?", "... có nên ... không?"
     cần trả lời rõ là CÓ hay KHÔNG.
   - Nếu tài liệu quy định nhân viên KHÔNG được/KHÔNG phải thì phải nói "KHÔNG",
     tuyệt đối không đảo ngược nghĩa (negation).
7. CÂU HỎI NHIỀU BƯỚC (multi-hop) và TÍNH TOÁN:
   - Nếu cần kết hợp nhiều đoạn context hoặc cần tính toán, hãy thực hiện từng
     bước và nêu rõ phép tính (ví dụ: "9 năm ÷ 3 năm/lần = 3 lần, cộng thêm 3 ngày,
     tổng 15 + 3 = 18 ngày").
8. Trích dẫn nguồn ở cuối câu trả lời dạng (nguồn: tên_file) khi thông tin đến từ
   một tài liệu cụ thể.

ĐỊNH DẠNG ĐẦU RA: trả lời ngắn gọn, đúng trọng tâm, có thể gồm 1-3 đoạn ngắn hoặc
danh sách gạch đầu dòng. Không thêm lời mở bài hoặc lời kết thừa."""


def generate_answer(query: str, contexts: list[dict]) -> str:
    """Sinh câu trả lời có grounding từ Groq. Fallback về context khi không có key."""
    if not contexts:
        return "Không tìm thấy thông tin trong tài liệu đã cung cấp."

    context_block = "\n\n".join(
        f"[{i + 1}] (nguồn: {c.get('source', 'unknown')})\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    user = (
        f"CONTEXT:\n{context_block}\n\n"
        f"CÂU HỎI: {query}\n\n"
        "Trả lời dựa trên CONTEXT ở trên."
    )

    answer = chat(GROUNDED_SYSTEM_PROMPT, user, max_tokens=700, temperature=0.0)
    if answer:
        return answer
    # Không có API key / call lỗi → trả về passage tốt nhất, KHÔNG bịa nội dung.
    return (
        "Không có GROQ_API_KEY nên không thể sinh câu trả lời bằng LLM. "
        f"Đoạn tài liệu liên quan nhất: {contexts[0]['text']}"
    )


# ── Pipeline build ────────────────────────────────────────


def build_pipeline(latency: LatencyTracker | None = None):
    """Build production RAG pipeline. Returns (search, reranker, parents_by_id)."""
    latency = latency or LatencyTracker()
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    with latency.time("document_loading"):
        print("\n[1/6] Loading documents...", flush=True)
        docs = load_documents()
    print(f"  ✓ {len(docs)} documents ({latency.stages['document_loading']:.0f} ms)", flush=True)

    with latency.time("chunking"):
        all_chunks = []
        parents_by_id: dict[str, str] = {}
        for doc in docs:
            parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
            for p in parents:
                parents_by_id[p.metadata.get("parent_id")] = p.text
            for child in children:
                all_chunks.append({
                    "text": child.text,
                    "metadata": {**child.metadata, "parent_id": child.parent_id},
                })
    print(
        f"  ✓ {len(all_chunks)} child chunks from {len(docs)} documents "
        f"({latency.stages['chunking']:.0f} ms)",
        flush=True,
    )

    # Step 2: Enrichment (M5) — combined mode, 1 Groq call/chunk
    with latency.time("enrichment"):
        print(f"\n[2/6] Enriching {len(all_chunks)} chunks (M5, combined mode)...", flush=True)
        enriched = enrich_chunks(all_chunks)
        if enriched:
            # Retrieval dùng enriched_text; GIỮ original_text để answer dùng source sạch.
            all_chunks = [
                {
                    "text": e.enriched_text,
                    "metadata": {**e.auto_metadata, "original_text": e.original_text},
                }
                for e in enriched
            ]
    print(
        f"  ✓ Enriched {len(all_chunks)} chunks ({latency.stages['enrichment']:.0f} ms)",
        flush=True,
    )

    # Step 3: Index (M2) — BM25 + Dense
    with latency.time("indexing"):
        print(f"\n[3/6] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
        search = HybridSearch()
        search.index(all_chunks)
    print(f"  ✓ Indexed ({latency.stages['indexing']:.0f} ms)", flush=True)

    # Step 4: Reranker (M3)
    with latency.time("reranker_loading"):
        print("\n[4/6] Loading cross-encoder reranker...", flush=True)
        reranker = CrossEncoderReranker()
        # Warm-up: model load không nên nằm trong latency của từng query.
        try:
            reranker.rerank("warmup", [{"text": "warmup", "score": 0.0, "metadata": {}}], top_k=1)
        except Exception as exc:  # pragma: no cover
            print(f"  ⚠️  Reranker warmup failed: {exc}", flush=True)
    print(
        f"  ✓ Reranker ready ({latency.stages['reranker_loading']:.0f} ms)"
        f"{' [FALLBACK]' if getattr(reranker, 'using_fallback', False) else ''}",
        flush=True,
    )

    return search, reranker, parents_by_id


def run_query(
    query: str,
    search: HybridSearch,
    reranker: CrossEncoderReranker,
    parents_by_id: dict[str, str] | None = None,
    latency: LatencyTracker | None = None,
) -> tuple[str, list[str]]:
    """Chạy 1 query qua toàn pipeline. Trả về (answer, contexts).

    Contexts trả về là SOURCE TEXT sạch (không phải enriched text) để RAGAS
    đánh giá faithfulness trên bằng chứng thực sự.
    """
    latency = latency or LatencyTracker()

    with latency.time("retrieval"):
        results = search.search(query, top_k=HYBRID_TOP_K)

    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    with latency.time("reranking"):
        reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)

    selected = reranked or results[:RERANK_TOP_K]

    # Phục hồi source text sạch từ metadata, fallback sang parent chunk.
    contexts: list[dict] = []
    seen: set[str] = set()
    for item in selected:
        if isinstance(item, tuple):  # RerankResult-like
            text, meta = item[0], item[3]
        else:  # SearchResult
            text, meta = item.text, item.metadata
        meta = meta or {}
        clean = meta.get("original_text") or text
        if not clean or clean in seen:
            continue
        seen.add(clean)
        contexts.append({
            "text": clean,
            "source": meta.get("source", "unknown"),
            "rerank_score": getattr(item, "rerank_score", item.score if hasattr(item, "score") else None),
        })

    with latency.time("answer_generation"):
        answer = generate_answer(query, contexts)

    return answer, [c["text"] for c in contexts]


def evaluate_pipeline(
    search: HybridSearch,
    reranker: CrossEncoderReranker,
    parents_by_id: dict[str, str] | None = None,
    latency: LatencyTracker | None = None,
) -> dict:
    """Chạy evaluation trên test set + RAGAS + failure analysis + report."""
    latency = latency or LatencyTracker()
    test_set = load_test_set()
    print(f"\n[5/6] Running {len(test_set)} queries...", flush=True)

    questions, answers, all_contexts, ground_truths = [], [], [], []
    for i, item in enumerate(test_set):
        answer, contexts = run_query(
            item["question"], search, reranker, parents_by_id, latency
        )
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i + 1}/{len(test_set)}] {item['question'][:60]}", flush=True)

    with latency.time("ragas_evaluation"):
        print(f"\n[6/6] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
        results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    status = results.get("status", "unknown")
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        raw = results.get(m)
        # None/NaN = judge unavailable (rate-limit/API error), NOT a real 0.0.
        # Do not coerce — printing 0.0000 here would resurrect the exact
        # false-confidence bug the NaN sentinel exists to prevent.
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            print(f"  – {m}: n/a (judge unavailable)")
            continue
        s = float(raw)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")
    print(f"  status: {status}")
    if status == "degraded":
        cov = results.get("metric_coverage", {})
        bad = ", ".join(
            f"{k} {v['n_measured']}/{v['n_total']}"
            for k, v in cov.items() if v.get("coverage", 1.0) < 1.0
        )
        print(f"  ⚠️  DEGRADED — metrics not fully measured: {bad}")
        print("      Scores below cover only the measured subset; do NOT read as a verdict.")
    if results.get("error"):
        print(f"  ⚠️  {results['error']}")

    failures = failure_analysis(results.get("per_question", []), bottom_n=5)

    # Ghi report với latency + thông tin provider/model.
    info = provider_info()
    results.setdefault("status", "unknown")
    save_report(
        results,
        failures,
        latency_breakdown_ms=latency.as_dict(),
        provider_info={
            **info,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": reranker.model_name,
        },
    )
    return results


if __name__ == "__main__":
    start = time.time()
    tracker = LatencyTracker()
    search, reranker, parents_by_id = build_pipeline(tracker)
    results = evaluate_pipeline(search, reranker, parents_by_id, tracker)
    total_ms = (time.time() - start) * 1000
    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN (ms)")
    print("=" * 60)
    for k, v in sorted(tracker.as_dict().items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v:>10.1f}")
    print(f"  {'total_pipeline':<22} {total_ms:>10.1f}")
    print(f"\nTotal: {total_ms / 1000:.1f}s")
