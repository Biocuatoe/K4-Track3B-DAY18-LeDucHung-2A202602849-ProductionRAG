from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    Loads lazily on first call to _load_model() and caches the model instance
    on self._model so that subsequent rerank() calls reuse it without
    re-downloading or re-instantiating.

    If the real model cannot be loaded (e.g. weights missing or torch unavailable),
    the class falls back to a deterministic heuristic that returns
    documents sorted by their original score with a flag set so callers can
    detect that fallback mode is active.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self.using_fallback = False  # set True if real model failed to load
        self._load_error: str | None = None  # stores exact error message if fallback used

    def _load_model(self):
        """Lazily load the CrossEncoder. Caches on self._model after first call.

        Falls back to None on error and sets self.using_fallback so callers can
        detect that the real model was unavailable.
        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
            # Use device='cpu' explicitly. max_length is left at default
            # so that sentence-transformers auto-selects an appropriate value
            # for the model's config (8194 for xlm-roberta).
            self._model = CrossEncoder(
                self.model_name,
                device="cpu",
            )
            return self._model
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            print(
                f"  ⚠️  CrossEncoderReranker: failed to load '{self.model_name}' "
                f"({self._load_error}). "
                "Falling back to deterministic score heuristic.",
                flush=True,
            )
            self.using_fallback = True
            self._model = None
            return None

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        """Rerank documents using the cross-encoder.

        Parameters
        ----------
        query : str
            The search query.
        documents : list[dict]
            Each dict must have a ``"text"`` key; may optionally have
            ``"score"`` (original retrieval score) and ``"metadata"`` keys.
        top_k : int, optional
            Maximum number of results to return. Defaults to RERANK_TOP_K.

        Returns
        -------
        list[RerankResult]
            Up to ``top_k`` results sorted by cross-encoder score descending.
            Falls back to sorting by original score if the real model is
            unavailable; callers can check ``reranker.using_fallback`` to detect
            this case.
        """
        if not documents:
            return []

        model = self._load_model()

        if model is not None:
            # ── Real cross-encoder path ─────────────────────────────────────
            pairs = [(query, doc["text"]) for doc in documents]
            raw_scores = model.predict(pairs)

            # model.predict may return a scalar or a 1-D array depending on
            # the input shape; normalise to a list.
            if isinstance(raw_scores, (int, float)):
                scores = [float(raw_scores)]
            else:
                scores = [float(s) for s in raw_scores]
        else:
            # ── Deterministic fallback ───────────────────────────────────────
            # Preserve the same interface but signal via using_fallback.
            scores = [doc.get("score", 0.0) for doc in documents]

        # Stable sort: sort by rerank score descending, original input order
        # preserved for equal scores. strict=True guards against a model that
        # returns fewer scores than documents (would silently drop docs).
        scored = sorted(
            enumerate(zip(scores, documents, strict=True)),
            key=lambda item: (item[1][0], -item[0]),
            reverse=True,
        )

        results = []
        for rank, (_, (score, doc)) in enumerate(scored[:top_k]):
            results.append(
                RerankResult(
                    text=doc["text"],
                    original_score=doc.get("score", 0.0),
                    rerank_score=float(score),
                    metadata=doc.get("metadata") or {},
                    rank=rank,
                )
            )
        return results


class FlashrankReranker:
    """Lightweight alternative reranker (<5ms) using FlashRank.

    FlashRank downloads its model on first instantiation. If no cached model
    is found and download fails, rerank() returns an empty list.
    """

    def __init__(self):
        self._model = None
        self._warned = False

    def _get_ranker(self):
        """Lazily instantiate and cache the FlashRank Ranker."""
        if self._model is not None:
            return self._model

        try:
            from flashrank import Ranker
            self._model = Ranker()
            return self._model
        except Exception as exc:
            if not self._warned:
                print(
                    f"  ⚠️  FlashrankReranker: failed to initialise "
                    f"(flashrank.Ranker error: {type(exc).__name__}: {exc}). "
                    "FlashRank requires an internet connection to download its "
                    "model on first run. Returning empty results.",
                    flush=True,
                )
                self._warned = True
            return None

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        """Rerank documents using FlashRank.

        Returns an empty list if the FlashRank model cannot be loaded.
        """
        if not documents:
            return []

        ranker = self._get_ranker()
        if ranker is None:
            return []

        passages = [{"text": doc["text"]} for doc in documents]

        try:
            from flashrank import RerankRequest
            request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(request)
        except Exception as exc:
            if not self._warned:
                print(
                    f"  ⚠️  FlashrankReranker.rerank() failed: "
                    f"{type(exc).__name__}: {exc}. "
                    "Returning empty results.",
                    flush=True,
                )
                self._warned = True
            return []

        # Build a lookup from text → original doc metadata/score.
        orig_by_text = {doc["text"]: doc for doc in documents}

        reranked = []
        for rank, item in enumerate(results[:top_k]):
            text = item.get("text", "")
            orig = orig_by_text.get(text, {})
            reranked.append(
                RerankResult(
                    text=text,
                    original_score=orig.get("score", 0.0),
                    rerank_score=float(item.get("score", 0.0)),
                    metadata=orig.get("metadata") or {},
                    rank=rank,
                )
            )
        return reranked


def benchmark_reranker(
    reranker,
    query: str,
    documents: list[dict],
    n_runs: int = 5,
) -> dict:
    """Benchmark reranking latency over n_runs.

    Warm-up: one warm-up call is made OUTSIDE the measured loop so that the
    first real timing is not polluted by model-loading I/O on a lazy loader.
    The warm-up is not subtracted from any reported timing — it is purely to
    ensure the model is resident when the measurement loop begins.

    Parameters
    ----------
    reranker : CrossEncoderReranker or FlashrankReranker
        The reranker instance to benchmark.
    query : str
        The search query.
    documents : list[dict]
        Documents to rerank.
    n_runs : int, default 5
        Number of timing iterations. Must be > 0.

    Returns
    -------
    dict
        Keys: ``avg_ms``, ``min_ms``, ``max_ms``.
        Returns zeros for all keys if ``n_runs <= 0`` or ``documents`` is empty.
    """
    if n_runs <= 0 or not documents:
        return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    # Warm-up: one non-measured call so lazy loaders (CrossEncoder, FlashRank)
    # bring the model into memory before the timing loop.
    reranker.rerank(query, documents)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]

    print("── CrossEncoderReranker ──", flush=True)
    reranker = CrossEncoderReranker()
    if reranker.using_fallback:
        print("NOTE: running in fallback mode (real model unavailable)", flush=True)
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] rerank={r.rerank_score:.4f}  original={r.original_score}  {r.text}", flush=True)

    stats_3 = benchmark_reranker(reranker, query, docs, n_runs=5)
    print(
        f"Benchmark (3 docs, 5 runs): "
        f"avg={stats_3['avg_ms']:.1f}ms  min={stats_3['min_ms']:.1f}ms  max={stats_3['max_ms']:.1f}ms",
        flush=True,
    )

    # Realistic ~20-doc benchmark
    docs_20 = [
        {"text": f"Policy document {i}: keyword density test text for topic relevance.", "score": 0.8 - i * 0.01, "metadata": {}}
        for i in range(20)
    ]
    docs_20[0] = {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.85, "metadata": {}}
    docs_20[5] = {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}}

    stats_20 = benchmark_reranker(reranker, query, docs_20, n_runs=3)
    print(
        f"Benchmark (20 docs, 3 runs): "
        f"avg={stats_20['avg_ms']:.1f}ms  min={stats_20['min_ms']:.1f}ms  max={stats_20['max_ms']:.1f}ms",
        flush=True,
    )

    print("\n── FlashrankReranker ──", flush=True)
    fl_reranker = FlashrankReranker()
    for r in fl_reranker.rerank(query, docs):
        print(f"[{r.rank}] rerank={r.rerank_score:.4f}  original={r.original_score}  {r.text}", flush=True)
