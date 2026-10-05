from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import hashlib
import os
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


def _json_safe(value):
    """Convert a value to something JSON-serialisable."""
    try:
        import json
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words using underthesea, then replace '_' with spaces.

    underthesea joins multi-syllable words with '_' (e.g. "nghỉ_phép").
    BM25 tokenises by splitting on spaces → "nghỉ_phép" becomes 1 token
    but query "nghỉ phép" becomes 2 tokens → NO match.
    The replace("_", " ") makes BM25 work correctly.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return ""

    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:  # noqa: BLE001 intentional fallback
        # Fall back to the original text on any tokenizer failure.
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = []
        if not chunks:
            self.bm25 = None
            return

        for chunk in chunks:
            text = chunk.get("text", "")
            tokenized = segment_vietnamese(text).split()
            self.corpus_tokens.append(tokenized)

        if self.corpus_tokens:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25 with Vietnamese tokenisation."""
        if self.bm25 is None:
            return []

        tokenized_query = segment_vietnamese(query).split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Pair docs with scores, filter zero-score, sort descending
        scored = [
            (i, score)
            for i, score in enumerate(scores)
            if score > 0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score in scored[:top_k]:
            chunk = self.documents[i]
            results.append(SearchResult(
                text=chunk["text"],
                score=float(score),
                metadata=chunk.get("metadata", {}),
                method="bm25",
            ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:  # noqa: BLE001 intentional fallback
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def _ensure_collection(self, collection: str) -> None:
        """Create or recreate the collection with the correct vector config."""
        from qdrant_client.models import Distance, VectorParams

        try:
            # Try the modern non-deprecated approach first
            if self.client.collection_exists(collection):
                self.client.delete_collection(collection)
            self.client.create_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
        except AttributeError:
            # Fall back to deprecated recreate_collection for older qdrant-client versions
            import warnings
            warnings.warn(
                "recreate_collection is deprecated; update qdrant-client for "
                "future compatibility.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

    def _stable_id(self, text: str) -> int:
        """Deterministic 64-bit integer ID from SHA-256 of chunk text."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return int.from_bytes(h[:8], byteorder="big", signed=False)

    def _make_payload(self, chunk: dict) -> dict:
        """Build a JSON-safe payload from a chunk."""
        payload = {"text": chunk.get("text", "")}
        for k, v in chunk.get("metadata", {}).items():
            payload[k] = _json_safe(v)
        return payload

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant with dense vectors."""
        if not chunks:
            return

        self._ensure_collection(collection)

        texts = [c["text"] for c in chunks]
        encoder = self._get_encoder()
        vectors = encoder.encode(texts, show_progress_bar=False)

        from qdrant_client.models import PointStruct
        # strict=True: mỗi chunk phải có đúng 1 vector — nếu encode bị lệch số
        # lượng thì đó là lỗi indexing thật, không nên âm thầm bỏ sót chunk.
        points = [
            PointStruct(
                id=self._stable_id(c["text"]),
                vector=v.tolist(),
                payload=self._make_payload(c),
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]

        self.client.upsert(collection, points)

    def search(self, query: str, top_k: int = DENSE_TOP_K,
               collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not query or not query.strip():
            return []

        encoder = self._get_encoder()
        query_vector = encoder.encode(query).tolist()

        try:
            response = self.client.query_points(
                collection,
                query=query_vector,
                limit=top_k,
            )
        except Exception as e:  # noqa: BLE001 intentional fallback
            print(
                f"  ⚠️  Dense search failed (collection={collection!r}): {e}",
                flush=True,
            )
            return []

        results = []
        for pt in response.points:
            payload = pt.payload
            text = payload.pop("text", "")
            results.append(SearchResult(
                text=text,
                score=pt.score,
                metadata=payload,
                method="dense",
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using Reciprocal Rank Fusion.

    score(d) = Σ  1 / (k + rank + 1)   (rank is 0-based)
    Deduplicates by text (not score).  Ranks from different retrieval methods are
    never compared directly — only their positions matter.
    """
    rrf_scores: dict[str, tuple[float, SearchResult]] = {}

    for result_list in results_list:
        if not result_list:
            continue
        for rank, result in enumerate(result_list):
            if not result.text:
                continue
            if result.text not in rrf_scores:
                rrf_scores[result.text] = (0.0, result)
            rrf_scores[result.text] = (
                rrf_scores[result.text][0] + 1.0 / (k + rank + 1),
                rrf_scores[result.text][1],
            )

    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        SearchResult(
            text=r.text,
            score=score,
            metadata=r.metadata,
            method="hybrid",
        )
        for score, r in sorted_results[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF."""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nghỉ phép năm')}")
