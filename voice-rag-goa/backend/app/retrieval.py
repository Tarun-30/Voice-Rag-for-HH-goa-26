"""
Hybrid retrieval: dense (ONNX embeddings) + sparse (BM25), fused.

Two interchangeable backends behind one `Retriever` API:

* **memory**  - the low-tail-latency hot path. The whole strategy's vector
  matrix lives in RAM as a contiguous float32 array; a query is one
  `q @ V.T` matmul (exact cosine, no approximation) plus a `rank_bm25` scan.
  For ~25k chunks that is ~38 MB and 2-4 ms - and crucially the *tail* is flat
  because there is no index structure to have a bad day.
* **lancedb** - the scalable path. IVF_PQ/HNSW ANN + native BM25 FTS, durable on
  disk, for corpora that outgrow RAM.

`RETRIEVAL_BACKEND="both"` loads both so `/api/analytics` can benchmark them
head to head. Serving prefers memory when present (faster, exact).

Fusion is either Reciprocal Rank Fusion (rank-based, robust to score-scale
mismatch between dense and sparse) or a min-max-normalised weighted blend.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from . import config
from . import textutil
from .embeddings import OnnxEmbedder, get_embedder
from .ingest import INDEX_DIR, indexed_strategies
from .schemas import RetrievedChunk

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Fusion primitives
# --------------------------------------------------------------------------- #


def _top_indices(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the k highest scores, in descending score order."""
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    # argpartition is O(n); we only sort the k survivors.
    part = np.argpartition(scores, -k)[-k:]
    return [int(i) for i in part[np.argsort(scores[part])[::-1]]]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]], k: int
) -> dict[int, float]:
    """RRF: score(d) = sum_l 1 / (k + rank_l(d)). Rank is 0-based here."""
    fused: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, doc in enumerate(ranking):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return fused


def _minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo < 1e-12:
        return {key: 1.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


# --------------------------------------------------------------------------- #
# In-memory backend
# --------------------------------------------------------------------------- #


class MemoryStrategyIndex:
    """One strategy's chunks held in RAM for exact dense + BM25 search."""

    def __init__(self, strategy: str):
        from rank_bm25 import BM25Okapi

        self.strategy = strategy
        out = INDEX_DIR / strategy
        self.vectors: np.ndarray = np.load(out / "vectors.npy").astype(np.float32)

        rows = _read_parquet_rows(out / "chunks.parquet")
        self.rows: list[dict] = rows
        if len(rows) != self.vectors.shape[0]:
            raise RuntimeError(
                f"{strategy}: {len(rows)} chunk rows but "
                f"{self.vectors.shape[0]} vectors - index is corrupt, rebuild it."
            )

        manifest_path = out / "manifest.json"
        self.manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        started = time.perf_counter()
        corpus_tokens = [textutil.tokenize_for_bm25(row["text"]) for row in rows]
        self.bm25 = BM25Okapi(corpus_tokens)
        logger.info(
            "memory index %s: %d chunks, dim=%d, bm25 built in %.2fs",
            strategy,
            len(rows),
            self.vectors.shape[1] if self.vectors.size else 0,
            time.perf_counter() - started,
        )

    @property
    def size(self) -> int:
        return len(self.rows)

    def dense(self, query_vector: np.ndarray, k: int) -> dict[int, float]:
        if self.vectors.size == 0:
            return {}
        scores = self.vectors @ query_vector.astype(np.float32)
        return {i: float(scores[i]) for i in _top_indices(scores, k)}

    def sparse(self, query_tokens: list[str], k: int) -> dict[int, float]:
        if not query_tokens or self.size == 0:
            return {}
        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)
        return {i: float(scores[i]) for i in _top_indices(scores, k)}

    def row(self, index: int) -> dict:
        return self.rows[index]


# --------------------------------------------------------------------------- #
# LanceDB backend
# --------------------------------------------------------------------------- #


class LanceStrategyIndex:
    """One strategy's chunks served from a LanceDB table (ANN + native FTS)."""

    def __init__(self, strategy: str, table):
        self.strategy = strategy
        self.table = table
        self._has_fts = True

    @property
    def size(self) -> int:
        try:
            return self.table.count_rows()
        except Exception:
            return 0

    def dense(self, query_vector: np.ndarray, k: int) -> list[dict]:
        results = (
            self.table.search(query_vector.astype(np.float32), vector_column_name="vector")
            .metric(config.VECTOR_METRIC)
            .limit(k)
            .to_list()
        )
        # LanceDB returns cosine *distance* in `_distance`; convert to similarity.
        for item in results:
            item["_score"] = 1.0 - float(item.get("_distance", 0.0))
        return results

    def sparse(self, query_text: str, k: int) -> list[dict]:
        if not self._has_fts:
            return []
        try:
            results = self.table.search(query_text, query_type="fts").limit(k).to_list()
        except Exception as exc:  # pragma: no cover - FTS may be absent
            logger.debug("lancedb fts failed for %s: %s", self.strategy, exc)
            self._has_fts = False
            return []
        for item in results:
            item["_score"] = float(item.get("_score", item.get("score", 0.0)))
        return results


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #


class Retriever:
    """Loads every indexed strategy for a backend and serves fused search."""

    def __init__(self, backend: str | None = None, embedder: OnnxEmbedder | None = None):
        self.backend = backend or config.RETRIEVAL_BACKEND
        self.embedder = embedder or get_embedder()
        self.memory: dict[str, MemoryStrategyIndex] = {}
        self.lance: dict[str, LanceStrategyIndex] = {}
        self._load()

    def _load(self) -> None:
        strategies = indexed_strategies(self.backend)
        if self.backend in ("memory", "both"):
            for strategy in strategies:
                try:
                    self.memory[strategy] = MemoryStrategyIndex(strategy)
                except Exception as exc:
                    logger.warning("failed loading memory index %s: %s", strategy, exc)
        if self.backend in ("lancedb", "both"):
            self._load_lance(strategies)
        logger.info(
            "retriever ready backend=%s memory=%s lancedb=%s",
            self.backend,
            list(self.memory),
            list(self.lance),
        )

    def _load_lance(self, strategies: Sequence[str]) -> None:
        try:
            import lancedb

            db = lancedb.connect(str(config.LANCEDB_URI))
            names = set(db.table_names())
        except Exception as exc:
            logger.warning("lancedb unavailable: %s", exc)
            return
        for strategy in config.ALL_STRATEGIES:
            name = config.table_name(strategy)
            if name in names:
                try:
                    self.lance[strategy] = LanceStrategyIndex(strategy, db.open_table(name))
                except Exception as exc:
                    logger.warning("failed opening lance table %s: %s", name, exc)

    # -- introspection ------------------------------------------------------ #

    @property
    def strategies(self) -> list[str]:
        return sorted(set(self.memory) | set(self.lance))

    @property
    def ready(self) -> bool:
        return bool(self.memory or self.lance)

    def total_chunks(self) -> int:
        if self.memory:
            return sum(index.size for index in self.memory.values())
        return sum(index.size for index in self.lance.values())

    def has_strategy(self, strategy: str) -> bool:
        return strategy in self.memory or strategy in self.lance

    def resolve_backend(self, requested: str | None) -> str:
        """Which concrete backend to use for a search."""
        if requested in ("memory", "lancedb"):
            return requested
        return "memory" if self.memory else "lancedb"

    # -- search ------------------------------------------------------------- #

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.embed_query(query)

    def search(
        self,
        query: str,
        *,
        query_vector: np.ndarray | None = None,
        strategy: str | None = None,
        mode: str = "hybrid",
        top_k: int | None = None,
        candidates: int | None = None,
        backend: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve and fuse. `query_vector` may be supplied to isolate the
        embedding cost from the retrieval cost when timing the pipeline."""
        strategy = strategy or config.DEFAULT_STRATEGY
        if not self.has_strategy(strategy):
            available = self.strategies
            if not available:
                return []
            strategy = available[0]
        top_k = top_k or config.RETRIEVE_TOP_K
        candidates = candidates or config.RETRIEVE_CANDIDATES
        backend = self.resolve_backend(backend)
        if query_vector is None and mode != "sparse":
            query_vector = self.embed_query(query)

        if backend == "memory" and strategy in self.memory:
            return self._search_memory(query, query_vector, strategy, mode, top_k, candidates)
        if strategy in self.lance:
            return self._search_lance(query, query_vector, strategy, mode, top_k, candidates)
        # Fallback if the requested backend lacks this strategy.
        if strategy in self.memory:
            return self._search_memory(query, query_vector, strategy, mode, top_k, candidates)
        return []

    def _search_memory(
        self,
        query: str,
        query_vector: np.ndarray | None,
        strategy: str,
        mode: str,
        top_k: int,
        candidates: int,
    ) -> list[RetrievedChunk]:
        index = self.memory[strategy]
        dense = (
            index.dense(query_vector, candidates)
            if mode in ("hybrid", "dense") and query_vector is not None
            else {}
        )
        sparse = (
            index.sparse(textutil.tokenize_for_bm25(query), candidates)
            if mode in ("hybrid", "sparse")
            else {}
        )
        fused = self._fuse(dense, sparse, mode, top_k)
        return [
            self._hydrate(index.row(doc_id), score, dense.get(doc_id, 0.0), sparse.get(doc_id, 0.0), rank)
            for rank, (doc_id, score) in enumerate(fused)
        ]

    def _search_lance(
        self,
        query: str,
        query_vector: np.ndarray | None,
        strategy: str,
        mode: str,
        top_k: int,
        candidates: int,
    ) -> list[RetrievedChunk]:
        index = self.lance[strategy]
        dense_rows = (
            index.dense(query_vector, candidates)
            if mode in ("hybrid", "dense") and query_vector is not None
            else []
        )
        sparse_rows = (
            index.sparse(query, candidates) if mode in ("hybrid", "sparse") else []
        )
        # Key rows by chunk_id so the two result sets can be fused by identity.
        by_id: dict[str, dict] = {}
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}
        for row in dense_rows:
            by_id[row["chunk_id"]] = row
            dense_scores[row["chunk_id"]] = row.get("_score", 0.0)
        for row in sparse_rows:
            by_id[row["chunk_id"]] = row
            sparse_scores[row["chunk_id"]] = row.get("_score", 0.0)

        dense_map = {cid: dense_scores[cid] for cid in dense_scores}
        sparse_map = {cid: sparse_scores[cid] for cid in sparse_scores}
        fused = self._fuse_ids(dense_map, sparse_map, mode, top_k)
        return [
            self._hydrate(by_id[cid], score, dense_scores.get(cid, 0.0), sparse_scores.get(cid, 0.0), rank)
            for rank, (cid, score) in enumerate(fused)
        ]

    # -- fusion ------------------------------------------------------------- #

    def _fuse(
        self, dense: dict[int, float], sparse: dict[int, float], mode: str, top_k: int
    ) -> list[tuple[int, float]]:
        if mode == "dense":
            return sorted(dense.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        if mode == "sparse":
            return sorted(sparse.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return self._hybrid(dense, sparse, top_k)

    def _fuse_ids(self, dense, sparse, mode, top_k):
        return self._fuse(dense, sparse, mode, top_k)

    def _hybrid(self, dense, sparse, top_k):
        if config.FUSION_METHOD == "rrf":
            dense_rank = [d for d, _ in sorted(dense.items(), key=lambda kv: kv[1], reverse=True)]
            sparse_rank = [d for d, _ in sorted(sparse.items(), key=lambda kv: kv[1], reverse=True)]
            fused = reciprocal_rank_fusion([dense_rank, sparse_rank], config.RRF_K)
        else:
            dense_norm = _minmax(dense)
            sparse_norm = _minmax(sparse)
            alpha = config.HYBRID_ALPHA
            fused = {}
            for doc in set(dense_norm) | set(sparse_norm):
                fused[doc] = alpha * dense_norm.get(doc, 0.0) + (1 - alpha) * sparse_norm.get(doc, 0.0)
        return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    # -- hydration ---------------------------------------------------------- #

    @staticmethod
    def _hydrate(row: dict, score: float, dense: float, sparse: float, rank: int) -> RetrievedChunk:
        parent = row.get("parent_id") or None
        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            text=row["text"],
            context_text=row.get("context_text") or row["text"],
            strategy=row["strategy"],
            score=round(float(score), 6),
            dense_score=round(float(dense), 6),
            sparse_score=round(float(sparse), 6),
            rank=rank,
            title=row.get("title", ""),
            passage_id=row.get("passage_id", ""),
            query_id=int(row.get("query_id", -1) or -1),
            language=row.get("language", ""),
            domain=row.get("domain", ""),
            token_count=int(row.get("token_count", 0) or 0),
            parent_id=parent if parent else None,
            is_gold=bool(row.get("is_gold", False)),
        )


# --------------------------------------------------------------------------- #
# Parquet helper + singleton
# --------------------------------------------------------------------------- #


def _read_parquet_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


_retriever: Retriever | None = None
_lock = threading.Lock()


def get_retriever(*, reload: bool = False) -> Retriever:
    global _retriever
    if _retriever is not None and not reload:
        return _retriever
    with _lock:
        if _retriever is None or reload:
            _retriever = Retriever()
        return _retriever


def reset_retriever() -> None:
    global _retriever
    with _lock:
        _retriever = None
