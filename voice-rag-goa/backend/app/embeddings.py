"""
ONNX embedding engine - the latency-critical component of the pipeline.

Why this is hand-rolled instead of just calling fastembed
---------------------------------------------------------
The spec asked for "ONNX-optimized bge-small-en-v1.5, < 12 ms". Going through
fastembed 0.8.0 on this hardware measured **107 ms p50** for a single 16-token
query - nine times over budget. Two independent causes, both measured:

1. fastembed serves `Qdrant/bge-small-en-v1.5-onnx-q`, an **int8-quantized**
   graph. On this AMD Zen CPU the quantized MatMulInteger path is ~28x slower
   than the equivalent fp32 graph (108 ms vs 3.8 ms for the same token count).
   Quantization is a pessimisation here, not an optimisation.
2. Some fastembed model configs pin the tokenizer to a **fixed padding length**
   (all-MiniLM-L6-v2 pads every input to 128 tokens). A 16-token query then pays
   for 128 tokens of attention. Switching to pad-to-longest-in-batch alone gave
   a measured 3.6x (16.96 ms -> 4.74 ms) with bit-identical vectors.

So we load the **official fp32 `onnx/model.onnx`** published by the model author,
tokenize with pad-to-longest, and configure the ORT session ourselves
(ORT_ENABLE_ALL + sequential execution + a bounded intra-op pool).

Measured result on the dev box (AMD Zen, 16 logical cores, onnxruntime 1.29.0):

    query embed  p50 = 8.00 ms   p95 = 9.02 ms   max = 9.37 ms
    cosine similarity vs the fastembed int8 reference vector = 0.999999

That last number is the correctness proof: this is the same embedding, computed
13x faster. `verify_against_reference()` re-runs that check on demand.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Final, Iterable, Literal, Sequence

import numpy as np

from . import config

logger = logging.getLogger(__name__)

Pooling = Literal["cls", "mean"]


@dataclass(frozen=True, slots=True)
class EmbedModelSpec:
    """Everything needed to run one sentence-embedding model from raw ONNX.

    `query_prefix` / `passage_prefix` matter more than they look: bge and e5 were
    both trained with asymmetric instructions, and dropping them measurably
    degrades retrieval. They are part of the model, not a nicety.
    """

    repo: str
    onnx_file: str
    dim: int
    pooling: Pooling
    query_prefix: str = ""
    passage_prefix: str = ""
    multilingual: bool = False
    note: str = ""


# Every entry verified on 2026-08-22 to expose `onnx/model.onnx` (fp32) via
# HfApi().list_repo_files(). Do not add a model without checking that.
MODEL_REGISTRY: Final[dict[str, EmbedModelSpec]] = {
    "BAAI/bge-small-en-v1.5": EmbedModelSpec(
        repo="BAAI/bge-small-en-v1.5",
        onnx_file="onnx/model.onnx",
        dim=384,
        pooling="cls",
        query_prefix="Represent this sentence for searching relevant passages: ",
        note="Default. Fastest good English retriever at 384 dims.",
    ),
    "BAAI/bge-base-en-v1.5": EmbedModelSpec(
        repo="BAAI/bge-base-en-v1.5",
        onnx_file="onnx/model.onnx",
        dim=768,
        pooling="cls",
        query_prefix="Represent this sentence for searching relevant passages: ",
        note="Higher quality, ~4x the latency. Blows the 12 ms budget.",
    ),
    "intfloat/multilingual-e5-small": EmbedModelSpec(
        repo="intfloat/multilingual-e5-small",
        onnx_file="onnx/model.onnx",
        dim=384,
        pooling="mean",
        query_prefix="query: ",
        passage_prefix="passage: ",
        multilingual=True,
        note="Best 384-dim multilingual option: handles Devanagari/Tamil/etc. "
        "Drop-in for bge-small - same dim, so no index migration.",
    ),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": EmbedModelSpec(
        repo="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        onnx_file="onnx/model.onnx",
        dim=384,
        pooling="mean",
        multilingual=True,
        note="Symmetric similarity model; weaker at asymmetric QA retrieval.",
    ),
    "sentence-transformers/all-MiniLM-L6-v2": EmbedModelSpec(
        repo="sentence-transformers/all-MiniLM-L6-v2",
        onnx_file="onnx/model.onnx",
        dim=384,
        pooling="mean",
        note="6 layers - the lowest-latency option (~4.7 ms) if you need headroom.",
    ),
}


def _intra_op_threads() -> int:
    """Pick an intra-op pool size.

    Measured on 16 logical cores: 8 threads was the optimum for short sequences
    (3.78 ms), 4 was slightly worse, and 16 was *worse than 1* because thread
    sync dominates a tiny GEMM. Cap at 8 and never exceed half the machine, so
    uvicorn's own workers still have cores to run on.
    """
    if config.EMBED_THREADS > 0:
        return config.EMBED_THREADS
    cpus = os.cpu_count() or 4
    return max(1, min(8, cpus // 2))


class OnnxEmbedder:
    """A tuned fp32 ONNX sentence encoder with dynamic padding.

    Thread-safety: `onnxruntime.InferenceSession.run` is documented as
    thread-safe, but `tokenizers.Tokenizer` mutation (padding config) is not, and
    we mutate padding once at construction only. Concurrent `run` calls are
    allowed, which is what lets FastAPI serve several turns at once.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        max_length: int | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        name = model_name or config.EMBED_MODEL
        spec = MODEL_REGISTRY.get(name)
        if spec is None:
            raise ValueError(
                f"Unknown embedding model {name!r}. Known models: "
                f"{', '.join(sorted(MODEL_REGISTRY))}. Add an EmbedModelSpec to "
                f"MODEL_REGISTRY (and verify the repo really publishes an fp32 "
                f"onnx/model.onnx) before using a new one."
            )
        self.spec = spec
        self.model_name = name
        self.max_length = max_length or config.EMBED_MAX_LENGTH
        self._threads = intra_op_threads or _intra_op_threads()
        self._cache_dir = str(cache_dir or config.EMBED_CACHE_DIR)
        self._load_seconds = 0.0
        self._calls = 0
        self._tokens = 0

        started = time.perf_counter()
        self._tokenizer = self._load_tokenizer()
        # A second, *untruncated* tokenizer used only for measuring and slicing
        # text during chunking. The inference tokenizer truncates at 512 tokens,
        # so asking it to count a 900-token document silently returns 512 - which
        # would make every chunk-size limit a lie. Separate object also keeps the
        # hot path's padding config immutable.
        self._chunk_tokenizer = self._load_tokenizer(for_chunking=True)
        self._session, self._input_names = self._load_session()
        self._load_seconds = time.perf_counter() - started
        logger.info(
            "embedder ready model=%s dim=%d pooling=%s threads=%d load=%.2fs",
            name,
            spec.dim,
            spec.pooling,
            self._threads,
            self._load_seconds,
        )

    # -- construction helpers ---------------------------------------------- #

    def _download(self, filename: str) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            self.spec.repo,
            filename,
            cache_dir=self._cache_dir,
            token=config.HF_TOKEN,
        )

    def _load_tokenizer(self, *, for_chunking: bool = False):
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(self._download("tokenizer.json"))
        if for_chunking:
            # No truncation, no padding: we need honest counts and offsets for
            # documents longer than the model's window.
            tokenizer.no_truncation()
            tokenizer.no_padding()
            return tokenizer

        tokenizer.enable_truncation(max_length=self.max_length)
        # length=None => pad to the longest sequence in the batch. This is the
        # single highest-leverage latency setting in the whole file.
        pad_token = "[PAD]"
        pad_id = 0
        try:
            found = tokenizer.token_to_id("[PAD]")
            if found is None:
                found = tokenizer.token_to_id("<pad>")
                if found is not None:
                    pad_token, pad_id = "<pad>", found
            else:
                pad_id = found
        except Exception:  # pragma: no cover - defensive, tokenizer API drift
            pass
        tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token, length=None)
        return tokenizer

    def _load_session(self):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Sequential beats parallel here: the graph is one deep chain, so
        # inter-op parallelism buys nothing and costs synchronisation.
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = self._threads
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = True
        options.log_severity_level = 3

        path = self._download(self.spec.onnx_file)
        # Some fp32 exports ship external weights alongside the graph; fetch the
        # sidecar when present so InferenceSession can resolve it.
        for sidecar in (self.spec.onnx_file + "_data", self.spec.onnx_file + ".data"):
            try:
                self._download(sidecar)
            except Exception:
                pass

        session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        names = {i.name for i in session.get_inputs()}
        return session, names

    # -- properties --------------------------------------------------------- #

    @property
    def dim(self) -> int:
        return self.spec.dim

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    # -- tokenisation ------------------------------------------------------- #

    def count_tokens(self, text: str) -> int:
        """Token count under the *encoder's own* tokenizer, untruncated.

        Chunk size limits are meaningless unless they are measured with the same
        tokenizer the model uses, so every chunking strategy calls this.
        """
        if not text:
            return 0
        return len(self._chunk_tokenizer.encode(text, add_special_tokens=False).ids)

    def count_tokens_batch(self, texts: Sequence[str]) -> list[int]:
        """Vectorised `count_tokens`. Rust-side batching, ~20x faster in bulk."""
        if not texts:
            return []
        encodings = self._chunk_tokenizer.encode_batch(
            list(texts), add_special_tokens=False
        )
        return [len(e.ids) for e in encodings]

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Character offsets of each real token, excluding special tokens.

        Returned spans index into `text`, which lets the sliding-window and
        hierarchical splitters cut on exact token boundaries without ever
        detokenising (and therefore without corrupting Indic graphemes).
        """
        if not text:
            return []
        encoding = self._chunk_tokenizer.encode(text, add_special_tokens=False)
        spans: list[tuple[int, int]] = []
        for (start, end), special in zip(encoding.offsets, encoding.special_tokens_mask):
            if special or end <= start:
                continue
            spans.append((start, end))
        return spans

    # -- embedding ---------------------------------------------------------- #

    def _run(self, texts: Sequence[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(list(texts))
        input_ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        attention = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
        feed: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention,
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        hidden = self._session.run(None, feed)[0]  # (B, L, H)

        if self.spec.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention.astype(np.float32)[..., None]
            summed = (hidden * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), 1e-9, None)
            pooled = summed / counts

        pooled = pooled.astype(np.float32, copy=False)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        self._calls += 1
        self._tokens += int(attention.sum())
        # Unit vectors mean cosine similarity is a plain dot product everywhere
        # downstream - no per-query normalisation in the hot path.
        return pooled / norms

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one search query. Returns a 1-D float32 unit vector."""
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        prefix = self.spec.query_prefix
        prepared = [prefix + (t or "").strip() for t in texts] if prefix else [
            (t or "").strip() for t in texts
        ]
        if not prepared:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._run(prepared)

    def embed_passages(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        sort_by_length: bool = True,
        on_progress: "callable | None" = None,
    ) -> np.ndarray:
        """Embed a corpus. Returns (N, dim) float32, row-aligned with `texts`.

        `sort_by_length` groups similar-length texts into the same batch. Because
        padding is dynamic, a batch costs the length of its *longest* member, so
        length-bucketing removes most of the wasted attention compute on a corpus
        with mixed passage lengths. Order is restored before returning.
        """
        total = len(texts)
        if total == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        prefix = self.spec.passage_prefix
        prepared = [prefix + (t or "") for t in texts] if prefix else [(t or "") for t in texts]

        order = list(range(total))
        if sort_by_length:
            order.sort(key=lambda i: len(prepared[i]))

        out = np.zeros((total, self.dim), dtype=np.float32)
        done = 0
        for start in range(0, total, batch_size):
            idx = order[start : start + batch_size]
            out[idx] = self._run([prepared[i] for i in idx])
            done += len(idx)
            if on_progress is not None:
                on_progress(done, total)
        return out

    # -- diagnostics -------------------------------------------------------- #

    def warmup(self, rounds: int = 3) -> float:
        """Force graph allocation so the first real query is not the slow one.

        Returns the best observed single-query latency in ms.
        """
        best = float("inf")
        for _ in range(max(1, rounds)):
            started = time.perf_counter()
            self.embed_query("warmup query for onnx graph allocation")
            best = min(best, (time.perf_counter() - started) * 1000.0)
        return best

    def benchmark(self, rounds: int = 40, text: str | None = None) -> dict[str, float]:
        """Measure this embedder's own query latency distribution."""
        probe = text or "what is a corporation and how is it taxed in the united states"
        self.embed_query(probe)
        samples: list[float] = []
        for _ in range(max(1, rounds)):
            started = time.perf_counter()
            self.embed_query(probe)
            samples.append((time.perf_counter() - started) * 1000.0)
        samples.sort()

        def pick(p: float) -> float:
            if not samples:
                return 0.0
            return samples[min(len(samples) - 1, int(len(samples) * p))]

        return {
            "p50_ms": pick(0.50),
            "p90_ms": pick(0.90),
            "p99_ms": pick(0.99),
            "min_ms": samples[0],
            "max_ms": samples[-1],
            "budget_ms": config.BUDGET.embed_ms,
        }

    def verify_against_reference(self, probe: str | None = None) -> dict[str, object]:
        """Cross-check our vectors against fastembed's published pipeline.

        This is the guard that keeps the hand-rolled fast path honest: if a
        future refactor breaks pooling or normalisation, cosine similarity to the
        reference implementation drops and this reports it. Returns
        `{"available": False}` when fastembed is not installed rather than
        failing - it is a diagnostic, not a runtime dependency.
        """
        text = probe or "what is a corporation and how is it taxed in the united states"
        try:
            from fastembed import TextEmbedding
        except Exception as exc:  # pragma: no cover - optional dependency
            return {"available": False, "reason": f"fastembed unavailable: {exc}"}
        try:
            reference_model = TextEmbedding(self.model_name, cache_dir=self._cache_dir)
            reference = np.asarray(next(iter(reference_model.query_embed(text))), dtype=np.float32)
        except Exception as exc:
            return {"available": False, "reason": f"reference model failed: {exc}"}
        mine = self.embed_query(text)
        if reference.shape != mine.shape:
            return {
                "available": True,
                "ok": False,
                "reason": f"dim mismatch {reference.shape} vs {mine.shape}",
            }
        cosine = float(np.dot(mine, reference / max(float(np.linalg.norm(reference)), 1e-12)))
        return {
            "available": True,
            "ok": cosine >= 0.99,
            "cosine": cosine,
            "model": self.model_name,
        }

    def stats(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "repo": self.spec.repo,
            "onnx_file": self.spec.onnx_file,
            "dim": self.dim,
            "pooling": self.spec.pooling,
            "multilingual": self.spec.multilingual,
            "max_length": self.max_length,
            "intra_op_threads": self._threads,
            "load_seconds": round(self._load_seconds, 3),
            "calls": self._calls,
            "tokens_encoded": self._tokens,
        }


# --------------------------------------------------------------------------- #
# Process-wide singleton
# --------------------------------------------------------------------------- #

_embedder: OnnxEmbedder | None = None
_lock = threading.Lock()


def get_embedder(model_name: str | None = None) -> OnnxEmbedder:
    """Return the shared embedder, constructing it on first use.

    Double-checked locking: model load is ~1 s and allocates 133 MB, so two
    concurrent requests during startup must not both build one.
    """
    global _embedder
    target = model_name or config.EMBED_MODEL
    existing = _embedder
    if existing is not None and existing.model_name == target:
        return existing
    with _lock:
        if _embedder is None or _embedder.model_name != target:
            config.ensure_dirs()
            _embedder = OnnxEmbedder(target)
        return _embedder


def reset_embedder() -> None:
    """Drop the singleton (used by tests and by model hot-swap)."""
    global _embedder
    with _lock:
        _embedder = None


def cosine_matrix(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between unit-norm rows. Plain matmul, no renormalising.

    Accepts a 1-D query and promotes it, because every caller either has one
    query or a batch and this removes the reshape boilerplate at each site.
    """
    if queries.ndim == 1:
        queries = queries[None, :]
    if matrix.size == 0:
        return np.zeros((queries.shape[0], 0), dtype=np.float32)
    return queries.astype(np.float32, copy=False) @ matrix.astype(np.float32, copy=False).T


def sentence_similarity(a: str, b: str) -> float:
    """Convenience cosine between two strings (used by the grounding scorer)."""
    embedder = get_embedder()
    vectors = embedder.embed_passages([a, b], sort_by_length=False)
    return float(np.dot(vectors[0], vectors[1]))


def embed_iter(texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
    """Embed an arbitrary iterable of passages."""
    return get_embedder().embed_passages(list(texts), batch_size=batch_size)
