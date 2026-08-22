"""
Central configuration for the Goa Hacker House Voice-RAG pipeline.

Everything tunable lives here. Values are read once at import time from the
environment (with `.env` support) so that hot-path code never touches os.environ.

Design notes
------------
* No pydantic-settings dependency: plain env parsing keeps cold-start fast and
  avoids a wheel that may lag new CPython releases.
* Windows consoles default to cp1252, which explodes on Devanagari/Tamil text
  the moment we log a retrieved passage. `_force_utf8_stdio()` fixes that
  process-wide before anything else can print.
"""

from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

BACKEND_DIR: Final[Path] = Path(__file__).resolve().parent.parent
PROJECT_ROOT: Final[Path] = BACKEND_DIR.parent
DATA_DIR: Final[Path] = BACKEND_DIR / "data"

# Loaded before any getenv call below.
load_dotenv(BACKEND_DIR / ".env", override=False)


def _force_utf8_stdio() -> None:
    """Make stdout/stderr UTF-8 so Indic script logging cannot crash the server.

    CPython on Windows picks the ANSI code page (cp1252) for std streams when
    they are redirected. Printing a single Devanagari character then raises
    UnicodeEncodeError from inside a log call, which in an async server surfaces
    as an unrelated 500. We reconfigure rather than wrap when possible so that
    the original file descriptors are preserved.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
                continue
            except (ValueError, OSError):
                pass
        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            setattr(sys, name, io.TextIOWrapper(buffer, encoding="utf-8", errors="replace"))


_force_utf8_stdio()


# --------------------------------------------------------------------------- #
# Env helpers
# --------------------------------------------------------------------------- #

def _env_str(key: str, default: str) -> str:
    raw = os.getenv(key)
    return raw.strip() if raw and raw.strip() else default


def _env_opt(key: str) -> str | None:
    raw = os.getenv(key)
    raw = raw.strip() if raw else ""
    # Treat the placeholder values shipped in .env.example as "unset" so a
    # half-filled .env degrades to offline mode instead of sending bad keys.
    if not raw or raw.lower().startswith(("your_", "changeme", "<", "sk-xxx")):
        return None
    return raw


def _env_int(key: str, default: int) -> int:
    try:
        return int(str(os.getenv(key, default)).strip())
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(str(os.getenv(key, default)).strip())
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# --------------------------------------------------------------------------- #
# Dataset: ai4bharat/MSMARCO-XI
# --------------------------------------------------------------------------- #
#
# Verified against the live repo on 2026-08-22 via HfApi().repo_info():
#   * 27 parquet shards, ONE per language per split, 55.6 GB total.
#   * Every shard contains exactly ONE row group, so "read one row group" is a
#     whole-file read. Partial reads must therefore rely on *column projection*
#     plus an early break out of iter_batches().
#   * Leaf column sizes for validation/hinval.parquet (462 MB total):
#       passages.Translated_passages.list.element -> 271.9 MB  <-- skipped
#       passages.English_passages.list.element    -> 173.0 MB
#       Answer / query / Eng_Answer / Eng_Query   ->  ~16.2 MB
#     Projecting away Translated_passages cuts time-to-first-batch from
#     ~30 s to ~11 s. Measured, not guessed.
#   * The HF datasets-server /rows API is BROKEN for this dataset
#     (ArrowNotImplementedError: nested data conversions for chunked arrays),
#     so we read parquet directly instead of using the viewer API.

HF_DATASET_REPO: Final[str] = _env_str("HF_DATASET_REPO", "ai4bharat/MSMARCO-XI")
HF_REPO_TYPE: Final[str] = "dataset"

# language-key -> (FLORES target_lang code, train shard, validation shard)
@dataclass(frozen=True, slots=True)
class ShardSpec:
    """One MSMARCO-XI language: its FLORES code and its parquet shard paths."""

    key: str
    label: str
    flores: str
    train: str | None
    validation: str | None

    def shard_for(self, split: str) -> str:
        path = self.train if split == "train" else self.validation
        if path is None:
            raise ValueError(
                f"Language {self.key!r} has no {split!r} shard in {HF_DATASET_REPO}. "
                f"Available: train={self.train!r} validation={self.validation!r}"
            )
        return path


# NOTE: Telugu ships a validation shard but no train shard in this repo.
LANGUAGE_SHARDS: Final[dict[str, ShardSpec]] = {
    s.key: s
    for s in (
        ShardSpec("asm", "Assamese", "asm_Beng", "train/asmtrain.parquet", "validation/asmval.parquet"),
        ShardSpec("ben", "Bengali", "ben_Beng", "train/bentrain.parquet", "validation/benval.parquet"),
        ShardSpec("guj", "Gujarati", "guj_Gujr", "train/gujtrain.parquet", "validation/gujval.parquet"),
        ShardSpec("hin", "Hindi", "hin_Deva", "train/hintrain.parquet", "validation/hinval.parquet"),
        ShardSpec("kan", "Kannada", "kan_Knda", "train/kantrain.parquet", "validation/kanval.parquet"),
        ShardSpec("mal", "Malayalam", "mal_Mlym", "train/maltrain.parquet", "validation/malval.parquet"),
        ShardSpec("mar", "Marathi", "mar_Deva", "train/martrain.parquet", "validation/marval.parquet"),
        ShardSpec("nep", "Nepali", "npi_Deva", "train/neptrain.parquet", "validation/nepval.parquet"),
        ShardSpec("ori", "Odia", "ory_Orya", "train/oritrain.parquet", "validation/orival.parquet"),
        ShardSpec("pan", "Punjabi", "pan_Guru", "train/pantrain.parquet", "validation/panval.parquet"),
        ShardSpec("san", "Sanskrit", "san_Deva", "train/santrain.parquet", "validation/sanval.parquet"),
        ShardSpec("tam", "Tamil", "tam_Taml", "train/tamtrain.parquet", "validation/tamval.parquet"),
        ShardSpec("tel", "Telugu", "tel_Telu", None, "validation/telval.parquet"),
        ShardSpec("urd", "Urdu", "urd_Arab", "train/urdtrain.parquet", "validation/urdval.parquet"),
    )
}

INGEST_LANGUAGES: Final[list[str]] = [
    lang.strip().lower()
    for lang in _env_str("INGEST_LANGUAGES", "hin").split(",")
    if lang.strip()
]
INGEST_SPLIT: Final[str] = _env_str("INGEST_SPLIT", "validation")
# Number of MSMARCO-XI *query rows* to pull per language. Each row carries ~10
# candidate passages, so 1200 rows ~= 12k passages ~= 25k chunks.
INGEST_ROW_LIMIT: Final[int] = _env_int("INGEST_ROW_LIMIT", 1200)
INGEST_BATCH_SIZE: Final[int] = _env_int("INGEST_BATCH_SIZE", 256)
# Also index the Indic translations of each passage (doubles bytes downloaded).
INGEST_INCLUDE_TRANSLATED: Final[bool] = _env_bool("INGEST_INCLUDE_TRANSLATED", False)

CORPUS_CACHE_PATH: Final[Path] = DATA_DIR / "corpus_cache.jsonl"
QUERYSET_CACHE_PATH: Final[Path] = DATA_DIR / "queryset.jsonl"

HF_TOKEN: Final[str | None] = _env_opt("HF_TOKEN") or _env_opt("HUGGING_FACE_HUB_TOKEN")


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

EMBED_MODEL: Final[str] = _env_str("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM: Final[int] = _env_int("EMBED_DIM", 384)
EMBED_MAX_LENGTH: Final[int] = _env_int("EMBED_MAX_LENGTH", 512)
# 0 => let onnxruntime decide. Pinning to physical cores avoids oversubscription
# when uvicorn workers and the ONNX pool fight over the same CPUs.
EMBED_THREADS: Final[int] = _env_int("EMBED_THREADS", 0)
EMBED_CACHE_DIR: Final[Path] = Path(_env_str("EMBED_CACHE_DIR", str(DATA_DIR / "models")))
SPARSE_MODEL: Final[str] = _env_str("SPARSE_MODEL", "Qdrant/bm25")

# Verified available in fastembed 0.8.0 via TextEmbedding.list_supported_models():
# this is the only *384-dimensional* multilingual model on offer, which makes it a
# drop-in swap for bge-small-en-v1.5 with zero schema migration. Set
# EMBED_MODEL to it when you want Devanagari/Tamil/etc. queries to match the
# Indic side of MSMARCO-XI rather than the English side.
EMBED_MODEL_MULTILINGUAL: Final[str] = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# Optional ONNX cross-encoder rerank stage. Off by default: it costs 20-40 ms,
# which blows the 200 ms budget. Turn on to trade latency for accuracy.
RERANKER_ENABLED: Final[bool] = _env_bool("RERANKER_ENABLED", False)
RERANKER_MODEL: Final[str] = _env_str("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")


# --------------------------------------------------------------------------- #
# Chunking strategies
# --------------------------------------------------------------------------- #

STRATEGY_SEMANTIC: Final[str] = "semantic"
STRATEGY_HIERARCHICAL: Final[str] = "hierarchical"
STRATEGY_SLIDING: Final[str] = "sliding"
ALL_STRATEGIES: Final[tuple[str, ...]] = (
    STRATEGY_SEMANTIC,
    STRATEGY_HIERARCHICAL,
    STRATEGY_SLIDING,
)
DEFAULT_STRATEGY: Final[str] = _env_str("DEFAULT_STRATEGY", STRATEGY_HIERARCHICAL)


@dataclass(frozen=True, slots=True)
class ChunkingParams:
    """Tunables shared by the three chunking pipelines.

    Token counts are measured with the embedding model's own tokenizer so the
    limits mean the same thing the encoder means.
    """

    # Strategy A - semantic
    semantic_min_tokens: int = 48
    semantic_max_tokens: int = 320
    # A sentence boundary becomes a chunk break when adjacent-sentence cosine
    # similarity drops more than this many standard deviations below the mean.
    semantic_breakpoint_sigma: float = 0.8
    semantic_buffer_size: int = 1  # sentences of context each side when embedding

    # Strategy B - hierarchical parent/child
    child_tokens: int = 128
    parent_tokens: int = 512
    child_overlap_tokens: int = 24

    # Strategy C - metadata-aware sliding window
    window_tokens: int = 256
    window_overlap_ratio: float = 0.25


CHUNKING: Final[ChunkingParams] = ChunkingParams(
    semantic_min_tokens=_env_int("SEMANTIC_MIN_TOKENS", 48),
    semantic_max_tokens=_env_int("SEMANTIC_MAX_TOKENS", 320),
    semantic_breakpoint_sigma=_env_float("SEMANTIC_BREAKPOINT_SIGMA", 0.8),
    child_tokens=_env_int("CHILD_TOKENS", 128),
    parent_tokens=_env_int("PARENT_TOKENS", 512),
    child_overlap_tokens=_env_int("CHILD_OVERLAP_TOKENS", 24),
    window_tokens=_env_int("WINDOW_TOKENS", 256),
    window_overlap_ratio=_env_float("WINDOW_OVERLAP_RATIO", 0.25),
)


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #

LANCEDB_URI: Final[str] = _env_str("LANCEDB_URI", str(DATA_DIR / "lancedb"))
TABLE_PREFIX: Final[str] = _env_str("TABLE_PREFIX", "msmarco_xi")
VECTOR_METRIC: Final[str] = _env_str("VECTOR_METRIC", "cosine")
# LanceDB refuses to build an ANN index on tiny tables; below this row count we
# stay on exact (brute-force) search, which is faster there anyway.
ANN_MIN_ROWS: Final[int] = _env_int("ANN_MIN_ROWS", 10_000)
IVF_PARTITION_DIVISOR: Final[int] = _env_int("IVF_PARTITION_DIVISOR", 256)

# Retrieval backend for the hot path:
#   "memory"  - NumPy exact dense matmul + rank_bm25 in RAM. At our corpus size
#               (~25k chunks x 384 dims = 38 MB) a full matmul is ~2-4 ms and
#               has zero index-drift or IO variance. This is the default because
#               the 200 ms budget cares about tail latency, not asymptotics.
#   "lancedb" - LanceDB IVF_PQ/HNSW ANN + native BM25 full-text index. Proves
#               the design scales past RAM and is what you'd ship at 10M chunks.
#   "both"    - keep both live so /api/analytics can benchmark them head to head.
RETRIEVAL_BACKEND: Final[str] = _env_str("RETRIEVAL_BACKEND", "both").lower()

RETRIEVE_TOP_K: Final[int] = _env_int("RETRIEVE_TOP_K", 5)
RETRIEVE_CANDIDATES: Final[int] = _env_int("RETRIEVE_CANDIDATES", 24)
HYBRID_ALPHA: Final[float] = _env_float("HYBRID_ALPHA", 0.65)  # 1.0 = pure dense
RRF_K: Final[int] = _env_int("RRF_K", 60)
# "rrf" = reciprocal rank fusion (rank-based, scale-free).
# "weighted" = min-max normalise both score lists then blend with HYBRID_ALPHA.
FUSION_METHOD: Final[str] = _env_str("FUSION_METHOD", "rrf").lower()


# --------------------------------------------------------------------------- #
# LLM (Groq primary, Cerebras fallback)
# --------------------------------------------------------------------------- #
#
# !! Model-ID correction, verified 2026-08-22 against console.groq.com/docs !!
# The original spec asked for "llama-3.1-8b-instant". Groq DECOMMISSIONED that
# model on 2026-08-16 (six days before this build) along with
# llama-3.3-70b-versatile. Requests to those IDs now return errors. Groq's own
# published migration targets are:
#     llama-3.1-8b-instant    -> openai/gpt-oss-20b
#     llama-3.3-70b-versatile -> openai/gpt-oss-120b
# gpt-oss-20b is also the faster of the two (~1000 tok/s vs ~500 tok/s), so it
# is both the correct and the lower-latency default for our TTFT target.

GROQ_API_KEY: Final[str | None] = _env_opt("GROQ_API_KEY")
GROQ_MODEL: Final[str] = _env_str("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_FALLBACK_MODEL: Final[str] = _env_str("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL: Final[str] = _env_str("GROQ_BASE_URL", "https://api.groq.com")

# Cerebras is an optional second provider. Its catalogue moves independently of
# Groq's, so verify the model ID against inference-docs.cerebras.ai before
# relying on it; the harness only dials Cerebras when a key is present.
CEREBRAS_API_KEY: Final[str | None] = _env_opt("CEREBRAS_API_KEY")
CEREBRAS_MODEL: Final[str] = _env_str("CEREBRAS_MODEL", "gpt-oss-120b")
CEREBRAS_BASE_URL: Final[str] = _env_str("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")

LLM_TEMPERATURE: Final[float] = _env_float("LLM_TEMPERATURE", 0.15)
LLM_MAX_TOKENS: Final[int] = _env_int("LLM_MAX_TOKENS", 420)
LLM_TIMEOUT_S: Final[float] = _env_float("LLM_TIMEOUT_S", 20.0)
LLM_MAX_RETRIES: Final[int] = _env_int("LLM_MAX_RETRIES", 2)
LLM_BACKOFF_BASE_S: Final[float] = _env_float("LLM_BACKOFF_BASE_S", 0.25)
LLM_BACKOFF_MAX_S: Final[float] = _env_float("LLM_BACKOFF_MAX_S", 4.0)

# Circuit breaker: after N consecutive provider failures, stop dialling for
# COOLDOWN seconds and fail fast to the fallback provider instead.
BREAKER_THRESHOLD: Final[int] = _env_int("BREAKER_THRESHOLD", 4)
BREAKER_COOLDOWN_S: Final[float] = _env_float("BREAKER_COOLDOWN_S", 20.0)


# --------------------------------------------------------------------------- #
# Speech-to-text
# --------------------------------------------------------------------------- #
#
# Sarvam facts verified against docs.sarvam.ai on 2026-08-22:
#   POST https://api.sarvam.ai/speech-to-text
#   auth header: "api-subscription-key"   (NOT Authorization: Bearer)
#   multipart fields: file, model, language_code, with_timestamps, input_audio_codec
#   models: saaras:v3 (default/recommended), saaras:v4 (latest)
#   `mode` (transcribe|translate|verbatim|translit|codemix) applies to v3 only.
#   response: {request_id, transcript, language_code, timestamps?, language_probability?}
# The spec's "sarvam-2.5"/"saaras:v1" ids are retired; v3/v4 are the live ones.
#
# ElevenLabs verified against elevenlabs.io/docs on 2026-08-22:
#   POST https://api.elevenlabs.io/v1/speech-to-text, multipart/form-data
#   header: "xi-api-key";  required field: model_id;  exactly one of file|source_url
#   current model shown throughout the docs: scribe_v2
#   response: {text, language_code, language_probability, words[], audio_duration_secs}
#
# Groq Whisper verified against console.groq.com/docs/speech-to-text on 2026-08-22:
#   POST https://api.groq.com/openai/v1/audio/transcriptions  (OpenAI-compatible)
#   multipart: model (required), file|url, language, prompt, temperature, response_format
#   models: whisper-large-v3-turbo (12% WER, $0.04/hr), whisper-large-v3 (10.3% WER)
#   Included because it is by far the lowest-latency option when you already
#   hold a Groq key - no extra vendor signup needed to demo the voice path.

STT_PROVIDER: Final[str] = _env_str("STT_PROVIDER", "sarvam").lower()
# Ordered failover chain. The first provider with a configured key wins; on a
# hard error the client walks down the list.
STT_FALLBACK_ORDER: Final[list[str]] = [
    p.strip().lower()
    for p in _env_str("STT_FALLBACK_ORDER", "sarvam,groq,elevenlabs").split(",")
    if p.strip()
]

SARVAM_API_KEY: Final[str | None] = _env_opt("SARVAM_API_KEY")
SARVAM_STT_URL: Final[str] = _env_str("SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text")
SARVAM_MODEL: Final[str] = _env_str("SARVAM_MODEL", "saaras:v3")
SARVAM_MODE: Final[str] = _env_str("SARVAM_MODE", "translate")
SARVAM_LANGUAGE: Final[str] = _env_str("SARVAM_LANGUAGE", "unknown")

ELEVENLABS_API_KEY: Final[str | None] = _env_opt("ELEVENLABS_API_KEY")
ELEVENLABS_STT_URL: Final[str] = _env_str(
    "ELEVENLABS_STT_URL", "https://api.elevenlabs.io/v1/speech-to-text"
)
ELEVENLABS_MODEL: Final[str] = _env_str("ELEVENLABS_MODEL", "scribe_v2")

GROQ_STT_URL: Final[str] = _env_str(
    "GROQ_STT_URL", "https://api.groq.com/openai/v1/audio/transcriptions"
)
GROQ_STT_MODEL: Final[str] = _env_str("GROQ_STT_MODEL", "whisper-large-v3-turbo")

STT_TIMEOUT_S: Final[float] = _env_float("STT_TIMEOUT_S", 15.0)
# Audio arriving over the websocket is raw 16-bit PCM at this rate; we wrap it
# in a WAV container in-process because both STT vendors want a real container.
AUDIO_SAMPLE_RATE: Final[int] = _env_int("AUDIO_SAMPLE_RATE", 16_000)
AUDIO_CHANNELS: Final[int] = 1
AUDIO_SAMPLE_WIDTH: Final[int] = 2
MAX_AUDIO_SECONDS: Final[float] = _env_float("MAX_AUDIO_SECONDS", 30.0)
MAX_AUDIO_BYTES: Final[int] = int(
    MAX_AUDIO_SECONDS * AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH
)


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

GROUNDING_THRESHOLD: Final[float] = _env_float("GROUNDING_THRESHOLD", 0.65)
# If the best retrieved chunk is less similar than this to the query, the corpus
# simply does not cover the question and we refuse rather than improvise.
CONTEXT_SUFFICIENCY_THRESHOLD: Final[float] = _env_float("CONTEXT_SUFFICIENCY_THRESHOLD", 0.32)
MIN_QUERY_CHARS: Final[int] = _env_int("MIN_QUERY_CHARS", 3)
MAX_QUERY_CHARS: Final[int] = _env_int("MAX_QUERY_CHARS", 512)
REFUSAL_MESSAGE: Final[str] = (
    "I cannot answer this based on the verified dataset."
)


# --------------------------------------------------------------------------- #
# Latency budget (milliseconds) - drives the UI gauges and benchmark verdicts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class LatencyBudget:
    embed_ms: float = 12.0
    retrieval_ms: float = 10.0
    guardrail_ms: float = 8.0
    ttft_ms: float = 80.0
    total_ms: float = 200.0  # transcript -> first generated token
    stages: tuple[str, ...] = field(
        default=("stt", "embed", "retrieval", "guardrail", "generation")
    )


BUDGET: Final[LatencyBudget] = LatencyBudget(
    embed_ms=_env_float("BUDGET_EMBED_MS", 12.0),
    retrieval_ms=_env_float("BUDGET_RETRIEVAL_MS", 10.0),
    guardrail_ms=_env_float("BUDGET_GUARDRAIL_MS", 8.0),
    ttft_ms=_env_float("BUDGET_TTFT_MS", 80.0),
    total_ms=_env_float("BUDGET_TOTAL_MS", 200.0),
)

PERCENTILES: Final[tuple[int, ...]] = (50, 70, 90, 95, 99, 100)
BENCHMARK_QUERIES: Final[int] = _env_int("BENCHMARK_QUERIES", 120)
BENCHMARK_CONCURRENCY: Final[int] = _env_int("BENCHMARK_CONCURRENCY", 4)
BENCHMARK_WARMUP: Final[int] = _env_int("BENCHMARK_WARMUP", 5)
BENCHMARK_REPORT_PATH: Final[Path] = DATA_DIR / "benchmark_report.json"


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #

HOST: Final[str] = _env_str("HOST", "0.0.0.0")
PORT: Final[int] = _env_int("PORT", 8000)
CORS_ORIGINS: Final[list[str]] = [
    o.strip()
    for o in _env_str("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]
LOG_LEVEL: Final[str] = _env_str("LOG_LEVEL", "INFO").upper()
# Build the index automatically on first boot if no table exists yet.
AUTO_INDEX_ON_STARTUP: Final[bool] = _env_bool("AUTO_INDEX_ON_STARTUP", True)


def ensure_dirs() -> None:
    """Create the writable directories the app assumes exist."""
    for path in (DATA_DIR, EMBED_CACHE_DIR, Path(LANCEDB_URI).parent):
        path.mkdir(parents=True, exist_ok=True)


def resolve_languages(requested: list[str] | None = None) -> list[ShardSpec]:
    """Validate language keys and return their shard specs.

    Raises a helpful error naming the valid keys rather than KeyError-ing deep
    inside the ingest loop.
    """
    keys = requested if requested else INGEST_LANGUAGES
    resolved: list[ShardSpec] = []
    for key in keys:
        spec = LANGUAGE_SHARDS.get(key.strip().lower())
        if spec is None:
            raise ValueError(
                f"Unknown language {key!r}. Valid keys: {', '.join(sorted(LANGUAGE_SHARDS))}"
            )
        resolved.append(spec)
    return resolved


def table_name(strategy: str) -> str:
    """LanceDB table name for a chunking strategy."""
    if strategy not in ALL_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Valid: {', '.join(ALL_STRATEGIES)}")
    return f"{TABLE_PREFIX}_{strategy}"


def stt_configured() -> bool:
    """True when at least one speech provider has a usable key."""
    return bool(SARVAM_API_KEY or ELEVENLABS_API_KEY or GROQ_API_KEY)


def stt_chain() -> list[str]:
    """Providers to try, in order, filtered to those that actually have keys."""
    keys = {
        "sarvam": SARVAM_API_KEY,
        "elevenlabs": ELEVENLABS_API_KEY,
        "groq": GROQ_API_KEY,
    }
    ordered = [STT_PROVIDER] + [p for p in STT_FALLBACK_ORDER if p != STT_PROVIDER]
    seen: set[str] = set()
    chain: list[str] = []
    for provider in ordered:
        if provider in seen or provider not in keys or not keys[provider]:
            continue
        seen.add(provider)
        chain.append(provider)
    return chain


def llm_configured() -> bool:
    """True when at least one inference provider has a usable key."""
    return bool(GROQ_API_KEY or CEREBRAS_API_KEY)
