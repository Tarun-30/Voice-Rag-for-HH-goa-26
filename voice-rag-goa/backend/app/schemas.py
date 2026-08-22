"""
Shared Pydantic contracts.

This module is the single source of truth for every payload that crosses a
process boundary: REST bodies, WebSocket frames, the LLM's structured output,
and the benchmark report on disk. The frontend's `src/lib/types.ts` mirrors
these names exactly - if you change one, change both.

Everything is `from __future__ import annotations`-safe and uses
`model_config = ConfigDict(extra="forbid")` on the LLM-facing models, because a
hallucinated extra field is a validation failure we *want* to see and retry.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import config

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

Strategy = Literal["semantic", "hierarchical", "sliding"]
RetrievalMode = Literal["hybrid", "dense", "sparse"]


class RetrievedChunk(BaseModel):
    """One retrieval hit, carrying enough provenance for the citation inspector.

    `text` is what matched the vector; `context_text` is what we actually feed
    the LLM. For the hierarchical strategy those differ - a 128-token child
    matches, but its 512-token parent is what gets read. That split is the whole
    point of Strategy B, so it is explicit in the wire format.
    """

    chunk_id: str
    text: str
    context_text: str
    strategy: Strategy
    score: float = Field(description="Final fused ranking score, higher is better.")
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rank: int = 0

    # Provenance / metadata (Strategy C embeds these into the searchable text).
    title: str = ""
    passage_id: str = ""
    query_id: int = -1
    language: str = ""
    domain: str = ""
    token_count: int = 0
    parent_id: str | None = None
    # MSMARCO-XI ships `is_selected` per candidate passage: the human-labelled
    # answer-bearing passage. We keep it so the benchmark can score real
    # retrieval quality (Recall@k / MRR), not just latency.
    is_gold: bool = False

    def as_context_block(self) -> str:
        """Render this chunk the way the LLM prompt shows it."""
        header = f"[{self.chunk_id}]"
        if self.title:
            header += f" {self.title}"
        return f"{header}\n{self.context_text.strip()}"


class ChunkStrategyStats(BaseModel):
    """Per-strategy index statistics, shown by ChunkVisualizer."""

    strategy: Strategy
    table: str
    rows: int = 0
    unique_passages: int = 0
    avg_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    p50_tokens: float = 0.0
    has_parents: bool = False
    has_ann_index: bool = False
    has_fts_index: bool = False
    build_seconds: float = 0.0
    description: str = ""


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

GuardrailDecision = Literal["allow", "block", "refuse"]
GuardrailCategory = Literal[
    "ok",
    "empty_query",
    "too_long",
    "prompt_injection",
    "unsafe_content",
    "off_topic",
    "insufficient_context",
    "ungrounded_answer",
    "gibberish",
]


class GuardrailCheck(BaseModel):
    """One individual guardrail probe and what it concluded."""

    name: str
    passed: bool
    score: float = 0.0
    threshold: float = 0.0
    detail: str = ""


class GuardrailVerdict(BaseModel):
    """Aggregate guardrail outcome for one turn."""

    decision: GuardrailDecision = "allow"
    category: GuardrailCategory = "ok"
    reason: str = ""
    risk_score: float = 0.0
    checks: list[GuardrailCheck] = Field(default_factory=list)
    sanitized_query: str = ""

    # Populated only by the post-generation grounding pass.
    grounding_score: float | None = None
    context_sufficiency: float | None = None
    grounded_sentences: int = 0
    total_sentences: int = 0
    unsupported_claims: list[str] = Field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"


# --------------------------------------------------------------------------- #
# LLM structured output (the harness contract)
# --------------------------------------------------------------------------- #


class AnswerResponse(BaseModel):
    """Strict schema the LLM must emit. Extra keys are a retryable error."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    thought_process: str = Field(
        default="",
        max_length=1200,
        description="Brief reasoning over the supplied context. Not shown to the user by default.",
    )
    answer: str = Field(
        description="The user-facing answer, grounded strictly in the provided context.",
        max_length=4000,
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Self-reported confidence in [0,1]."
    )
    cited_chunk_ids: list[str] = Field(
        default_factory=list,
        description="IDs of context chunks that support the answer.",
    )
    is_grounded: bool = Field(
        default=True,
        description="False when the context does not contain the answer.",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: object) -> float:
        """Models love to emit '0.9', '90%', or 90. Normalise all three."""
        if isinstance(value, str):
            cleaned = value.strip().rstrip("%")
            try:
                value = float(cleaned)
            except ValueError:
                return 0.5
            if "%" in str(value) or value > 1.0:
                value = value / 100.0
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            number = float(value)
            if number > 1.0:
                number = number / 100.0 if number <= 100.0 else 1.0
            return max(0.0, min(1.0, number))
        return 0.5

    @field_validator("cited_chunk_ids", mode="before")
    @classmethod
    def _coerce_citations(cls, value: object) -> list[str]:
        """Accept a comma-joined string or a list of ints/strings."""
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip(" []") for part in value.split(",") if part.strip(" []")]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip(" []") for item in value if str(item).strip(" []")]
        return [str(value)]

    @field_validator("is_grounded", mode="before")
    @classmethod
    def _coerce_bool(cls, value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1", "y", "grounded"}
        return bool(value)


ANSWER_JSON_SCHEMA: dict[str, object] = AnswerResponse.model_json_schema()


# --------------------------------------------------------------------------- #
# Timings
# --------------------------------------------------------------------------- #


class StageTiming(BaseModel):
    """A single pipeline stage's measured duration against its budget."""

    name: str
    ms: float
    budget_ms: float = 0.0

    @property
    def within_budget(self) -> bool:
        return self.budget_ms <= 0 or self.ms <= self.budget_ms


class PipelineTimings(BaseModel):
    """Full latency breakdown for one turn, in milliseconds.

    `total_e2e_ms` is wall-clock from request receipt to the final token.
    `query_to_first_token_ms` is the number the 200 ms target refers to:
    transcript in hand -> first generated token out. STT is excluded from it
    because network round-trips to a third-party ASR vendor are not something
    the retrieval pipeline can be held to.
    """

    stt_ms: float = 0.0
    embed_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    guardrail_ms: float = 0.0
    ttft_ms: float = 0.0
    generation_ms: float = 0.0
    grounding_ms: float = 0.0
    total_e2e_ms: float = 0.0
    query_to_first_token_ms: float = 0.0
    retrieval_to_generation_ms: float = 0.0
    tokens_out: int = 0
    tokens_per_second: float = 0.0
    provider: str = ""
    model: str = ""
    attempts: int = 1
    cache_hit: bool = False

    def stages(self) -> list[StageTiming]:
        budget = config.BUDGET
        return [
            StageTiming(name="stt", ms=self.stt_ms, budget_ms=0.0),
            StageTiming(name="embed", ms=self.embed_ms, budget_ms=budget.embed_ms),
            StageTiming(name="retrieval", ms=self.retrieval_ms, budget_ms=budget.retrieval_ms),
            StageTiming(name="guardrail", ms=self.guardrail_ms, budget_ms=budget.guardrail_ms),
            StageTiming(name="generation", ms=self.ttft_ms, budget_ms=budget.ttft_ms),
        ]

    @property
    def within_target(self) -> bool:
        return self.query_to_first_token_ms <= config.BUDGET.total_ms


# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #


class QueryRequest(BaseModel):
    """POST /api/query body."""

    query: str = Field(min_length=1, max_length=config.MAX_QUERY_CHARS)
    strategy: Strategy | None = None
    mode: RetrievalMode = "hybrid"
    top_k: int = Field(default=config.RETRIEVE_TOP_K, ge=1, le=20)
    include_thought_process: bool = False
    language: str | None = None


class QueryResponse(BaseModel):
    """POST /api/query response - the non-streaming path used by benchmarks."""

    query: str
    transcript: str = ""
    answer: str
    refused: bool = False
    confidence: float = 0.0
    thought_process: str = ""
    strategy: Strategy
    mode: RetrievalMode
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    guardrail: GuardrailVerdict
    timings: PipelineTimings
    request_id: str = ""


class HealthResponse(BaseModel):
    """GET /api/health - what the UI polls to decide which badges to light up."""

    status: Literal["ok", "degraded", "initialising"]
    version: str
    dataset: str
    indexed_strategies: list[Strategy] = Field(default_factory=list)
    default_strategy: Strategy
    total_chunks: int = 0
    languages: list[str] = Field(default_factory=list)
    embed_model: str = ""
    embed_dim: int = 0
    llm_provider: str = "offline"
    llm_model: str = ""
    stt_provider: str = "offline"
    stt_model: str = ""
    offline_mode: bool = False
    warnings: list[str] = Field(default_factory=list)


class TranscriptionResponse(BaseModel):
    """POST /api/transcribe response."""

    transcript: str
    language_code: str = ""
    language_probability: float = 0.0
    provider: str = ""
    model: str = ""
    stt_ms: float = 0.0
    audio_seconds: float = 0.0
    request_id: str = ""


# --------------------------------------------------------------------------- #
# Benchmark / analytics
# --------------------------------------------------------------------------- #


class MetricSummary(BaseModel):
    """Percentile summary of one measured metric across a benchmark run."""

    metric: str
    unit: Literal["ms", "ratio", "count", "tok/s"] = "ms"
    samples: int = 0
    mean: float = 0.0
    stdev: float = 0.0
    min: float = 0.0
    p50: float = 0.0
    p70: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    p100: float = 0.0
    budget_ms: float = 0.0
    within_budget_ratio: float = 0.0


class RetrievalQuality(BaseModel):
    """Real retrieval accuracy, scored against MSMARCO-XI `is_selected` labels."""

    queries_scored: int = 0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    hit_rate: float = 0.0


class StrategyBenchmark(BaseModel):
    """One (strategy, mode) cell of the benchmark grid."""

    strategy: Strategy
    mode: RetrievalMode
    queries: int = 0
    errors: int = 0
    refusals: int = 0
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    quality: RetrievalQuality = Field(default_factory=RetrievalQuality)


class BenchmarkReport(BaseModel):
    """The artefact written to data/benchmark_report.json and served by /api/analytics."""

    generated_at: str
    dataset: str
    languages: list[str] = Field(default_factory=list)
    embed_model: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    total_queries: int = 0
    concurrency: int = 1
    wall_seconds: float = 0.0
    llm_enabled: bool = True
    percentiles: list[int] = Field(default_factory=lambda: list(config.PERCENTILES))
    budget: dict[str, float] = Field(default_factory=dict)
    overall: dict[str, MetricSummary] = Field(default_factory=dict)
    per_strategy: list[StrategyBenchmark] = Field(default_factory=list)
    chunk_stats: list[ChunkStrategyStats] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class AnalyticsResponse(BaseModel):
    """GET /api/analytics.

    Carries two independent things, and the distinction matters when reading the
    dashboard:

    * `report` - the offline benchmark artefact (120+ queries, every strategy,
      real percentiles). Absent until `POST /api/benchmark` has been run once.
    * `live` - percentiles over the requests this server process has actually
      served, from a rolling in-memory window. Always available, and it is the
      honest answer to "how fast is it *right now*".
    """

    available: bool = False
    report: BenchmarkReport | None = None
    chunk_stats: list[ChunkStrategyStats] = Field(default_factory=list)
    live_requests: int = 0
    live: dict[str, MetricSummary] = Field(default_factory=dict)
    budget: dict[str, float] = Field(default_factory=dict)
    generated_at: str = ""
    benchmark_running: bool = False


class BenchmarkRequest(BaseModel):
    """POST /api/benchmark body. All fields optional - defaults come from config."""

    queries: int = Field(default=config.BENCHMARK_QUERIES, ge=1, le=2000)
    concurrency: int = Field(default=config.BENCHMARK_CONCURRENCY, ge=1, le=32)
    strategies: list[Strategy] | None = None
    modes: list[RetrievalMode] | None = None
    include_llm: bool = True


# --------------------------------------------------------------------------- #
# WebSocket protocol
# --------------------------------------------------------------------------- #
#
# Every frame the server sends is one of the models below, discriminated on
# `type`. The client switch in VoiceRecorder.tsx / page.tsx must stay exhaustive.


class WSBase(BaseModel):
    request_id: str = ""
    t_ms: float = Field(default=0.0, description="ms since the turn started")


class WSReady(WSBase):
    type: Literal["ready"] = "ready"
    sample_rate: int = config.AUDIO_SAMPLE_RATE
    strategy: Strategy = config.DEFAULT_STRATEGY  # type: ignore[assignment]
    offline_mode: bool = False


class WSStage(WSBase):
    """Stage lifecycle event that drives the Live Pipeline Breakdown timers."""

    type: Literal["stage"] = "stage"
    stage: Literal["stt", "embed", "retrieval", "guardrail", "generation", "grounding"]
    status: Literal["start", "done", "error", "skipped"]
    ms: float = 0.0
    budget_ms: float = 0.0
    detail: str = ""


class WSTranscript(WSBase):
    type: Literal["transcript"] = "transcript"
    text: str
    is_final: bool = True
    language_code: str = ""
    language_probability: float = 0.0
    provider: str = ""
    audio_seconds: float = 0.0


class WSChunks(WSBase):
    type: Literal["chunks"] = "chunks"
    strategy: Strategy
    mode: RetrievalMode
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class WSGuardrail(WSBase):
    type: Literal["guardrail"] = "guardrail"
    verdict: GuardrailVerdict
    phase: Literal["input", "context", "output"] = "input"


class WSToken(WSBase):
    type: Literal["token"] = "token"
    text: str
    index: int = 0
    is_first: bool = False


class WSDone(WSBase):
    type: Literal["done"] = "done"
    answer: str
    refused: bool = False
    confidence: float = 0.0
    thought_process: str = ""
    cited_chunk_ids: list[str] = Field(default_factory=list)
    guardrail: GuardrailVerdict
    timings: PipelineTimings


class WSError(WSBase):
    type: Literal["error"] = "error"
    message: str
    fatal: bool = False
    code: str = "internal_error"


ServerEvent = Annotated[
    WSReady | WSStage | WSTranscript | WSChunks | WSGuardrail | WSToken | WSDone | WSError,
    Field(discriminator="type"),
]


class WSClientConfig(BaseModel):
    """Client -> server control frame (sent as JSON text on the audio socket)."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["config", "start", "end", "text", "cancel"]
    strategy: Strategy | None = None
    mode: RetrievalMode | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    language: str | None = None
    sample_rate: int | None = None
    include_thought_process: bool | None = None
    # Present for type == "text": lets the UI run the pipeline without a mic.
    text: str | None = None
