/**
 * TypeScript mirror of backend/app/schemas.py.
 *
 * These types are the wire contract between the FastAPI backend and this UI.
 * If a field changes in schemas.py, change it here too - the names are
 * deliberately identical so a diff is obvious.
 */

export type Strategy = "semantic" | "hierarchical" | "sliding";
export type RetrievalMode = "hybrid" | "dense" | "sparse";

export type GuardrailDecision = "allow" | "block" | "refuse";
export type GuardrailCategory =
  | "ok"
  | "empty_query"
  | "too_long"
  | "prompt_injection"
  | "unsafe_content"
  | "off_topic"
  | "insufficient_context"
  | "ungrounded_answer"
  | "gibberish";

export interface RetrievedChunk {
  chunk_id: string;
  text: string;
  context_text: string;
  strategy: Strategy;
  score: number;
  dense_score: number;
  sparse_score: number;
  rank: number;
  title: string;
  passage_id: string;
  query_id: number;
  language: string;
  domain: string;
  token_count: number;
  parent_id: string | null;
  is_gold: boolean;
}

export interface GuardrailCheck {
  name: string;
  passed: boolean;
  score: number;
  threshold: number;
  detail: string;
}

export interface GuardrailVerdict {
  decision: GuardrailDecision;
  category: GuardrailCategory;
  reason: string;
  risk_score: number;
  checks: GuardrailCheck[];
  sanitized_query: string;
  grounding_score: number | null;
  context_sufficiency: number | null;
  grounded_sentences: number;
  total_sentences: number;
  unsupported_claims: string[];
}

export interface PipelineTimings {
  stt_ms: number;
  embed_ms: number;
  retrieval_ms: number;
  rerank_ms: number;
  guardrail_ms: number;
  ttft_ms: number;
  generation_ms: number;
  grounding_ms: number;
  total_e2e_ms: number;
  query_to_first_token_ms: number;
  retrieval_to_generation_ms: number;
  tokens_out: number;
  tokens_per_second: number;
  provider: string;
  model: string;
  attempts: number;
  cache_hit: boolean;
}

export interface QueryResponse {
  query: string;
  transcript: string;
  answer: string;
  refused: boolean;
  confidence: number;
  thought_process: string;
  strategy: Strategy;
  mode: RetrievalMode;
  chunks: RetrievedChunk[];
  cited_chunk_ids: string[];
  guardrail: GuardrailVerdict;
  timings: PipelineTimings;
  request_id: string;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "initialising";
  version: string;
  dataset: string;
  indexed_strategies: Strategy[];
  default_strategy: Strategy;
  total_chunks: number;
  languages: string[];
  embed_model: string;
  embed_dim: number;
  llm_provider: string;
  llm_model: string;
  stt_provider: string;
  stt_model: string;
  offline_mode: boolean;
  warnings: string[];
}

export interface TranscriptionResponse {
  transcript: string;
  language_code: string;
  language_probability: number;
  provider: string;
  model: string;
  stt_ms: number;
  audio_seconds: number;
  request_id: string;
}

export interface MetricSummary {
  metric: string;
  unit: "ms" | "ratio" | "count" | "tok/s";
  samples: number;
  mean: number;
  stdev: number;
  min: number;
  p50: number;
  p70: number;
  p90: number;
  p95: number;
  p99: number;
  p100: number;
  budget_ms: number;
  within_budget_ratio: number;
}

export interface RetrievalQuality {
  queries_scored: number;
  recall_at_1: number;
  recall_at_3: number;
  recall_at_5: number;
  mrr_at_10: number;
  ndcg_at_5: number;
  hit_rate: number;
}

export interface ChunkStrategyStats {
  strategy: Strategy;
  table: string;
  rows: number;
  unique_passages: number;
  avg_tokens: number;
  min_tokens: number;
  max_tokens: number;
  p50_tokens: number;
  has_parents: boolean;
  has_ann_index: boolean;
  has_fts_index: boolean;
  build_seconds: number;
  description: string;
}

export interface StrategyBenchmark {
  strategy: Strategy;
  mode: RetrievalMode;
  queries: number;
  errors: number;
  refusals: number;
  metrics: Record<string, MetricSummary>;
  quality: RetrievalQuality;
}

export interface BenchmarkReport {
  generated_at: string;
  dataset: string;
  languages: string[];
  embed_model: string;
  llm_provider: string;
  llm_model: string;
  total_queries: number;
  concurrency: number;
  wall_seconds: number;
  llm_enabled: boolean;
  percentiles: number[];
  budget: Record<string, number>;
  overall: Record<string, MetricSummary>;
  per_strategy: StrategyBenchmark[];
  chunk_stats: ChunkStrategyStats[];
  notes: string[];
}

export interface AnalyticsResponse {
  available: boolean;
  report: BenchmarkReport | null;
  chunk_stats: ChunkStrategyStats[];
  live_requests: number;
  live: Record<string, MetricSummary>;
  budget: Record<string, number>;
  generated_at: string;
  benchmark_running: boolean;
}

// --------------------------------------------------------------------------- //
// WebSocket protocol (server -> client), discriminated on `type`.
// --------------------------------------------------------------------------- //

export interface WSBase {
  request_id: string;
  t_ms: number;
}

export type PipelineStage =
  | "stt"
  | "embed"
  | "retrieval"
  | "guardrail"
  | "generation"
  | "grounding";
export type StageStatus = "start" | "done" | "error" | "skipped";
export type GuardrailPhase = "input" | "context" | "output";

export interface WSReady extends WSBase {
  type: "ready";
  sample_rate: number;
  strategy: Strategy;
  offline_mode: boolean;
}

export interface WSStage extends WSBase {
  type: "stage";
  stage: PipelineStage;
  status: StageStatus;
  ms: number;
  budget_ms: number;
  detail: string;
}

export interface WSTranscript extends WSBase {
  type: "transcript";
  text: string;
  is_final: boolean;
  language_code: string;
  language_probability: number;
  provider: string;
  audio_seconds: number;
}

export interface WSChunks extends WSBase {
  type: "chunks";
  strategy: Strategy;
  mode: RetrievalMode;
  chunks: RetrievedChunk[];
}

export interface WSGuardrail extends WSBase {
  type: "guardrail";
  verdict: GuardrailVerdict;
  phase: GuardrailPhase;
}

export interface WSToken extends WSBase {
  type: "token";
  text: string;
  index: number;
  is_first: boolean;
}

export interface WSDone extends WSBase {
  type: "done";
  answer: string;
  refused: boolean;
  confidence: number;
  thought_process: string;
  cited_chunk_ids: string[];
  guardrail: GuardrailVerdict;
  timings: PipelineTimings;
}

export interface WSError extends WSBase {
  type: "error";
  message: string;
  fatal: boolean;
  code: string;
}

export type ServerEvent =
  | WSReady
  | WSStage
  | WSTranscript
  | WSChunks
  | WSGuardrail
  | WSToken
  | WSDone
  | WSError;

// Client -> server control frame.
export interface WSClientConfig {
  type: "config" | "start" | "end" | "text" | "cancel";
  strategy?: Strategy;
  mode?: RetrievalMode;
  top_k?: number;
  language?: string | null;
  sample_rate?: number;
  include_thought_process?: boolean;
  text?: string;
}

export interface QueryRequest {
  query: string;
  strategy?: Strategy | null;
  mode?: RetrievalMode;
  top_k?: number;
  include_thought_process?: boolean;
  language?: string | null;
}
