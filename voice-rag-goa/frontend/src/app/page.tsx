"use client";

/**
 * Home - the single-page console.
 *
 * Owns UI settings (strategy / mode / top-k / thought-process), drives the
 * voice pipeline via useVoicePipeline, polls health + analytics, and lays out
 * the four instrument panels around the streaming answer.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Brain,
  ChevronDown,
  Cpu,
  Database,
  Sparkles,
  WifiOff,
} from "lucide-react";

import ChunkVisualizer from "@/components/ChunkVisualizer";
import GroundingBadge from "@/components/GroundingBadge";
import LatencyDashboard from "@/components/LatencyDashboard";
import VoiceRecorder from "@/components/VoiceRecorder";
import { getAnalytics, getHealth, triggerBenchmark } from "@/lib/api";
import type {
  AnalyticsResponse,
  HealthResponse,
  RetrievalMode,
  Strategy,
} from "@/lib/types";
import {
  useVoicePipeline,
  type PipelineSettings,
} from "@/lib/useVoicePipeline";

const TOP_K_OPTIONS = [3, 5, 8, 10];

export default function Home() {
  const pipeline = useVoicePipeline();
  const { turn } = pipeline;

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [benchmarkPending, setBenchmarkPending] = useState(false);

  const [strategy, setStrategy] = useState<Strategy>("hierarchical");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [topK, setTopK] = useState(5);
  const [includeThought, setIncludeThought] = useState(false);
  const strategyLocked = useRef(false);

  const settings: PipelineSettings = {
    strategy,
    mode,
    topK,
    includeThoughtProcess: includeThought,
    language: null,
  };

  // --- polling: health + analytics --------------------------------------- //
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const [h, a] = await Promise.all([getHealth(), getAnalytics()]);
        if (!alive) return;
        setHealth(h);
        setAnalytics(a);
        if (!strategyLocked.current && h.default_strategy) {
          setStrategy(h.default_strategy);
          strategyLocked.current = true;
        }
      } catch {
        /* backend still warming up; keep last known state */
      }
    };
    void tick();
    const id = setInterval(tick, 4000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  // Refresh analytics immediately when a turn finishes (live percentiles move).
  const lastStatus = useRef<string>("");
  useEffect(() => {
    if (turn?.status === "done" && lastStatus.current !== "done") {
      getAnalytics().then(setAnalytics).catch(() => {});
    }
    lastStatus.current = turn?.status ?? "";
  }, [turn?.status]);

  const onBenchmark = useCallback(async () => {
    setBenchmarkPending(true);
    try {
      await triggerBenchmark({ queries: 40 });
    } catch {
      /* the analytics poll surfaces benchmark_running / errors */
    } finally {
      // Let the poll take over reporting the running state.
      setTimeout(() => setBenchmarkPending(false), 2000);
    }
  }, []);

  const benchmarkRunning = benchmarkPending || Boolean(analytics?.benchmark_running);

  const stats =
    analytics?.chunk_stats?.length
      ? analytics.chunk_stats
      : (analytics?.report?.chunk_stats ?? []);

  const handleStrategyChange = (next: Strategy) => {
    strategyLocked.current = true;
    setStrategy(next);
  };

  const offline = health?.offline_mode ?? pipeline.offlineMode;

  return (
    <div className="mx-auto flex min-h-screen max-w-[1500px] flex-col gap-5 px-4 py-5 sm:px-6">
      <SiteHeader
        health={health}
        offline={offline}
        topK={topK}
        setTopK={setTopK}
        includeThought={includeThought}
        setIncludeThought={setIncludeThought}
      />

      <main className="grid flex-1 gap-5 lg:grid-cols-12">
        {/* Left rail: input + guardrail */}
        <div className="flex flex-col gap-5 lg:col-span-4">
          <VoiceRecorder
            busy={pipeline.busy}
            connected={pipeline.connected}
            status={turn?.status ?? "idle"}
            transcript={turn?.transcript ?? ""}
            onAudio={(pcm) => pipeline.sendAudio(pcm, settings)}
            onText={(text) => pipeline.sendText(text, settings)}
          />
          <GroundingBadge
            guardrails={turn?.guardrails ?? {}}
            refused={turn?.refused ?? false}
            status={turn?.status ?? "idle"}
            confidence={turn?.confidence ?? 0}
          />
        </div>

        {/* Right: answer on top, instruments below */}
        <div className="flex flex-col gap-5 lg:col-span-8">
          <AnswerPanel turn={turn} />
          <div className="grid gap-5 xl:grid-cols-2">
            <ChunkVisualizer
              chunks={turn?.chunks ?? []}
              strategy={turn?.strategy ?? strategy}
              mode={turn?.mode ?? mode}
              citedChunkIds={turn?.citedChunkIds ?? []}
              stats={stats}
              disabled={pipeline.busy}
              onStrategyChange={handleStrategyChange}
              onModeChange={setMode}
            />
            <LatencyDashboard
              stages={
                turn?.stages ?? {
                  stt: blankStage(),
                  embed: blankStage(),
                  retrieval: blankStage(),
                  guardrail: blankStage(),
                  generation: blankStage(),
                  grounding: blankStage(),
                }
              }
              timings={turn?.timings ?? null}
              analytics={analytics}
              benchmarkRunning={benchmarkRunning}
              onBenchmark={onBenchmark}
            />
          </div>
        </div>
      </main>

      <footer className="pb-2 text-center text-xs text-cream/40">
        Voice RAG · Goa Hacker House — hybrid retrieval over{" "}
        {health?.dataset ?? "ai4bharat/MSMARCO-XI"} ·{" "}
        {health?.total_chunks ? `${health.total_chunks.toLocaleString()} chunks` : "indexing…"}
      </footer>
    </div>
  );
}

function blankStage() {
  return { status: "pending" as const, ms: 0, budget_ms: 0, detail: "" };
}

/* ------------------------------------------------------------------------- */
/* Header                                                                    */
/* ------------------------------------------------------------------------- */

function SiteHeader({
  health,
  offline,
  topK,
  setTopK,
  includeThought,
  setIncludeThought,
}: {
  health: HealthResponse | null;
  offline: boolean;
  topK: number;
  setTopK: (n: number) => void;
  includeThought: boolean;
  setIncludeThought: (b: boolean) => void;
}) {
  const status = health?.status ?? "initialising";
  const statusTone =
    status === "ok"
      ? "bg-emerald-400"
      : status === "degraded"
        ? "bg-gold"
        : "bg-cream/40 animate-pulse";

  return (
    <header className="glass flex flex-col gap-4 p-5">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className="grid h-12 w-12 place-items-center rounded-2xl bg-gradient-to-br from-gold to-pink text-ink shadow-[var(--shadow-glow-gold)]">
            <Sparkles className="h-6 w-6" aria-hidden />
          </span>
          <div>
            <h1 className="text-2xl font-bold leading-tight">
              <span className="gradient-text">Voice RAG</span>
              <span className="text-cream"> · Goa Hacker House</span>
            </h1>
            <p className="text-xs text-cream/60">
              Real-time voice → grounded answer · sub-200ms retrieval-to-generation
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="chip">
            <span className={`h-2 w-2 rounded-full ${statusTone}`} aria-hidden />
            {status}
          </span>
          {offline && (
            <span className="chip border-gold/40 bg-gold/10 text-gold">
              <WifiOff className="h-3 w-3" aria-hidden /> offline mode
            </span>
          )}
          <span className="chip" title="Embedding model">
            <Cpu className="h-3 w-3" aria-hidden />
            {health?.embed_model?.split("/").pop() ?? "bge-small"}
          </span>
          <span className="chip" title="LLM provider / model">
            <Brain className="h-3 w-3" aria-hidden />
            {health?.llm_provider ?? "offline"}
            {health?.llm_model ? ` · ${health.llm_model.split("/").pop()}` : ""}
          </span>
          <span className="chip" title="Dataset">
            <Database className="h-3 w-3" aria-hidden />
            {(health?.dataset ?? "MSMARCO-XI").split("/").pop()}
          </span>
        </div>
      </div>

      {/* Controls + warnings */}
      <div className="flex flex-wrap items-center gap-3 border-t border-cream/10 pt-3">
        <label className="flex items-center gap-2 text-xs text-cream/70">
          top-k
          <select
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="focus-ring rounded-lg border border-cream/15 bg-ink/50 px-2 py-1 text-cream"
          >
            {TOP_K_OPTIONS.map((k) => (
              <option key={k} value={k} className="bg-palm">
                {k}
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          onClick={() => setIncludeThought(!includeThought)}
          aria-pressed={includeThought}
          className={`focus-ring inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold transition ${
            includeThought
              ? "border-gold/60 bg-gold/15 text-gold"
              : "border-cream/15 bg-cream/5 text-cream/70"
          }`}
        >
          <Brain className="h-3.5 w-3.5" aria-hidden /> thought process
        </button>

        {health?.warnings && health.warnings.length > 0 && (
          <span className="ml-auto flex items-center gap-1.5 text-xs text-gold/80">
            <AlertTriangle className="h-3.5 w-3.5" aria-hidden />
            {health.warnings[0]}
          </span>
        )}
      </div>
    </header>
  );
}

/* ------------------------------------------------------------------------- */
/* Answer panel                                                              */
/* ------------------------------------------------------------------------- */

function AnswerPanel({ turn }: { turn: ReturnType<typeof useVoicePipeline>["turn"] }) {
  const [showThought, setShowThought] = useState(false);

  if (!turn) {
    return (
      <section className="glass grid min-h-56 place-items-center p-8 text-center">
        <div className="max-w-md space-y-2">
          <Sparkles className="mx-auto h-8 w-8 text-gold" aria-hidden />
          <h2 className="text-xl font-bold text-cream">Ask anything, out loud.</h2>
          <p className="text-sm text-cream/60">
            Speak or type a question. It&apos;s transcribed, matched against the
            MSMARCO-XI corpus with hybrid search, and answered with a grounded,
            streaming response — every stage timed live on the right.
          </p>
        </div>
      </section>
    );
  }

  const refused = turn.refused;
  const hasAnswer = turn.answer.length > 0;
  const t = turn.timings;

  return (
    <section className="glass flex flex-col gap-4 p-6">
      {turn.transcript && (
        <div className="flex items-start gap-2">
          <span className="chip shrink-0 border-pink/40 bg-pink/10 text-pink">you</span>
          <p className="text-sm text-cream/80">{turn.transcript}</p>
        </div>
      )}

      {turn.error ? (
        <div className="flex items-start gap-2 rounded-xl border border-pink/50 bg-pink/10 p-4 text-pink">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden />
          <div>
            <p className="font-semibold">Something went wrong</p>
            <p className="text-sm opacity-80">{turn.error}</p>
          </div>
        </div>
      ) : (
        <div
          className={`rounded-xl border p-4 ${
            refused
              ? "border-gold/40 bg-gold/8"
              : "border-cream/12 bg-cream/4"
          }`}
        >
          <p
            className={`whitespace-pre-wrap text-[15px] leading-relaxed text-cream ${
              turn.streaming && hasAnswer ? "caret" : ""
            }`}
          >
            {hasAnswer ? (
              turn.answer
            ) : turn.status === "streaming" || turn.status === "running" ? (
              <span className="text-cream/50">Generating grounded answer…</span>
            ) : turn.status === "transcribing" ? (
              <span className="text-cream/50">Transcribing your question…</span>
            ) : (
              <span className="text-cream/40">…</span>
            )}
          </p>
        </div>
      )}

      {/* Thought process */}
      {turn.thoughtProcess && (
        <div className="rounded-xl border border-cream/12 bg-ink/30">
          <button
            type="button"
            onClick={() => setShowThought(!showThought)}
            className="focus-ring flex w-full items-center justify-between p-3 text-left text-xs font-semibold uppercase tracking-wide text-cream/60"
            aria-expanded={showThought}
          >
            <span className="flex items-center gap-1.5">
              <Brain className="h-3.5 w-3.5" aria-hidden /> Reasoning
            </span>
            <ChevronDown
              className={`h-4 w-4 transition-transform ${showThought ? "rotate-180" : ""}`}
              aria-hidden
            />
          </button>
          {showThought && (
            <p className="whitespace-pre-wrap border-t border-cream/10 p-3 text-xs leading-relaxed text-cream/70">
              {turn.thoughtProcess}
            </p>
          )}
        </div>
      )}

      {/* Footer meta */}
      {t && (t.provider || t.tokens_out > 0) && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-cream/10 pt-3 text-[11px] text-cream/50">
          {t.provider && (
            <span>
              {t.provider}
              {t.model ? ` · ${t.model.split("/").pop()}` : ""}
            </span>
          )}
          {t.tokens_out > 0 && <span>{t.tokens_out} tokens</span>}
          {t.tokens_per_second > 0 && <span>{t.tokens_per_second.toFixed(0)} tok/s</span>}
          {t.attempts > 1 && <span className="text-gold">{t.attempts} attempts</span>}
          {t.cache_hit && <span className="text-emerald-300">cache hit</span>}
        </div>
      )}
    </section>
  );
}
