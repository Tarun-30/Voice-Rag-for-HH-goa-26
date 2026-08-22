"use client";

/**
 * LatencyDashboard - the telemetry surface.
 *
 * Live pipeline breakdown: per-stage timers for the current turn, each drawn
 * against its budget (green within, pink over). The hero is
 * query -> first token, the sub-200ms target the whole system is tuned around.
 *
 * Aggregate telemetry: P50 / P70 / P100 gauges over either the live rolling
 * window (once any turns have run this session) or the last saved benchmark
 * report. A button kicks off a fresh benchmark run on the backend.
 */

import { Activity, Gauge, Loader2, Play, Timer, Zap } from "lucide-react";

import type {
  AnalyticsResponse,
  MetricSummary,
  PipelineStage,
  PipelineTimings,
} from "@/lib/types";
import type { StageState } from "@/lib/useVoicePipeline";
import { STAGE_ORDER } from "@/lib/useVoicePipeline";

interface LatencyDashboardProps {
  stages: Record<PipelineStage, StageState>;
  timings: PipelineTimings | null;
  analytics: AnalyticsResponse | null;
  benchmarkRunning: boolean;
  onBenchmark: () => void;
}

const STAGE_META: Record<PipelineStage, { label: string; defaultBudget: number }> = {
  stt: { label: "Speech-to-text", defaultBudget: 400 },
  embed: { label: "Embed query", defaultBudget: 25 },
  retrieval: { label: "Hybrid retrieval", defaultBudget: 40 },
  guardrail: { label: "Guardrails", defaultBudget: 30 },
  generation: { label: "First token", defaultBudget: 200 },
  grounding: { label: "Grounding check", defaultBudget: 40 },
};

const GAUGE_METRICS: { key: string; label: string; budget: number }[] = [
  { key: "query_to_first_token_ms", label: "Query → first token", budget: 200 },
  { key: "retrieval_ms", label: "Retrieval", budget: 40 },
  { key: "embed_ms", label: "Embed", budget: 25 },
  { key: "total_e2e_ms", label: "End-to-end", budget: 1500 },
];

function fmtMs(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  if (value >= 100) return `${Math.round(value)}ms`;
  return `${value.toFixed(1)}ms`;
}

export default function LatencyDashboard({
  stages,
  timings,
  analytics,
  benchmarkRunning,
  onBenchmark,
}: LatencyDashboardProps) {
  const liveN = analytics?.live_requests ?? 0;
  const useLive = liveN > 0 && analytics?.live;
  const metrics: Record<string, MetricSummary> | null = useLive
    ? analytics!.live
    : (analytics?.report?.overall ?? null);
  const sourceLabel = useLive
    ? `live · ${liveN} turn${liveN === 1 ? "" : "s"}`
    : analytics?.report
      ? `benchmark · ${analytics.report.total_queries} queries`
      : "no data yet";

  const budget = analytics?.budget ?? {};
  const heroTtft = timings?.query_to_first_token_ms ?? 0;

  return (
    <section className="glass flex h-full flex-col gap-4 p-5">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-2 text-lg font-bold text-cream">
          <Gauge className="h-5 w-5 text-gold" aria-hidden />
          Latency telemetry
        </h2>
        <button
          type="button"
          onClick={onBenchmark}
          disabled={benchmarkRunning}
          className="focus-ring inline-flex items-center gap-1.5 rounded-full border border-pink/50 bg-pink/12 px-3 py-1 text-xs font-semibold text-pink transition hover:bg-pink/20 disabled:opacity-60"
        >
          {benchmarkRunning ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden /> Benchmarking…
            </>
          ) : (
            <>
              <Play className="h-3.5 w-3.5" aria-hidden /> Run benchmark
            </>
          )}
        </button>
      </header>

      {/* Live pipeline breakdown */}
      <div className="glass-tight p-3">
        <div className="mb-2 flex items-center justify-between">
          <p className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-cream/60">
            <Activity className="h-3.5 w-3.5" aria-hidden /> Live pipeline
          </p>
          {timings && (
            <p className="text-xs text-cream/60">
              total{" "}
              <span className="font-mono text-cream/85">{fmtMs(timings.total_e2e_ms)}</span>
            </p>
          )}
        </div>
        <div className="space-y-1.5">
          {STAGE_ORDER.map((key) => (
            <StageBar key={key} stage={key} state={stages[key]} />
          ))}
        </div>
      </div>

      {/* Hero: query -> first token */}
      <div className="relative overflow-hidden rounded-2xl border border-gold/40 bg-gradient-to-br from-gold/12 to-pink/8 p-4">
        <div className="flex items-end justify-between">
          <div>
            <p className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-gold/90">
              <Zap className="h-3.5 w-3.5" aria-hidden /> Query → first token
            </p>
            <p className="mt-1 font-mono text-4xl font-bold text-cream">
              {heroTtft > 0 ? fmtMs(heroTtft) : "—"}
            </p>
          </div>
          <span
            className={`chip ${
              heroTtft > 0 && heroTtft <= 200
                ? "border-emerald-300/50 bg-emerald-400/12 text-emerald-200"
                : heroTtft > 200
                  ? "border-pink/50 bg-pink/12 text-pink"
                  : "text-cream/50"
            }`}
          >
            <Timer className="h-3 w-3" aria-hidden /> budget 200ms
          </span>
        </div>
      </div>

      {/* Percentile gauges */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mb-2 flex items-center justify-between">
          <p className="text-xs font-semibold uppercase tracking-wide text-cream/50">
            P50 / P70 / P100
          </p>
          <span className="chip text-[10px]">{sourceLabel}</span>
        </div>
        {metrics ? (
          <div className="grid gap-2.5 sm:grid-cols-2">
            {GAUGE_METRICS.map(({ key, label, budget: fallback }) => (
              <PercentileGauge
                key={key}
                label={label}
                summary={metrics[key]}
                budget={budget[key] ?? fallback}
              />
            ))}
          </div>
        ) : (
          <div className="grid h-24 place-items-center rounded-xl border border-dashed border-cream/15 text-center text-sm text-cream/40">
            Run a query or a benchmark to populate percentiles.
          </div>
        )}
      </div>
    </section>
  );
}

function StageBar({ stage, state }: { stage: PipelineStage; state: StageState }) {
  const meta = STAGE_META[stage];
  const budget = state.budget_ms || meta.defaultBudget;
  const ran = state.status === "done" || state.status === "error";
  const over = ran && state.ms > budget;
  const width = ran ? Math.max(3, Math.min(100, (state.ms / budget) * 100)) : 0;

  const dotColor =
    state.status === "done"
      ? over
        ? "bg-pink"
        : "bg-emerald-400"
      : state.status === "error"
        ? "bg-pink"
        : state.status === "skipped"
          ? "bg-cream/25"
          : state.status === "start"
            ? "bg-gold animate-pulse"
            : "bg-cream/20";

  return (
    <div className="flex items-center gap-2 text-xs">
      <span className={`h-2 w-2 shrink-0 rounded-full ${dotColor}`} aria-hidden />
      <span className="w-28 shrink-0 truncate text-cream/70">{meta.label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-cream/10">
        <div
          className={`h-full rounded-full transition-[width] duration-300 ${
            over ? "bg-pink" : "bg-emerald-400/80"
          }`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span
        className={`w-14 shrink-0 text-right font-mono ${
          state.status === "skipped" ? "text-cream/30" : over ? "text-pink" : "text-cream/70"
        }`}
      >
        {state.status === "skipped"
          ? "skip"
          : state.status === "start"
            ? "…"
            : ran
              ? fmtMs(state.ms)
              : "—"}
      </span>
    </div>
  );
}

function PercentileGauge({
  label,
  summary,
  budget,
}: {
  label: string;
  summary: MetricSummary | undefined;
  budget: number;
}) {
  if (!summary || summary.samples === 0) {
    return (
      <div className="glass-tight p-3 opacity-60">
        <p className="text-[11px] font-semibold uppercase tracking-wide text-cream/50">{label}</p>
        <p className="mt-2 font-mono text-lg text-cream/40">—</p>
      </div>
    );
  }
  const withinBudget = summary.p50 <= budget;
  const fill = Math.max(2, Math.min(100, (summary.p50 / (budget * 1.5)) * 100));
  const p100Mark = Math.min(100, (summary.p100 / (budget * 1.5)) * 100);

  return (
    <div className="glass-tight p-3">
      <div className="flex items-center justify-between">
        <p className="text-[11px] font-semibold uppercase tracking-wide text-cream/60">{label}</p>
        <span
          className={`text-[10px] font-semibold ${withinBudget ? "text-emerald-300" : "text-pink"}`}
        >
          {withinBudget ? "PASS" : "OVER"}
        </span>
      </div>
      <p className="mt-1 font-mono text-2xl font-bold text-cream">{fmtMs(summary.p50)}</p>
      <div className="relative mt-2 h-1.5 overflow-hidden rounded-full bg-cream/10">
        <div
          className={`h-full rounded-full ${withinBudget ? "bg-emerald-400/80" : "bg-pink"}`}
          style={{ width: `${fill}%` }}
        />
        {/* budget marker at 2/3 (budget = budget*1.5 scale) */}
        <div
          className="absolute top-0 h-full w-px bg-gold"
          style={{ left: "66.6%" }}
          title={`budget ${fmtMs(budget)}`}
        />
        <div
          className="absolute top-0 h-full w-px bg-cream/60"
          style={{ left: `${p100Mark}%` }}
          title={`P100 ${fmtMs(summary.p100)}`}
        />
      </div>
      <div className="mt-1.5 flex justify-between font-mono text-[10px] text-cream/50">
        <span>P50 {fmtMs(summary.p50)}</span>
        <span>P70 {fmtMs(summary.p70)}</span>
        <span>P100 {fmtMs(summary.p100)}</span>
      </div>
    </div>
  );
}
