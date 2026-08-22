"use client";

/**
 * ChunkVisualizer - the retrieval inspector.
 *
 * Top: segmented controls to switch chunking strategy (semantic / hierarchical
 * / sliding) and retrieval mode (hybrid / dense / sparse) for the next query,
 * with a one-line description + corpus stats for the active strategy.
 *
 * Body: the ranked chunks from the last turn. Each row shows the fused score
 * plus a dense/sparse score split (bars scaled to the max in the set so they're
 * comparable), a gold-label marker when the chunk is a MSMARCO-XI relevant
 * passage, and a "cited" marker when the LLM referenced it. Clicking a row opens
 * the citation inspector: full chunk text and, for hierarchical chunks, the
 * parent context the model actually reads.
 */

import { useMemo, useState } from "react";
import { Boxes, Layers, MoveHorizontal, Quote, Star } from "lucide-react";

import type {
  ChunkStrategyStats,
  RetrievalMode,
  RetrievedChunk,
  Strategy,
} from "@/lib/types";

interface ChunkVisualizerProps {
  chunks: RetrievedChunk[];
  strategy: Strategy;
  mode: RetrievalMode;
  citedChunkIds: string[];
  stats: ChunkStrategyStats[];
  disabled?: boolean;
  onStrategyChange: (strategy: Strategy) => void;
  onModeChange: (mode: RetrievalMode) => void;
}

const STRATEGIES: { key: Strategy; label: string; icon: typeof Boxes; blurb: string }[] = [
  { key: "semantic", label: "Semantic", icon: Boxes, blurb: "Cosine-distance breakpoints" },
  { key: "hierarchical", label: "Hierarchical", icon: Layers, blurb: "Small children, parent context" },
  { key: "sliding", label: "Sliding", icon: MoveHorizontal, blurb: "256 tok · 25% overlap" },
];

const MODES: { key: RetrievalMode; label: string; blurb: string }[] = [
  { key: "hybrid", label: "Hybrid", blurb: "Dense + BM25 fused (RRF)" },
  { key: "dense", label: "Dense", blurb: "Vector cosine only" },
  { key: "sparse", label: "Sparse", blurb: "BM25 lexical only" },
];

function scoreColor(mode: RetrievalMode): string {
  if (mode === "dense") return "bg-gold";
  if (mode === "sparse") return "bg-pink";
  return "bg-emerald-400";
}

export default function ChunkVisualizer({
  chunks,
  strategy,
  mode,
  citedChunkIds,
  stats,
  disabled = false,
  onStrategyChange,
  onModeChange,
}: ChunkVisualizerProps) {
  const [openId, setOpenId] = useState<string | null>(null);

  const activeStat = stats.find((s) => s.strategy === strategy);
  const cited = useMemo(() => new Set(citedChunkIds), [citedChunkIds]);

  const maxima = useMemo(() => {
    let score = 1e-6;
    let dense = 1e-6;
    let sparse = 1e-6;
    for (const chunk of chunks) {
      score = Math.max(score, Math.abs(chunk.score));
      dense = Math.max(dense, Math.abs(chunk.dense_score));
      sparse = Math.max(sparse, Math.abs(chunk.sparse_score));
    }
    return { score, dense, sparse };
  }, [chunks]);

  const goldHits = chunks.filter((c) => c.is_gold).length;

  return (
    <section className="glass flex h-full flex-col gap-4 p-5">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-2 text-lg font-bold text-cream">
          <Quote className="h-5 w-5 text-gold" aria-hidden />
          Retrieval
        </h2>
        <div className="flex items-center gap-2 text-xs text-cream/60">
          {chunks.length > 0 && (
            <span className="chip">
              {chunks.length} chunks
              {goldHits > 0 && (
                <span className="ml-1 text-gold">· {goldHits} gold</span>
              )}
            </span>
          )}
        </div>
      </header>

      {/* Strategy switcher */}
      <div className="space-y-2">
        <p className="text-[11px] font-semibold uppercase tracking-wide text-cream/50">
          Chunking strategy
        </p>
        <div className="grid grid-cols-3 gap-2">
          {STRATEGIES.map(({ key, label, icon: Icon }) => {
            const active = key === strategy;
            return (
              <button
                key={key}
                type="button"
                disabled={disabled}
                onClick={() => onStrategyChange(key)}
                className={`focus-ring flex flex-col items-center gap-1 rounded-xl border px-2 py-2.5 text-xs font-semibold transition disabled:opacity-50 ${
                  active
                    ? "border-gold/70 bg-gold/15 text-gold shadow-[var(--shadow-glow-gold)]"
                    : "border-cream/15 bg-cream/5 text-cream/70 hover:border-cream/30"
                }`}
                aria-pressed={active}
              >
                <Icon className="h-4 w-4" aria-hidden />
                {label}
              </button>
            );
          })}
        </div>
        {activeStat && (
          <p className="text-[11px] text-cream/55">
            {activeStat.description} ·{" "}
            <span className="text-cream/75">
              {activeStat.rows.toLocaleString()} chunks
            </span>
            , avg <span className="text-cream/75">{Math.round(activeStat.avg_tokens)}</span> tok
            {activeStat.has_parents ? " · parent context" : ""}
          </p>
        )}
      </div>

      {/* Mode switcher */}
      <div className="space-y-2">
        <p className="text-[11px] font-semibold uppercase tracking-wide text-cream/50">
          Retrieval mode
        </p>
        <div className="grid grid-cols-3 gap-2">
          {MODES.map(({ key, label, blurb }) => {
            const active = key === mode;
            return (
              <button
                key={key}
                type="button"
                disabled={disabled}
                onClick={() => onModeChange(key)}
                title={blurb}
                className={`focus-ring rounded-xl border px-2 py-2 text-xs font-semibold transition disabled:opacity-50 ${
                  active
                    ? "border-pink/60 bg-pink/15 text-pink"
                    : "border-cream/15 bg-cream/5 text-cream/70 hover:border-cream/30"
                }`}
                aria-pressed={active}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Chunk list */}
      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
        {chunks.length === 0 ? (
          <div className="grid h-full min-h-32 place-items-center rounded-xl border border-dashed border-cream/15 text-center text-sm text-cream/40">
            Retrieved passages appear here after a query.
          </div>
        ) : (
          chunks.map((chunk) => {
            const open = openId === chunk.chunk_id;
            const isCited = cited.has(chunk.chunk_id);
            return (
              <article
                key={chunk.chunk_id}
                className={`animate-fade-in-up rounded-xl border transition ${
                  isCited
                    ? "border-gold/50 bg-gold/8"
                    : "border-cream/12 bg-cream/4 hover:border-cream/25"
                }`}
              >
                <button
                  type="button"
                  onClick={() => setOpenId(open ? null : chunk.chunk_id)}
                  className="focus-ring flex w-full items-start gap-3 p-3 text-left"
                  aria-expanded={open}
                >
                  <span className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full bg-palm-600 text-[11px] font-bold text-gold">
                    {chunk.rank}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <p className="truncate text-sm font-semibold text-cream">
                        {chunk.title || chunk.passage_id || chunk.chunk_id}
                      </p>
                      {chunk.is_gold && (
                        <span title="Gold-labelled relevant passage (MSMARCO-XI)">
                          <Star className="h-3.5 w-3.5 shrink-0 fill-gold text-gold" aria-hidden />
                        </span>
                      )}
                      {isCited && (
                        <span className="chip border-gold/40 bg-gold/15 !py-0 text-[10px] text-gold">
                          cited
                        </span>
                      )}
                    </div>
                    <p className="mt-0.5 line-clamp-2 text-xs text-cream/60">{chunk.text}</p>

                    {/* score bars */}
                    <div className="mt-2 space-y-1">
                      <ScoreRow
                        label={mode === "hybrid" ? "fused" : mode}
                        value={chunk.score}
                        width={Math.abs(chunk.score) / maxima.score}
                        color={scoreColor(mode)}
                      />
                      {mode !== "sparse" && (
                        <ScoreRow
                          label="dense"
                          value={chunk.dense_score}
                          width={Math.abs(chunk.dense_score) / maxima.dense}
                          color="bg-gold/70"
                        />
                      )}
                      {mode !== "dense" && (
                        <ScoreRow
                          label="sparse"
                          value={chunk.sparse_score}
                          width={Math.abs(chunk.sparse_score) / maxima.sparse}
                          color="bg-pink/70"
                        />
                      )}
                    </div>
                  </div>
                </button>

                {open && (
                  <div className="space-y-3 border-t border-cream/12 px-3 pb-3 pt-3">
                    <div className="flex flex-wrap gap-2 text-[11px] text-cream/55">
                      <span className="chip">{chunk.token_count} tok</span>
                      <span className="chip">{chunk.strategy}</span>
                      {chunk.language && <span className="chip">{chunk.language}</span>}
                      {chunk.domain && <span className="chip">{chunk.domain}</span>}
                      <span className="chip">id {chunk.chunk_id}</span>
                    </div>
                    <div>
                      <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-cream/45">
                        Chunk text
                      </p>
                      <p className="whitespace-pre-wrap text-xs leading-relaxed text-cream/80">
                        {chunk.text}
                      </p>
                    </div>
                    {chunk.context_text && chunk.context_text !== chunk.text && (
                      <div>
                        <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-cream/45">
                          Parent context (read by the model)
                        </p>
                        <p className="max-h-40 overflow-y-auto whitespace-pre-wrap rounded-lg bg-ink/40 p-2 text-xs leading-relaxed text-cream/65">
                          {chunk.context_text}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </article>
            );
          })
        )}
      </div>
    </section>
  );
}

function ScoreRow({
  label,
  value,
  width,
  color,
}: {
  label: string;
  value: number;
  width: number;
  color: string;
}) {
  const pct = Math.round(Math.max(0, Math.min(1, width)) * 100);
  return (
    <div className="flex items-center gap-2">
      <span className="w-11 shrink-0 text-[10px] uppercase tracking-wide text-cream/45">
        {label}
      </span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-cream/10">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-12 shrink-0 text-right font-mono text-[10px] text-cream/55">
        {value.toFixed(3)}
      </span>
    </div>
  );
}
