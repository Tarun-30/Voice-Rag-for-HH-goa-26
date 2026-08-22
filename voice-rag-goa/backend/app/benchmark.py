"""
Benchmark harness: the honest latency + retrieval-quality report.

Runs the real pipeline components (embedder, retriever, guardrails, and
optionally the LLM harness) over a suite of MSMARCO-XI queries, across the full
{strategy} x {mode} grid, and produces `data/benchmark_report.json`
(`schemas.BenchmarkReport`) which `/api/analytics` serves and the frontend
dashboard renders.

Two independent things are measured:

* **Latency** - per-stage (embed / retrieval / guardrail / ttft) and end-to-end,
  summarised as P50/P70/P90/P95/P99/P100 via `metrics.summarize`, and checked
  against `config.BUDGET`. Queries are run under a configurable concurrency so
  the tail reflects contention, not a quiet single-shot best case.

* **Retrieval quality** - Recall@1/3/5, MRR@10, and nDCG@5, scored against the
  dataset's own `is_selected` gold-passage labels (carried through ingest as
  `gold_passage_ids`). This is real accuracy on human labels, not a proxy.

Standalone:
    python -m app.benchmark
    python -m app.benchmark --queries 200 --concurrency 8 --no-llm
    python -m app.benchmark --strategies hierarchical,semantic --modes hybrid,dense
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import config
from . import guardrails, ingest
from .harness import get_harness
from .metrics import summarize
from .retrieval import get_retriever
from .schemas import (
    BenchmarkReport,
    MetricSummary,
    RetrievalQuality,
    StrategyBenchmark,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Per-query sample
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Sample:
    """One query's measured latencies and retrieved ordering."""

    embed_ms: float = 0.0
    retrieval_ms: float = 0.0
    guardrail_ms: float = 0.0
    ttft_ms: float = 0.0
    query_to_first_token_ms: float = 0.0
    total_e2e_ms: float = 0.0
    refused: bool = False
    error: bool = False
    retrieved_passages: list[str] = field(default_factory=list)
    gold_passages: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Latency metric layout (name -> budget)
# --------------------------------------------------------------------------- #

_LATENCY_FIELDS: dict[str, float] = {
    "embed_ms": config.BUDGET.embed_ms,
    "retrieval_ms": config.BUDGET.retrieval_ms,
    "guardrail_ms": config.BUDGET.guardrail_ms,
    "ttft_ms": config.BUDGET.ttft_ms,
    "query_to_first_token_ms": config.BUDGET.total_ms,
    "total_e2e_ms": 0.0,
}


# --------------------------------------------------------------------------- #
# Single-query runner
# --------------------------------------------------------------------------- #


def _retrieve_and_guard(retriever, query_text: str, strategy: str, mode: str, top_k: int):
    """The synchronous (CPU-bound) part of a turn: guard -> embed -> retrieve.

    Returns (chunks, sample, clean_query, refused_early). Run in a worker thread
    by the async caller so N of these can overlap without blocking the loop.
    """
    sample = _Sample()
    g0 = time.perf_counter()
    verdict = guardrails.check_input(query_text)
    clean = verdict.sanitized_query or query_text
    if not verdict.allowed:
        sample.guardrail_ms = (time.perf_counter() - g0) * 1000.0
        sample.refused = True
        return [], sample, clean, True

    e0 = time.perf_counter()
    query_vector = retriever.embed_query(clean) if mode != "sparse" else None
    sample.embed_ms = (time.perf_counter() - e0) * 1000.0

    r0 = time.perf_counter()
    chunks = retriever.search(
        clean, query_vector=query_vector, strategy=strategy, mode=mode, top_k=top_k
    )
    sample.retrieval_ms = (time.perf_counter() - r0) * 1000.0

    ctx = guardrails.check_context(clean, chunks, query_vector=query_vector)
    sample.guardrail_ms = (time.perf_counter() - g0) * 1000.0 - sample.embed_ms - sample.retrieval_ms
    sample.refused = not ctx.allowed
    sample.retrieved_passages = _dedupe([c.passage_id for c in chunks])
    return chunks, sample, clean, sample.refused


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


async def _run_one(
    retriever,
    harness,
    query_text: str,
    gold: list[str],
    strategy: str,
    mode: str,
    top_k: int,
    include_llm: bool,
) -> _Sample:
    try:
        chunks, sample, clean, refused_early = await asyncio.to_thread(
            _retrieve_and_guard, retriever, query_text, strategy, mode, top_k
        )
        sample.gold_passages = gold

        if include_llm and not refused_early and chunks:
            first: dict[str, float] = {}
            started = time.perf_counter()

            async def on_token(_delta: str, is_first: bool) -> None:
                if is_first and "t" not in first:
                    first["t"] = time.perf_counter()

            gen = await harness.generate(clean, chunks, on_token=on_token)
            sample.ttft_ms = (
                (first["t"] - started) * 1000.0 if "t" in first else gen.ttft_ms
            )

        sample.query_to_first_token_ms = (
            sample.embed_ms + sample.retrieval_ms + sample.guardrail_ms + sample.ttft_ms
        )
        # No STT in the benchmark, so e2e == query-to-first-token here.
        sample.total_e2e_ms = sample.query_to_first_token_ms
        return sample
    except Exception as exc:  # pragma: no cover - recorded, not fatal
        logger.warning("benchmark query failed (%s/%s): %s", strategy, mode, exc)
        return _Sample(error=True, gold_passages=gold)


# --------------------------------------------------------------------------- #
# Retrieval-quality scoring (against is_selected gold labels)
# --------------------------------------------------------------------------- #


def _score_quality(samples: list[_Sample]) -> RetrievalQuality:
    scored = [s for s in samples if s.gold_passages and not s.error]
    if not scored:
        return RetrievalQuality()

    recall_1 = recall_3 = recall_5 = mrr = ndcg = hits = 0.0
    for sample in scored:
        gold = set(sample.gold_passages)
        ranked = sample.retrieved_passages
        recall_1 += _recall_at(ranked, gold, 1)
        recall_3 += _recall_at(ranked, gold, 3)
        recall_5 += _recall_at(ranked, gold, 5)
        mrr += _mrr_at(ranked, gold, 10)
        ndcg += _ndcg_at(ranked, gold, 5)
        hits += 1.0 if gold.intersection(ranked) else 0.0

    n = len(scored)
    return RetrievalQuality(
        queries_scored=n,
        recall_at_1=round(recall_1 / n, 4),
        recall_at_3=round(recall_3 / n, 4),
        recall_at_5=round(recall_5 / n, 4),
        mrr_at_10=round(mrr / n, 4),
        ndcg_at_5=round(ndcg / n, 4),
        hit_rate=round(hits / n, 4),
    )


def _recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    found = sum(1 for pid in ranked[:k] if pid in gold)
    return found / len(gold)


def _mrr_at(ranked: list[str], gold: set[str], k: int) -> float:
    for rank, pid in enumerate(ranked[:k], start=1):
        if pid in gold:
            return 1.0 / rank
    return 0.0


def _ndcg_at(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for i, pid in enumerate(ranked[:k]):
        if pid in gold:
            dcg += 1.0 / math.log2(i + 2)  # binary relevance
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


# --------------------------------------------------------------------------- #
# Cell + full run
# --------------------------------------------------------------------------- #


async def _run_cell(
    retriever,
    harness,
    queries: list[tuple[str, list[str]]],
    strategy: str,
    mode: str,
    top_k: int,
    include_llm: bool,
    concurrency: int,
) -> tuple[StrategyBenchmark, list[_Sample]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(query_text: str, gold: list[str]) -> _Sample:
        async with semaphore:
            return await _run_one(
                retriever, harness, query_text, gold, strategy, mode, top_k, include_llm
            )

    samples = await asyncio.gather(*(guarded(q, g) for q, g in queries))
    samples = list(samples)

    metrics: dict[str, MetricSummary] = {}
    for name, budget in _LATENCY_FIELDS.items():
        values = [getattr(s, name) for s in samples if not s.error and getattr(s, name) > 0]
        if values:
            metrics[name] = summarize(name, values, budget_ms=budget)

    cell = StrategyBenchmark(
        strategy=strategy,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        queries=len(samples),
        errors=sum(1 for s in samples if s.error),
        refusals=sum(1 for s in samples if s.refused),
        metrics=metrics,
        quality=_score_quality(samples),
    )
    return cell, samples


async def run_benchmark(
    *,
    queries: int = config.BENCHMARK_QUERIES,
    concurrency: int = config.BENCHMARK_CONCURRENCY,
    strategies: list[str] | None = None,
    modes: list[str] | None = None,
    include_llm: bool = True,
    warmup: int = config.BENCHMARK_WARMUP,
    write: bool = True,
) -> BenchmarkReport:
    """Run the full grid and (optionally) persist the report to disk."""
    config.ensure_dirs()
    retriever = get_retriever()
    harness = get_harness()
    if not retriever.ready:
        raise RuntimeError("No indexes loaded - build them with `python -m app.ingest` first.")

    available = retriever.strategies
    strategies = [s for s in (strategies or available) if s in available] or available
    modes = modes or ["hybrid", "dense", "sparse"]
    include_llm = include_llm and harness.enabled  # offline harness is deterministic; skip it in bench by default unless keys exist

    queryset = ingest.load_queryset()
    if not queryset:
        raise RuntimeError("Query set is empty - run ingest first.")

    # Prefer queries that carry gold labels so the quality numbers are meaningful,
    # then top up with the rest to reach the requested count.
    with_gold = [q for q in queryset if q.gold_passage_ids]
    without_gold = [q for q in queryset if not q.gold_passage_ids]
    ordered = with_gold + without_gold
    picked = ordered[:queries]
    pairs: list[tuple[str, list[str]]] = [
        ((q.eng_query or q.query), q.gold_passage_ids) for q in picked
    ]
    logger.info(
        "benchmark: %d queries (%d with gold), strategies=%s modes=%s llm=%s conc=%d",
        len(pairs), sum(1 for _, g in pairs if g), strategies, modes, include_llm, concurrency,
    )

    # Warm the ONNX graph + BM25 caches so cold-start does not skew p50.
    retriever.embedder.warmup()
    for query_text, _ in pairs[: max(0, warmup)]:
        await _run_one(retriever, harness, query_text, [], strategies[0], modes[0], config.RETRIEVE_TOP_K, False)

    started = time.perf_counter()
    per_strategy: list[StrategyBenchmark] = []
    all_samples: list[_Sample] = []
    for strategy in strategies:
        for mode in modes:
            cell, samples = await _run_cell(
                retriever, harness, pairs, strategy, mode,
                config.RETRIEVE_TOP_K, include_llm, concurrency,
            )
            per_strategy.append(cell)
            all_samples.extend(samples)
            q2ft = cell.metrics.get("query_to_first_token_ms")
            quality = cell.quality
            logger.info(
                "  %-12s %-7s q2ft p50=%.2f p95=%.2f | recall@5=%.3f mrr@10=%.3f (%d scored)",
                strategy, mode,
                q2ft.p50 if q2ft else 0.0, q2ft.p95 if q2ft else 0.0,
                quality.recall_at_5, quality.mrr_at_10, quality.queries_scored,
            )
    wall = time.perf_counter() - started

    overall: dict[str, MetricSummary] = {}
    for name, budget in _LATENCY_FIELDS.items():
        values = [getattr(s, name) for s in all_samples if not s.error and getattr(s, name) > 0]
        if values:
            overall[name] = summarize(name, values, budget_ms=budget)

    chunk_stats = [
        st for st in (ingest.load_stats(s) for s in retriever.strategies) if st is not None
    ]

    report = BenchmarkReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset=config.HF_DATASET_REPO,
        languages=[s.key for s in config.resolve_languages()],
        embed_model=config.EMBED_MODEL,
        llm_provider=harness.primary_provider,
        llm_model=harness.primary_model,
        total_queries=len(pairs),
        concurrency=concurrency,
        wall_seconds=round(wall, 2),
        llm_enabled=include_llm,
        budget=_budget_dict(),
        overall=overall,
        per_strategy=per_strategy,
        chunk_stats=chunk_stats,
        notes=_notes(include_llm, harness),
    )

    if write:
        config.BENCHMARK_REPORT_PATH.write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info("wrote %s", config.BENCHMARK_REPORT_PATH)
    return report


def _budget_dict() -> dict[str, float]:
    b = config.BUDGET
    return {
        "embed_ms": b.embed_ms, "retrieval_ms": b.retrieval_ms,
        "guardrail_ms": b.guardrail_ms, "ttft_ms": b.ttft_ms, "total_ms": b.total_ms,
    }


def _notes(include_llm: bool, harness) -> list[str]:
    notes = [
        "query_to_first_token_ms excludes STT (third-party ASR network time), per the "
        "sub-200ms retrieval-to-generation target.",
        f"Retrieval quality scored against MSMARCO-XI is_selected gold labels; "
        f"backend={get_retriever().resolve_backend(None)}.",
    ]
    if not include_llm:
        notes.append("LLM stage disabled: ttft/generation reflect retrieval+guardrail only.")
    elif harness.primary_provider == "offline":
        notes.append("Offline extractive harness used (no LLM API key configured).")
    return notes


# --------------------------------------------------------------------------- #
# Report loader (used by /api/analytics)
# --------------------------------------------------------------------------- #


def load_report() -> BenchmarkReport | None:
    path = config.BENCHMARK_REPORT_PATH
    if not path.exists():
        return None
    try:
        return BenchmarkReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        logger.warning("failed to load benchmark report: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the Voice-RAG pipeline.")
    parser.add_argument("--queries", type=int, default=config.BENCHMARK_QUERIES)
    parser.add_argument("--concurrency", type=int, default=config.BENCHMARK_CONCURRENCY)
    parser.add_argument("--strategies", help="comma-separated (default: all indexed)")
    parser.add_argument("--modes", help="comma-separated of hybrid,dense,sparse")
    parser.add_argument("--no-llm", action="store_true", help="skip the generation stage")
    args = parser.parse_args()

    logging.basicConfig(
        level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    strategies = args.strategies.split(",") if args.strategies else None
    modes = args.modes.split(",") if args.modes else None
    report = asyncio.run(
        run_benchmark(
            queries=args.queries,
            concurrency=args.concurrency,
            strategies=strategies,
            modes=modes,
            include_llm=not args.no_llm,
        )
    )
    _print_summary(report)


def _print_summary(report: BenchmarkReport) -> None:
    print("\n" + "=" * 72)
    print(f"Voice-RAG benchmark  |  {report.total_queries} queries  |  {report.wall_seconds}s")
    print(f"dataset={report.dataset}  embed={report.embed_model}")
    print(f"llm={report.llm_provider}:{report.llm_model} (enabled={report.llm_enabled})")
    print("-" * 72)
    q2ft = report.overall.get("query_to_first_token_ms")
    if q2ft:
        verdict = "PASS" if q2ft.p50 <= report.budget.get("total_ms", 200) else "OVER"
        print(f"query->first-token   p50={q2ft.p50:7.2f}ms  p70={q2ft.p70:7.2f}ms  "
              f"p95={q2ft.p95:7.2f}ms  p100={q2ft.p100:7.2f}ms  [{verdict}]")
    for name in ("embed_ms", "retrieval_ms", "guardrail_ms", "ttft_ms"):
        m = report.overall.get(name)
        if m:
            print(f"{name:20s} p50={m.p50:7.2f}ms  p70={m.p70:7.2f}ms  "
                  f"p95={m.p95:7.2f}ms  p100={m.p100:7.2f}ms")
    print("-" * 72)
    for cell in report.per_strategy:
        q = cell.quality
        print(f"{cell.strategy:12s} {cell.mode:7s}  recall@1={q.recall_at_1:.3f} "
              f"recall@5={q.recall_at_5:.3f}  mrr@10={q.mrr_at_10:.3f}  "
              f"ndcg@5={q.ndcg_at_5:.3f}  ({q.queries_scored} scored, {cell.refusals} refused)")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
