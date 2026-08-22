"""
Percentile + summary helpers shared by the live-metrics window and the offline
benchmark. Kept dependency-light (numpy only) so both callers agree exactly on
how a p70 is computed.
"""

from __future__ import annotations

import statistics
from typing import Sequence

import numpy as np

from .schemas import MetricSummary


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy's default), pct in [0,100]."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def summarize(
    metric: str,
    values: Sequence[float],
    *,
    unit: str = "ms",
    budget_ms: float = 0.0,
) -> MetricSummary:
    """Build a MetricSummary (mean/stdev/min + p50..p100 + budget adherence)."""
    data = [float(v) for v in values if v is not None]
    if not data:
        return MetricSummary(metric=metric, unit=unit, budget_ms=budget_ms)  # type: ignore[arg-type]
    within = (
        sum(1 for v in data if v <= budget_ms) / len(data) if budget_ms > 0 else 0.0
    )
    return MetricSummary(
        metric=metric,
        unit=unit,  # type: ignore[arg-type]
        samples=len(data),
        mean=round(statistics.fmean(data), 3),
        stdev=round(statistics.pstdev(data), 3) if len(data) > 1 else 0.0,
        min=round(min(data), 3),
        p50=round(percentile(data, 50), 3),
        p70=round(percentile(data, 70), 3),
        p90=round(percentile(data, 90), 3),
        p95=round(percentile(data, 95), 3),
        p99=round(percentile(data, 99), 3),
        p100=round(max(data), 3),
        budget_ms=budget_ms,
        within_budget_ratio=round(within, 4),
    )
