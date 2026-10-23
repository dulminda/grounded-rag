"""Retrieval and generation metrics, plus bootstrap statistics for comparing systems.

All retrieval metrics use binary relevance at the document level.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    return len(relevant.intersection(ranked[:k])) / k


def mrr_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(
    ranked: Sequence[str], relevant: frozenset[str], ks: Sequence[int] = (1, 5, 10, 100)
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
    out["ndcg@10"] = ndcg_at_k(ranked, relevant, 10)
    out["mrr@10"] = mrr_at_k(ranked, relevant, 10)
    return out


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred, strict=True))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred, strict=True))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred, strict=True))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_resamples, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_test(
    baseline: Sequence[float], candidate: Sequence[float], n_resamples: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """Mean difference (candidate - baseline) and a two-sided paired-bootstrap p-value.

    Resamples queries, not scores, so per-query difficulty is shared between the two systems;
    this is far more sensitive than comparing two independent confidence intervals.
    """
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired test needs per-query scores for the same queries")
    diff = b - a
    observed = float(diff.mean())
    if not diff.any():
        return observed, 1.0
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, diff.size, size=(n_resamples, diff.size))].mean(axis=1)
    # Shift to the null (mean difference 0) and count resamples at least as extreme.
    p = float(np.mean(np.abs(boot - observed) >= abs(observed)))
    return observed, max(p, 1.0 / n_resamples)
