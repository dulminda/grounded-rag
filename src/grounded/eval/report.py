"""Render evaluation results as a markdown report with CIs and paired significance tests."""

from __future__ import annotations

import math
from typing import Any

from grounded.eval.metrics import paired_bootstrap_test


def _fmt(m: dict[str, float]) -> str:
    if math.isnan(m.get("ci_low", math.nan)):
        return f"{m['mean']:.3f}"
    return f"{m['mean']:.3f} <sub>[{m['ci_low']:.3f}, {m['ci_high']:.3f}]</sub>"


def retrieval_table(runs: list[dict[str, Any]], baseline: str | None = None,
                    key_metric: str = "ndcg@10") -> str:
    metrics = ["ndcg@10", "mrr@10", "recall@10", "recall@100"]
    base = next((r for r in runs if r["name"] == baseline), runs[0])
    header = (
        "| System | " + " | ".join(metrics)
        + f" | Δ {key_metric} vs `{base['name']}` | p50 / p95 latency |"
    )
    lines = [header, "|" + "---|" * (len(metrics) + 3)]
    for r in runs:
        cells = [_fmt(r["metrics"][m]) for m in metrics]
        if r is base:
            delta = "—"
        else:
            d, p = paired_bootstrap_test(base["per_query"][key_metric],
                                         r["per_query"][key_metric])
            sig = " ✱" if p < 0.05 else ""
            p_txt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
            delta = f"{d:+.3f} ({p_txt}){sig}"
        lat = r["latency_ms"]
        lines.append(
            f"| `{r['name']}` | " + " | ".join(cells)
            + f" | {delta} | {lat['p50']:.2f} / {lat['p95']:.2f} ms |"
        )
    lines.append("")
    lines.append(
        "Means over queries with 95% bootstrap CIs. Δ uses a paired bootstrap test over "
        "queries; ✱ marks p < 0.05. Latency is retrieval only, single-threaded CPU."
    )
    return "\n".join(lines)


def verify_table(runs: list[dict[str, Any]]) -> str:
    cols = ["accuracy", "macro_f1", "citation_precision", "citation_recall", "evidence_in_context"]
    lines = ["| System | " + " | ".join(cols) + " | $/query | p50 latency |",
             "|" + "---|" * (len(cols) + 3)]
    for r in runs:
        cells = [_fmt(r["metrics"][c]) if c in r["metrics"] else "—" for c in cols]
        lines.append(
            f"| `{r['name']}` | " + " | ".join(cells)
            + f" | ${r['extra']['cost_per_query_usd']:.4f}"
            + f" | {r['latency_ms'].get('p50', 0):.0f} ms |"
        )
    if runs:
        x = runs[0]["extra"]
        lines.append(
            f"| majority class (`{x['majority_class']}`) | "
            f"{x['majority_baseline_accuracy']:.3f} | — | — | — | — | $0 | — |"
        )
    return "\n".join(lines)
