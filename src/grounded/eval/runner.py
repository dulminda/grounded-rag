"""Evaluation runners. Each run produces a JSON-serialisable result with per-query scores,
so any two runs over the same queries can be compared with a paired significance test."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from grounded.config import PipelineConfig
from grounded.eval.datasets import VERDICTS, EvalExample
from grounded.eval.metrics import bootstrap_ci, macro_f1, retrieval_metrics
from grounded.generation.llm import LLMError
from grounded.pipeline import RAGPipeline
from grounded.types import Document, unique_doc_ids


@dataclass
class RunResult:
    name: str
    kind: str  # "retrieval" | "verify" | "qa"
    n_queries: int
    metrics: dict[str, dict[str, float]]  # metric -> {mean, ci_low, ci_high}
    per_query: dict[str, list[float]]
    latency_ms: dict[str, float]
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(per_query: dict[str, list[float]], seed: int = 0) -> dict[str, dict[str, float]]:
    out = {}
    for name, values in per_query.items():
        lo, hi = bootstrap_ci(values, seed=seed)
        out[name] = {"mean": float(np.mean(values)), "ci_low": lo, "ci_high": hi}
    return out


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    x = np.asarray(latencies_ms)
    return {
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "mean": float(x.mean()),
    }


def run_retrieval_eval(
    config: PipelineConfig,
    docs: list[Document],
    examples: list[EvalExample],
    depth: int = 100,
) -> RunResult:
    t0 = time.perf_counter()
    pipeline = RAGPipeline.from_documents(config, docs)
    build_s = time.perf_counter() - t0

    for ex in examples[:10]:  # warm caches (stemmer, BLAS) so the first system isn't penalised
        pipeline.retrieve(ex.query, k=depth)

    per_query: dict[str, list[float]] = {}
    latencies = []
    for ex in examples:
        start = time.perf_counter()
        results = pipeline.retrieve(ex.query, k=depth)
        latencies.append((time.perf_counter() - start) * 1000)
        for name, value in retrieval_metrics(unique_doc_ids(results), ex.relevant_doc_ids).items():
            per_query.setdefault(name, []).append(value)

    return RunResult(
        name=config.name,
        kind="retrieval",
        n_queries=len(examples),
        metrics=summarize(per_query),
        per_query=per_query,
        latency_ms=latency_summary(latencies),
        extra={"index_build_s": round(build_s, 2), "config": config.model_dump(),
               "query_ids": [ex.id for ex in examples]},
    )


def _parallel_map(fn: Callable, items: list, workers: int) -> list:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def run_verify_eval(
    pipeline: RAGPipeline,
    examples: list[EvalExample],
    workers: int = 8,
    on_progress: Callable[[int, float], None] | None = None,
) -> RunResult:
    """End-to-end claim verification against SciFact gold verdicts and evidence documents."""
    done, spent = 0, 0.0
    lock = threading.Lock()

    def run_one(ex: EvalExample) -> dict[str, Any]:
        nonlocal done, spent
        try:
            resp = pipeline.verify(ex.query)
        except LLMError as e:
            return {"id": ex.id, "error": str(e)}
        a = resp.answer
        assert a is not None
        with lock:
            done += 1
            spent += a.usage.cost_usd
            if on_progress:
                on_progress(done, spent)
        return {
            "id": ex.id,
            "gold": ex.label,
            "pred": a.verdict,
            "cited": a.cited_doc_ids,
            "retrieved": [s.doc_id for s in resp.sources],
            "invalid_citations": len(a.invalid_citations),
            "cost_usd": a.usage.cost_usd,
            "latency_ms": sum(s["duration_ms"] for s in resp.trace.spans),
        }

    rows = _parallel_map(run_one, examples, workers)
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    by_id = {ex.id: ex for ex in examples}

    per_query: dict[str, list[float]] = {"accuracy": [], "evidence_in_context": []}
    cit_precision, cit_recall = [], []
    for r in rows:
        ex = by_id[r["id"]]
        per_query["accuracy"].append(float(r["pred"] == r["gold"]))
        gold_docs = set(ex.evidence)
        if gold_docs:
            per_query["evidence_in_context"].append(float(bool(gold_docs & set(r["retrieved"]))))
            if r["pred"] == r["gold"]:
                cited = set(r["cited"])
                cit_recall.append(len(cited & gold_docs) / len(gold_docs))
                if cited:
                    cit_precision.append(len(cited & gold_docs) / len(cited))
    per_query["citation_precision"] = cit_precision
    per_query["citation_recall"] = cit_recall

    metrics = summarize({k: v for k, v in per_query.items() if v})
    golds, preds = [r["gold"] for r in rows], [r["pred"] for r in rows]
    metrics["macro_f1"] = {"mean": macro_f1(golds, preds, VERDICTS), "ci_low": float("nan"),
                           "ci_high": float("nan")}
    majority = max(VERDICTS, key=lambda v: sum(g == v for g in golds)) if golds else None
    return RunResult(
        name=pipeline.config.name,
        kind="verify",
        n_queries=len(rows),
        metrics=metrics,
        per_query=per_query,
        latency_ms=latency_summary([r["latency_ms"] for r in rows]) if rows else {},
        extra={
            "errors": len(errors),
            "error_samples": [e["error"] for e in errors[:5]],
            "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
            "cost_per_query_usd": round(sum(r["cost_usd"] for r in rows) / max(len(rows), 1), 5),
            "invalid_citations": sum(r["invalid_citations"] for r in rows),
            "majority_class": majority,
            "majority_baseline_accuracy": (
                sum(g == majority for g in golds) / len(golds) if golds else None
            ),
            "confusion": {
                g: {p: sum(r["gold"] == g and r["pred"] == p for r in rows) for p in VERDICTS}
                for g in VERDICTS
            },
            "rows": rows,
        },
    )


def run_qa_eval(
    pipeline: RAGPipeline,
    examples: list[EvalExample],
    judge: Any,
    workers: int = 8,
) -> RunResult:
    """Free-form QA scored by an LLM judge for faithfulness, plus retrieval hit-rate."""

    def run_one(ex: EvalExample) -> dict[str, Any]:
        try:
            resp = pipeline.answer(ex.query)
            a = resp.answer
            assert a is not None
            verdict = None if a.abstained else judge.judge(ex.query, resp.sources, a.text)
        except LLMError as e:
            return {"id": ex.id, "error": str(e)}
        retrieved = {s.doc_id for s in resp.sources}
        return {
            "id": ex.id,
            "abstained": a.abstained,
            "context_hit": float(bool(ex.relevant_doc_ids & retrieved)),
            "faithfulness": None if verdict is None else verdict.faithfulness,
            "answers_question": None if verdict is None else float(verdict.answers_question),
            "cited_relevant": float(bool(ex.relevant_doc_ids & set(a.cited_doc_ids))),
            "cost_usd": a.usage.cost_usd + (0.0 if verdict is None else verdict.cost_usd),
            "latency_ms": sum(s["duration_ms"] for s in resp.trace.spans),
            "answer": a.text,
        }

    rows = [r for r in _parallel_map(run_one, examples, workers) if "error" not in r]
    answered = [r for r in rows if not r["abstained"]]
    per_query = {
        "context_hit": [r["context_hit"] for r in rows],
        "abstention_rate": [float(r["abstained"]) for r in rows],
        "faithfulness": [r["faithfulness"] for r in answered if r["faithfulness"] is not None],
        "answer_relevance": [r["answers_question"] for r in answered],
        "cited_relevant_doc": [r["cited_relevant"] for r in answered],
    }
    per_query = {k: v for k, v in per_query.items() if v}
    return RunResult(
        name=pipeline.config.name,
        kind="qa",
        n_queries=len(rows),
        metrics=summarize(per_query),
        per_query=per_query,
        latency_ms=latency_summary([r["latency_ms"] for r in rows]) if rows else {},
        extra={
            "errors": len(examples) - len(rows),
            "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
            "rows": rows,
        },
    )
