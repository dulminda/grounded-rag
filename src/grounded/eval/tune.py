"""Hyperparameter search on the TRAIN split. Report on dev only with the chosen values,
so the published dev numbers are not tuned on the data they are reported on."""

from __future__ import annotations

import itertools

import numpy as np

from grounded.eval.datasets import EvalExample
from grounded.eval.metrics import ndcg_at_k
from grounded.retrieval.base import Retriever
from grounded.retrieval.bm25 import BM25Retriever
from grounded.retrieval.hybrid import linear_fusion, reciprocal_rank_fusion
from grounded.types import Chunk, ScoredChunk, unique_doc_ids


def _mean_ndcg(ranked_lists: list[list[ScoredChunk]], examples: list[EvalExample]) -> float:
    return float(np.mean([
        ndcg_at_k(unique_doc_ids(r), ex.relevant_doc_ids, 10)
        for r, ex in zip(ranked_lists, examples, strict=True)
    ]))


def tune_bm25(
    chunks: list[Chunk],
    examples: list[EvalExample],
    k1_grid: tuple[float, ...] = (0.6, 0.9, 1.2, 1.5),
    b_grid: tuple[float, ...] = (0.3, 0.4, 0.6, 0.75, 0.9),
) -> list[dict[str, float]]:
    rows = []
    for k1, b in itertools.product(k1_grid, b_grid):
        r = BM25Retriever(chunks, k1=k1, b=b)
        rows.append({"k1": k1, "b": b,
                     "ndcg@10": _mean_ndcg([r.retrieve(e.query, 10) for e in examples], examples)})
    return sorted(rows, key=lambda x: -x["ndcg@10"])


def tune_fusion(
    lexical: Retriever,
    dense: Retriever,
    examples: list[EvalExample],
    dense_weights: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    candidates: int = 100,
) -> list[dict[str, float | str]]:
    """Retrieve once per retriever, then sweep fusion settings over cached ranked lists."""
    lex = [lexical.retrieve(e.query, candidates) for e in examples]
    den = [dense.retrieve(e.query, candidates) for e in examples]
    rows: list[dict[str, float | str]] = []
    for w in dense_weights:
        weights = [1.0 - w, w]
        for method in ("rrf", "linear"):
            fused = [
                (reciprocal_rank_fusion([a, b], weights=weights) if method == "rrf"
                 else linear_fusion([a, b], weights=weights))[:10]
                for a, b in zip(lex, den, strict=True)
            ]
            rows.append({"method": method, "dense_weight": w,
                         "ndcg@10": _mean_ndcg(fused, examples)})
    return sorted(rows, key=lambda x: -float(x["ndcg@10"]))
