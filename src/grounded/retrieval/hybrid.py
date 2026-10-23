"""Hybrid retrieval: fuse ranked lists from several retrievers.

* ``rrf``: Reciprocal Rank Fusion (Cormack et al., 2009). Rank-based, so it needs no score
  calibration between BM25 (unbounded) and cosine similarity ([-1, 1]).
* ``linear``: convex combination of per-query min-max normalised scores. Can beat RRF when
  the weights are tuned, but is sensitive to score distributions.
"""

from __future__ import annotations

from collections.abc import Sequence

from grounded.retrieval.base import Retriever
from grounded.types import Chunk, ScoredChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[list[ScoredChunk]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[ScoredChunk]:
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for w, results in zip(weights, ranked_lists, strict=True):
        for rank, r in enumerate(results, start=1):
            scores[r.chunk.id] = scores.get(r.chunk.id, 0.0) + w / (k + rank)
            chunks[r.chunk.id] = r.chunk
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [ScoredChunk(chunks[cid], scores[cid], "rrf") for cid in order]


def linear_fusion(
    ranked_lists: Sequence[list[ScoredChunk]], weights: Sequence[float] | None = None
) -> list[ScoredChunk]:
    weights = weights or [1.0 / len(ranked_lists)] * len(ranked_lists)
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for w, results in zip(weights, ranked_lists, strict=True):
        if not results:
            continue
        hi = max(r.score for r in results)
        lo = min(r.score for r in results)
        span = (hi - lo) or 1.0
        for r in results:
            scores[r.chunk.id] = scores.get(r.chunk.id, 0.0) + w * (r.score - lo) / span
            chunks[r.chunk.id] = r.chunk
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [ScoredChunk(chunks[cid], scores[cid], "linear") for cid in order]


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        method: str = "rrf",
        weights: Sequence[float] | None = None,
        rrf_k: int = 60,
        candidates: int = 100,
    ) -> None:
        if method not in {"rrf", "linear"}:
            raise ValueError(f"unknown fusion method {method!r}")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("need one weight per retriever")
        self.retrievers = list(retrievers)
        self.chunks = self.retrievers[0].chunks
        self.method, self.weights, self.rrf_k, self.candidates = method, weights, rrf_k, candidates

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        n = max(k, self.candidates)
        lists = [r.retrieve(query, n) for r in self.retrievers]
        if self.method == "rrf":
            fused = reciprocal_rank_fusion(lists, k=self.rrf_k, weights=self.weights)
        else:
            fused = linear_fusion(lists, weights=self.weights)
        return fused[:k]
