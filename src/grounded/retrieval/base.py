"""Retriever interface and top-k helper."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from grounded.types import Chunk, ScoredChunk


class Retriever(Protocol):
    name: str
    chunks: list[Chunk]

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]: ...


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best first, in O(n + k log k)."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]
