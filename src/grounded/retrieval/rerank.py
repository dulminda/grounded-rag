"""Second-stage rerankers applied to the top-N first-stage candidates."""

from __future__ import annotations

from typing import Protocol

from grounded.types import ScoredChunk


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]: ...


class CrossEncoderReranker:
    """Cross-encoder reranking, e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``."""

    name = "cross-encoder"

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 32):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("pip install 'grounded-rag[neural]' for reranking") from e
        self._model = CrossEncoder(model)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.indexed_text) for c in candidates]
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: -float(x[1]))
        return [ScoredChunk(c.chunk, float(s), self.name) for c, s in ranked[:k]]
