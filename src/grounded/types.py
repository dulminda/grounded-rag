"""Core data types shared across ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    title: str = ""
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def indexed_text(self) -> str:
        """Text used for indexing: the title carries a lot of signal for short documents."""
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    source: str  # which retriever / reranker produced the score

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


def unique_doc_ids(results: list[ScoredChunk]) -> list[str]:
    """Collapse a ranked chunk list to a ranked document list (first occurrence wins)."""
    seen: set[str] = set()
    ranked: list[str] = []
    for r in results:
        if r.doc_id not in seen:
            seen.add(r.doc_id)
            ranked.append(r.doc_id)
    return ranked
