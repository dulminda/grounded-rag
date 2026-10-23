"""Grounded answering: format retrieved sources, call the LLM, validate citations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from grounded.generation.llm import LLMClient, Usage
from grounded.generation.prompts import (
    QA_SYSTEM,
    VERIFY_SYSTEM,
    format_sources,
    qa_prompt,
    verify_prompt,
)
from grounded.generation.schemas import ClaimVerdict, GroundedAnswer
from grounded.types import ScoredChunk

_MARKER_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    source_id: int
    chunk_id: str
    doc_id: str
    title: str
    snippet: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    abstained: bool
    verdict: str | None = None  # set in claim-verification mode
    invalid_citations: list[int] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""

    @property
    def cited_doc_ids(self) -> list[str]:
        return list(dict.fromkeys(c.doc_id for c in self.citations))


def _resolve(ids: list[int], sources: list[ScoredChunk]) -> tuple[list[Citation], list[int]]:
    """Map 1-based source numbers to chunks, separating out hallucinated numbers."""
    valid, invalid = [], []
    for i in dict.fromkeys(ids):
        if 1 <= i <= len(sources):
            c = sources[i - 1].chunk
            valid.append(Citation(i, c.id, c.doc_id, c.title, c.text[:300]))
        else:
            invalid.append(i)
    return valid, invalid


class Answerer:
    def __init__(self, llm: LLMClient, max_source_chars: int = 2500) -> None:
        self.llm = llm
        self.max_source_chars = max_source_chars

    def answer(self, question: str, sources: list[ScoredChunk]) -> Answer:
        if not sources:
            return Answer("No relevant sources were found.", [], abstained=True)
        ctx = format_sources(sources, self.max_source_chars)
        result = self.llm.generate(QA_SYSTEM, qa_prompt(question, ctx), GroundedAnswer)
        out = result.parsed
        # Trust inline markers as well as the citations list; models occasionally miss one.
        ids = out.citations + [int(m) for m in _MARKER_RE.findall(out.answer)]
        citations, invalid = _resolve(ids, sources)
        abstained = out.insufficient_context or not citations
        return Answer(out.answer, citations, abstained, None, invalid, result.usage, result.model)

    def verify(self, claim: str, sources: list[ScoredChunk]) -> Answer:
        if not sources:
            return Answer("No sources.", [], abstained=True, verdict="NOT_ENOUGH_INFO")
        ctx = format_sources(sources, self.max_source_chars)
        result = self.llm.generate(VERIFY_SYSTEM, verify_prompt(claim, ctx), ClaimVerdict)
        out = result.parsed
        citations, invalid = _resolve(out.citations, sources)
        verdict = out.verdict
        # A verdict with no valid evidence is not grounded: downgrade it rather than trust it.
        if verdict != "NOT_ENOUGH_INFO" and not citations:
            verdict = "NOT_ENOUGH_INFO"
        return Answer(
            out.rationale, citations, verdict == "NOT_ENOUGH_INFO", verdict, invalid,
            result.usage, result.model,
        )
