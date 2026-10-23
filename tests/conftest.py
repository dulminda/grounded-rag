from __future__ import annotations

import json
from pathlib import Path

import pytest

from grounded.config import PipelineConfig
from grounded.generation.llm import LLMRefusal, LLMResult, Usage
from grounded.generation.schemas import (
    ClaimVerdict,
    FaithfulnessJudgement,
    GroundedAnswer,
    JudgedClaim,
)
from grounded.ingest.loaders import load_jsonl_corpus
from grounded.pipeline import RAGPipeline

FIXTURES = Path(__file__).parent / "fixtures"


class FakeLLM:
    """Deterministic stand-in for Claude. Responses are keyed by output schema."""

    model = "fake-model"

    def __init__(self, citations=(1,), verdict="SUPPORT", insufficient=False, refuse=False):
        self.citations = list(citations)
        self.verdict = verdict
        self.insufficient = insufficient
        self.refuse = refuse
        self.calls: list[tuple[str, str]] = []

    def generate(self, system, prompt, schema):
        self.calls.append((system, prompt))
        if self.refuse:
            raise LLMRefusal("declined (category=test)")
        if schema is GroundedAnswer:
            markers = " ".join(f"[{c}]" for c in self.citations)
            parsed = GroundedAnswer(
                answer=f"Answer text {markers}".strip(),
                citations=self.citations,
                insufficient_context=self.insufficient,
            )
        elif schema is ClaimVerdict:
            parsed = ClaimVerdict(
                verdict=self.verdict, rationale="because", citations=self.citations
            )
        elif schema is FaithfulnessJudgement:
            parsed = FaithfulnessJudgement(
                claims=[JudgedClaim(claim="a", supported=True, source_ids=[1]),
                        JudgedClaim(claim="b", supported=False, source_ids=[])],
                answers_question=True,
            )
        else:  # pragma: no cover
            raise AssertionError(f"unexpected schema {schema}")
        return LLMResult(parsed=parsed, usage=Usage(100, 20, 0.001), model=self.model,
                         latency_ms=1.0)


@pytest.fixture
def docs():
    return load_jsonl_corpus(FIXTURES / "corpus.jsonl")


@pytest.fixture
def examples():
    from grounded.eval.datasets import load_jsonl_examples

    return load_jsonl_examples(FIXTURES / "examples.jsonl")


@pytest.fixture
def bm25_config():
    return PipelineConfig.model_validate(
        {"name": "test-bm25", "chunking": {"max_words": 60, "overlap_sentences": 1},
         "retrieval": {"method": "bm25"}, "generation": {"context_chunks": 3}}
    )


@pytest.fixture
def pipeline(docs, bm25_config):
    return RAGPipeline.from_documents(bm25_config, docs, llm=FakeLLM())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
