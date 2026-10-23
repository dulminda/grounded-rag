"""The RAG pipeline: retrieve -> (rerank) -> generate, with tracing and cost accounting."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grounded.config import PipelineConfig
from grounded.generation.answerer import Answer, Answerer
from grounded.generation.llm import LLMClient
from grounded.ingest.chunking import chunk_corpus
from grounded.retrieval.base import Retriever
from grounded.retrieval.bm25 import BM25Retriever, RM3Params
from grounded.retrieval.dense import DenseRetriever, LSAEmbedder, SentenceTransformerEmbedder
from grounded.retrieval.hybrid import HybridRetriever
from grounded.retrieval.rerank import CrossEncoderReranker, Reranker
from grounded.tracing import Trace
from grounded.types import Chunk, Document, ScoredChunk


def build_retriever(cfg: PipelineConfig, chunks: list[Chunk]) -> Retriever:
    r = cfg.retrieval

    def bm25() -> BM25Retriever:
        rm3 = RM3Params(**r.bm25.rm3.model_dump()) if r.bm25.rm3 else None
        return BM25Retriever(chunks, k1=r.bm25.k1, b=r.bm25.b, rm3=rm3)

    def dense() -> DenseRetriever:
        if r.dense.backend == "lsa":
            return DenseRetriever(chunks, LSAEmbedder(dim=r.dense.dim))
        return DenseRetriever(
            chunks, SentenceTransformerEmbedder(r.dense.model, query_prefix=r.dense.query_prefix)
        )

    if r.method == "bm25":
        return bm25()
    if r.method == "dense":
        return dense()
    f = r.fusion
    return HybridRetriever(
        [bm25(), dense()], method=f.method, weights=f.weights, rrf_k=f.rrf_k,
        candidates=f.candidates,
    )


def build_reranker(cfg: PipelineConfig) -> Reranker | None:
    rr = cfg.retrieval.rerank
    if rr.backend == "cross-encoder":
        return CrossEncoderReranker(rr.model)
    return None


@dataclass
class RAGResponse:
    query: str
    answer: Answer | None
    sources: list[ScoredChunk]
    trace: Trace

    def to_dict(self) -> dict[str, Any]:
        a = self.answer
        return {
            "query": self.query,
            "answer": None if a is None else {
                "text": a.text,
                "abstained": a.abstained,
                "verdict": a.verdict,
                "citations": [c.__dict__ for c in a.citations],
                "invalid_citations": a.invalid_citations,
                "model": a.model,
            },
            "sources": [
                {"rank": i, "chunk_id": s.chunk.id, "doc_id": s.doc_id, "title": s.chunk.title,
                 "score": round(s.score, 6), "retriever": s.source}
                for i, s in enumerate(self.sources, start=1)
            ],
            "usage": None if a is None else a.usage.__dict__,
            "trace": self.trace.to_dict(),
        }


class RAGPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        retriever: Retriever,
        reranker: Reranker | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.reranker = reranker
        self.answerer = (
            Answerer(llm, config.generation.max_source_chars) if llm is not None else None
        )

    @classmethod
    def from_documents(
        cls, config: PipelineConfig, docs: list[Document], llm: LLMClient | None = None
    ) -> RAGPipeline:
        chunks = chunk_corpus(docs, config.chunking.max_words, config.chunking.overlap_sentences)
        return cls(config, build_retriever(config, chunks), build_reranker(config), llm)

    def retrieve(self, query: str, k: int = 10, trace: Trace | None = None) -> list[ScoredChunk]:
        trace = trace or Trace()
        rr = self.config.retrieval.rerank
        first_k = max(k, rr.candidates) if self.reranker else k
        with trace.span("retrieve", method=self.config.retrieval.method, k=first_k) as attrs:
            results = self.retriever.retrieve(query, first_k)
            attrs["n_results"] = len(results)
        if self.reranker:
            with trace.span("rerank", backend=rr.backend):
                results = self.reranker.rerank(query, results, k)
        return results[:k]

    def _generate(self, query: str, mode: str) -> RAGResponse:
        if self.answerer is None:
            raise RuntimeError("pipeline was built without an LLM; only retrieve() is available")
        trace = Trace()
        sources = self.retrieve(query, self.config.generation.context_chunks, trace)
        with trace.span("generate", model=self.config.generation.model, mode=mode) as attrs:
            fn = self.answerer.answer if mode == "qa" else self.answerer.verify
            answer = fn(query, sources)
            attrs.update(
                input_tokens=answer.usage.input_tokens,
                output_tokens=answer.usage.output_tokens,
                cost_usd=round(answer.usage.cost_usd, 6),
            )
        return RAGResponse(query, answer, sources, trace)

    def answer(self, question: str) -> RAGResponse:
        return self._generate(question, "qa")

    def verify(self, claim: str) -> RAGResponse:
        return self._generate(claim, "verify")

    # Persistence: indexes are pickled. Only load index files you built yourself.
    def save_index(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"config": self.config.model_dump(), "retriever": self.retriever}, f)

    @classmethod
    def load_index(cls, path: str | Path, llm: LLMClient | None = None) -> RAGPipeline:
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301 - trusted, locally built artifact
        config = PipelineConfig.model_validate(state["config"])
        return cls(config, state["retriever"], build_reranker(config), llm)
