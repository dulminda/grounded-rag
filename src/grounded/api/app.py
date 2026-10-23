"""HTTP API.

Run with ``grounded serve --index indexes/bm25.pkl`` or
``GROUNDED_INDEX=... uvicorn grounded.api.app:create_app --factory``.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from grounded.generation.llm import LLMError, LLMRefusal
from grounded.pipeline import RAGPipeline

log = logging.getLogger("grounded.api")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=10, ge=1, le=100)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class VerifyRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=2000)


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    if pipeline is None:
        from grounded.generation.llm import ClaudeClient

        index = os.environ.get("GROUNDED_INDEX")
        if not index:
            raise RuntimeError("set GROUNDED_INDEX to the path of a built index")
        pipeline = RAGPipeline.load_index(index)
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            g = pipeline.config.generation
            pipeline = RAGPipeline(pipeline.config, pipeline.retriever, pipeline.reranker,
                                   ClaudeClient(model=g.model, effort=g.effort))

    app = FastAPI(title="grounded-rag", version="0.1.0")
    app.state.pipeline = pipeline

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        log.info("%s %s %s %.1fms id=%s", request.method, request.url.path,
                 response.status_code, elapsed, request_id)
        return response

    def _generate(fn, text: str):
        p: RAGPipeline = app.state.pipeline
        if p.answerer is None:
            raise HTTPException(503, "generation is disabled: no Anthropic credentials configured")
        try:
            return fn(text).to_dict()
        except LLMRefusal as e:
            raise HTTPException(422, f"request declined: {e}") from e
        except LLMError as e:
            log.exception("generation failed")
            raise HTTPException(502, f"upstream model error: {e}") from e

    @app.get("/healthz")
    def healthz() -> dict:
        p: RAGPipeline = app.state.pipeline
        return {
            "status": "ok",
            "config": p.config.name,
            "chunks": len(p.retriever.chunks),
            "generation": p.answerer is not None,
        }

    @app.post("/v1/search")
    def search(req: SearchRequest) -> dict:
        p: RAGPipeline = app.state.pipeline
        results = p.retrieve(req.query, req.k)
        return {
            "query": req.query,
            "results": [
                {"rank": i, "chunk_id": r.chunk.id, "doc_id": r.doc_id, "title": r.chunk.title,
                 "text": r.chunk.text, "score": r.score}
                for i, r in enumerate(results, start=1)
            ],
        }

    @app.post("/v1/answer")
    def answer(req: AskRequest) -> dict:
        return _generate(app.state.pipeline.answer, req.question)

    @app.post("/v1/verify")
    def verify(req: VerifyRequest) -> dict:
        return _generate(app.state.pipeline.verify, req.claim)

    return app
