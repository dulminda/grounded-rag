"""Typed pipeline configuration, loaded from YAML. One file = one system under test."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkingConfig(_Strict):
    max_words: int = 200
    overlap_sentences: int = 1


class RM3Config(_Strict):
    fb_docs: int = 10
    fb_terms: int = 10
    original_weight: float = 0.5


class BM25Config(_Strict):
    k1: float = 0.9
    b: float = 0.4
    rm3: RM3Config | None = None


class DenseConfig(_Strict):
    backend: Literal["lsa", "sentence-transformers"] = "lsa"
    model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    dim: int = 256  # LSA only


class FusionConfig(_Strict):
    method: Literal["rrf", "linear"] = "rrf"
    weights: list[float] | None = None  # [bm25, dense]
    rrf_k: int = 60
    candidates: int = 100


class RerankConfig(_Strict):
    backend: Literal["none", "cross-encoder"] = "none"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates: int = 50


class RetrievalConfig(_Strict):
    method: Literal["bm25", "dense", "hybrid"] = "bm25"
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)


class GenerationConfig(_Strict):
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    context_chunks: int = 5
    max_source_chars: int = 2500


class PipelineConfig(_Strict):
    name: str = "default"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        data = yaml.safe_load(Path(path).read_text()) or {}
        data.setdefault("name", Path(path).stem)
        return cls.model_validate(data)
