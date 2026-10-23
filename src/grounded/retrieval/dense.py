"""Dense retrieval: pluggable embedders over an exact (brute-force) cosine index.

Two embedder backends:

* ``lsa``: TF-IDF + truncated SVD fitted on the corpus. Needs no model download, so the
  whole pipeline (and CI) runs offline. It is a weak dense baseline, and the benchmarks say so.
* ``sentence-transformers``: any SentenceTransformer model (e.g. ``BAAI/bge-small-en-v1.5``).
  Install the ``neural`` extra to use it.

Exact search is the right call below ~1M vectors: a 5k x 384 float32 matmul takes microseconds.
Swap in FAISS/HNSW only when the latency budget says so.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from grounded.retrieval.base import top_k
from grounded.text import tokenize
from grounded.types import Chunk, ScoredChunk


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class LSAEmbedder:
    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        self.dim, self.seed = dim, seed
        self._vectorizer = None
        self._svd = None

    def fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True, min_df=2
        )
        x = self._vectorizer.fit_transform(texts)
        dim = min(self.dim, x.shape[1] - 1, x.shape[0] - 1)
        self._svd = TruncatedSVD(n_components=max(dim, 1), random_state=self.seed)
        self._svd.fit(x)

    def _transform(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("LSAEmbedder.fit() must be called first")
        return _l2_normalize(self._svd.transform(self._vectorizer.transform(texts))).astype(
            np.float32
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._transform(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._transform([text])[0]


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        batch_size: int = 64,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("pip install 'grounded-rag[neural]' for neural embeddings") from e
        self._model = SentenceTransformer(model)
        self.query_prefix, self.batch_size = query_prefix, batch_size

    def fit(self, texts: list[str]) -> None:  # pretrained: nothing to fit
        return None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self.query_prefix + text])[0]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("cannot index an empty corpus")
        self.chunks = chunks
        self.embedder = embedder
        texts = [c.indexed_text for c in chunks]
        embedder.fit(texts)
        self.matrix = embedder.embed_documents(texts)

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        scores = self.matrix @ self.embedder.embed_query(query)
        return [ScoredChunk(self.chunks[i], float(scores[i]), self.name) for i in top_k(scores, k)]
