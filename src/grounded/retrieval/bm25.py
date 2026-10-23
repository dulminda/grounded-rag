"""Okapi BM25 over a sparse matrix, with optional RM3 pseudo-relevance feedback.

Term weights are precomputed once at index time into a CSC matrix W (docs x vocab), so a query
is a sparse column gather plus a weighted sum: ``scores = W[:, q] @ w_q``. On SciFact (5k docs)
this answers a query in well under a millisecond.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from grounded.retrieval.base import top_k
from grounded.text import tokenize
from grounded.types import Chunk, ScoredChunk


@dataclass(frozen=True)
class RM3Params:
    fb_docs: int = 10
    fb_terms: int = 10
    original_weight: float = 0.5


class BM25Retriever:
    name = "bm25"

    def __init__(
        self,
        chunks: list[Chunk],
        k1: float = 0.9,
        b: float = 0.4,
        rm3: RM3Params | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("cannot index an empty corpus")
        self.chunks = chunks
        self.k1, self.b, self.rm3 = k1, b, rm3

        docs_tokens = [tokenize(c.indexed_text) for c in chunks]
        vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        for i, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                j = vocab.setdefault(term, len(vocab))
                rows.append(i)
                cols.append(j)
                tfs.append(tf)
        self.vocab = vocab
        n_docs, n_terms = len(chunks), len(vocab)

        tf = sparse.csr_matrix(
            (np.asarray(tfs, dtype=np.float32), (rows, cols)), shape=(n_docs, n_terms)
        )
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(doc_len.mean()) or 1.0
        df = np.bincount(tf.indices, minlength=n_terms)
        # Lucene's non-negative IDF variant.
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # Saturated, length-normalised TF, computed only on the non-zeros.
        norm = (k1 * (1.0 - b + b * doc_len / avgdl)).astype(np.float32)
        row_of_nnz = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        data = tf.data * (k1 + 1.0) / (tf.data + norm[row_of_nnz])
        data *= self.idf[tf.indices]
        self._w = sparse.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape).tocsc()
        # Row-normalised TF for RM3's relevance model.
        self._tf = tf
        self._doc_len = doc_len

    def _score(self, weights: dict[int, float]) -> np.ndarray:
        if not weights:
            return np.zeros(len(self.chunks), dtype=np.float32)
        cols = np.fromiter(weights.keys(), dtype=np.int64)
        w = np.fromiter(weights.values(), dtype=np.float32)
        return np.asarray(self._w[:, cols] @ w).ravel()

    def _query_weights(self, query: str) -> dict[int, float]:
        counts = Counter(t for t in tokenize(query) if t in self.vocab)
        return {self.vocab[t]: float(c) for t, c in counts.items()}

    def _expand_rm3(self, weights: dict[int, float], scores: np.ndarray) -> dict[int, float]:
        p = self.rm3
        assert p is not None
        fb = top_k(scores, p.fb_docs)
        fb = fb[scores[fb] > 0]
        if fb.size == 0:
            return weights
        # Relevance model: P(t|R) ~ sum_d P(t|d) * P(q|d), with P(q|d) ~ softmax of BM25 score.
        doc_w = np.exp(scores[fb] - scores[fb].max())
        doc_w /= doc_w.sum()
        tf = self._tf[fb].multiply(1.0 / self._doc_len[fb, None]).tocsr()
        term_w = np.asarray(tf.T @ doc_w).ravel()
        expansion = top_k(term_w, p.fb_terms)
        exp_norm = term_w[expansion].sum() or 1.0
        q_norm = sum(weights.values()) or 1.0

        merged: dict[int, float] = {
            j: p.original_weight * w / q_norm for j, w in weights.items()
        }
        for j in expansion:
            merged[int(j)] = merged.get(int(j), 0.0) + (1 - p.original_weight) * (
                float(term_w[j]) / exp_norm
            )
        return merged

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        weights = self._query_weights(query)
        scores = self._score(weights)
        if self.rm3 is not None:
            scores = self._score(self._expand_rm3(weights, scores))
        idx = top_k(scores, k)
        return [
            ScoredChunk(self.chunks[i], float(scores[i]), self.name) for i in idx if scores[i] > 0
        ]
