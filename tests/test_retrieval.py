import math

import pytest

from grounded.ingest.chunking import chunk_corpus
from grounded.retrieval.bm25 import BM25Retriever, RM3Params
from grounded.retrieval.dense import DenseRetriever, LSAEmbedder
from grounded.retrieval.hybrid import HybridRetriever, linear_fusion, reciprocal_rank_fusion
from grounded.types import Chunk, ScoredChunk, unique_doc_ids


def _chunks(texts):
    return [Chunk(id=f"c{i}", doc_id=f"d{i}", text=t) for i, t in enumerate(texts)]


def test_bm25_matches_reference_formula():
    chunks = _chunks(["apple banana", "apple apple cherry", "durian"])
    k1, b = 1.2, 0.75
    r = BM25Retriever(chunks, k1=k1, b=b)
    results = {s.chunk.id: s.score for s in r.retrieve("apple", 10)}

    n, avgdl = 3, (2 + 3 + 1) / 3
    idf = math.log(1 + (n - 2 + 0.5) / (2 + 0.5))

    def expected(tf, dl):
        return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

    assert results["c0"] == pytest.approx(expected(1, 2), rel=1e-5)
    assert results["c1"] == pytest.approx(expected(2, 3), rel=1e-5)
    assert "c2" not in results  # zero-score documents are not returned


def test_bm25_ranks_relevant_doc_first(docs):
    r = BM25Retriever(chunk_corpus(docs))
    assert r.retrieve("Which enzyme do statins inhibit?", 3)[0].doc_id == "statins"
    assert r.retrieve("zzzz unknown words", 3) == []


def test_bm25_rm3_still_finds_target(docs):
    r = BM25Retriever(chunk_corpus(docs), rm3=RM3Params(fb_docs=2, fb_terms=5))
    assert r.retrieve("guide RNA Cas9", 3)[0].doc_id == "crispr"


def test_lsa_dense_retrieval(docs):
    r = DenseRetriever(chunk_corpus(docs), LSAEmbedder(dim=6))
    results = r.retrieve("Cas9 guide RNA double-strand break", 3)
    assert results[0].doc_id == "crispr"
    assert all(-1.0001 <= s.score <= 1.0001 for s in results)


def _sc(cid, score=1.0):
    return ScoredChunk(Chunk(id=cid, doc_id=cid, text=""), score, "x")


def test_rrf_exact_scores():
    fused = reciprocal_rank_fusion([[_sc("a"), _sc("b")], [_sc("b"), _sc("c")]], k=60)
    scores = {s.chunk.id: s.score for s in fused}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert [s.chunk.id for s in fused] == ["b", "a", "c"]


def test_linear_fusion_normalises_scores():
    fused = linear_fusion(
        [[_sc("a", 100.0), _sc("b", 50.0)], [_sc("b", 0.9), _sc("a", 0.1)]], weights=[0.5, 0.5]
    )
    scores = {s.chunk.id: s.score for s in fused}
    assert scores["a"] == pytest.approx(0.5)
    assert scores["b"] == pytest.approx(0.5)


def test_hybrid_validates_weights(docs):
    chunks = chunk_corpus(docs)
    bm25 = BM25Retriever(chunks)
    with pytest.raises(ValueError):
        HybridRetriever([bm25], weights=[0.5, 0.5])
    with pytest.raises(ValueError):
        HybridRetriever([bm25], method="magic")


def test_hybrid_retrieval(docs):
    chunks = chunk_corpus(docs)
    h = HybridRetriever([BM25Retriever(chunks), DenseRetriever(chunks, LSAEmbedder(dim=6))])
    assert unique_doc_ids(h.retrieve("aspirin platelets thromboxane", 5))[0] == "aspirin"
