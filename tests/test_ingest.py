import pytest

from grounded.ingest.chunking import chunk_document
from grounded.ingest.loaders import load_corpus
from grounded.text import split_sentences, tokenize
from grounded.types import Document


def test_tokenize_stems_and_drops_stopwords():
    tokens = tokenize("The inhibitors are inhibiting the enzymes")
    assert tokens == ["inhibitor", "inhibit", "enzym"]
    assert tokenize("COX-1", stem=False) == ["cox-1"]


def test_split_sentences():
    text = "Aspirin works. It blocks COX-1! Does it? Yes (mostly)."
    expected = ["Aspirin works.", "It blocks COX-1!", "Does it?", "Yes (mostly)."]
    assert split_sentences(text) == expected


def _doc(n_sentences: int, words_per_sentence: int = 10) -> Document:
    sentences = [" ".join([f"w{i}"] * words_per_sentence) + "." for i in range(n_sentences)]
    return Document(id="d", text=" ".join(sentences), metadata={"sentences": sentences})


def test_chunks_respect_budget_and_never_split_sentences():
    chunks = chunk_document(_doc(10), max_words=30, overlap_sentences=1)
    for c in chunks:
        assert len(c.text.split()) <= 30
        assert c.text.endswith(".")
    # Every sentence appears in at least one chunk.
    covered = {s for c in chunks for s in range(*c.metadata["sentence_span"])}
    assert covered == set(range(10))


def test_overlap_repeats_last_sentence():
    chunks = chunk_document(_doc(6), max_words=30, overlap_sentences=1)
    spans = [c.metadata["sentence_span"] for c in chunks]
    assert spans[0] == [0, 3]
    assert spans[1][0] == 2  # starts on the last sentence of the previous chunk


def test_oversized_sentence_becomes_its_own_chunk_and_terminates():
    chunks = chunk_document(_doc(3, words_per_sentence=50), max_words=20, overlap_sentences=2)
    assert [c.metadata["sentence_span"] for c in chunks] == [[0, 1], [1, 2], [2, 3]]


def test_chunk_ids_are_stable_and_unique():
    chunks = chunk_document(_doc(10), max_words=30)
    assert [c.id for c in chunks] == [f"d#{i}" for i in range(len(chunks))]


def test_invalid_chunk_params():
    with pytest.raises(ValueError):
        chunk_document(_doc(2), max_words=0)


def test_load_text_dir(tmp_path):
    (tmp_path / "a.md").write_text("# Runbook\n\nRestart the service.")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("Plain notes.")
    docs = {d.id: d for d in load_corpus(tmp_path)}
    assert docs["a.md"].title == "Runbook"
    assert docs["sub/b.txt"].title == "b"
