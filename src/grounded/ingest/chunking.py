"""Sentence-aware chunking with overlap.

Chunks never split a sentence, so citations can always be resolved back to whole sentences.
A single sentence longer than ``max_words`` becomes its own chunk rather than being cut.
"""

from __future__ import annotations

from collections.abc import Iterable

from grounded.text import split_sentences
from grounded.types import Chunk, Document


def chunk_document(doc: Document, max_words: int = 200, overlap_sentences: int = 1) -> list[Chunk]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    sentences = doc.metadata.get("sentences") or split_sentences(doc.text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(sentences):
        end = start
        words = 0
        while end < len(sentences):
            n = len(sentences[end].split())
            if end > start and words + n > max_words:
                break
            words += n
            end += 1
        chunks.append(
            Chunk(
                id=f"{doc.id}#{len(chunks)}",
                doc_id=doc.id,
                text=" ".join(sentences[start:end]),
                title=doc.title,
                position=len(chunks),
                metadata={"sentence_span": [start, end]},
            )
        )
        if end >= len(sentences):
            break
        # Step back by the overlap, but always make forward progress.
        start = max(end - overlap_sentences, start + 1)
    return chunks


def chunk_corpus(
    docs: Iterable[Document], max_words: int = 200, overlap_sentences: int = 1
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, max_words=max_words, overlap_sentences=overlap_sentences))
    return out
