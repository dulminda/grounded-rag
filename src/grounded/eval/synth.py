"""Synthetic evaluation sets for corpora that have no labelled queries.

Samples chunks, asks the LLM for one realistic question per chunk, and records the source
document as the relevant document. Synthetic questions skew towards lexical overlap with their
source, so they flatter BM25: use them to catch regressions, not to pick a retriever.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from grounded.generation.llm import LLMClient, LLMError
from grounded.generation.prompts import SYNTH_SYSTEM
from grounded.generation.schemas import SyntheticQuestion
from grounded.types import Chunk


def synthesize_examples(
    chunks: list[Chunk],
    llm: LLMClient,
    n: int,
    out_path: str | Path,
    seed: int = 0,
    min_words: int = 40,
) -> int:
    rng = random.Random(seed)
    pool = [c for c in chunks if len(c.text.split()) >= min_words]
    rng.shuffle(pool)
    written = 0
    seen: set[str] = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in pool:
            if written >= n:
                break
            prompt = f"<passage title=\"{chunk.title}\">\n{chunk.text}\n</passage>"
            try:
                q = llm.generate(SYNTH_SYSTEM, prompt, SyntheticQuestion).parsed
            except LLMError:
                continue
            key = q.question.strip().lower()
            if not q.answerable or key in seen:
                continue
            seen.add(key)
            rec = {
                "id": f"synth-{written}",
                "query": q.question,
                "relevant_doc_ids": [chunk.doc_id],
                "reference_answer": q.answer,
            }
            f.write(json.dumps(rec) + "\n")
            written += 1
    return written
