"""Evaluation datasets.

SciFact (Wadden et al., 2020) gives us three kinds of ground truth from one dataset:

* retrieval: ``cited_doc_ids`` per claim, the same qrels BEIR uses for its SciFact test split;
* end-to-end verdicts: SUPPORT / CONTRADICT / NOT_ENOUGH_INFO per claim;
* citations: the gold evidence documents (and sentences) behind each verdict.

Custom corpora use a generic JSONL format (see ``load_jsonl_examples``), which
``grounded synth`` can produce from your own documents.
"""

from __future__ import annotations

import json
import tarfile
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from grounded.types import Document

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
VERDICTS = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")


@dataclass(frozen=True)
class EvalExample:
    id: str
    query: str
    relevant_doc_ids: frozenset[str]
    label: str | None = None  # gold verdict, for claim-verification datasets
    evidence: dict[str, list[int]] = field(default_factory=dict)  # doc_id -> sentence indices
    reference_answer: str | None = None


def download_scifact(dest: str | Path) -> Path:
    dest = Path(dest)
    if (dest / "corpus.jsonl").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "data.tar.gz"
    urllib.request.urlretrieve(SCIFACT_URL, archive)  # noqa: S310 - fixed https URL
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            # Only the flat release files; skip cross-validation folds and AppleDouble files.
            if member.isfile() and len(name.parts) == 2 and not name.name.startswith("._"):
                member.name = name.name
                tar.extract(member, dest, filter="data")
    archive.unlink()
    return dest


def load_scifact_corpus(root: str | Path) -> list[Document]:
    docs = []
    with open(Path(root) / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sentences = [s.strip() for s in rec["abstract"]]
            docs.append(
                Document(
                    id=str(rec["doc_id"]),
                    title=rec["title"],
                    text=" ".join(sentences),
                    # Keep the gold sentence segmentation so evidence indices line up.
                    metadata={"sentences": sentences},
                )
            )
    return docs


def _claim_label(evidence: dict) -> str:
    labels = Counter(e["label"] for doc in evidence.values() for e in doc)
    if not labels:
        return "NOT_ENOUGH_INFO"
    return labels.most_common(1)[0][0]


def load_scifact_examples(root: str | Path, split: str = "dev") -> list[EvalExample]:
    """``split`` is ``train`` or ``dev``; the official test split has no public labels."""
    if split not in {"train", "dev"}:
        raise ValueError("SciFact split must be 'train' or 'dev' (test labels are hidden)")
    examples = []
    with open(Path(root) / f"claims_{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            evidence = {
                str(doc_id): sorted({s for e in ev for s in e["sentences"]})
                for doc_id, ev in rec["evidence"].items()
            }
            examples.append(
                EvalExample(
                    id=str(rec["id"]),
                    query=rec["claim"],
                    relevant_doc_ids=frozenset(str(d) for d in rec["cited_doc_ids"]),
                    label=_claim_label(rec["evidence"]),
                    evidence=evidence,
                )
            )
    return examples


def load_jsonl_examples(path: str | Path) -> list[EvalExample]:
    """``{"id", "query", "relevant_doc_ids": [...], "reference_answer"?, "label"?}`` per line."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out.append(
                EvalExample(
                    id=str(rec["id"]),
                    query=rec["query"],
                    relevant_doc_ids=frozenset(str(d) for d in rec.get("relevant_doc_ids", [])),
                    label=rec.get("label"),
                    reference_answer=rec.get("reference_answer"),
                )
            )
    return out
