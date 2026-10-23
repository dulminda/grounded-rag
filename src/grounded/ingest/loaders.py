"""Corpus loaders: JSONL documents and directories of text/markdown files."""

from __future__ import annotations

import json
from pathlib import Path

from grounded.types import Document


def load_jsonl_corpus(path: str | Path) -> list[Document]:
    """Load ``{"id", "text", "title"?, "metadata"?}`` records, one per line."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if "id" not in rec or "text" not in rec:
                raise ValueError(f"{path}:{line_no}: record needs 'id' and 'text'")
            docs.append(
                Document(
                    id=str(rec["id"]),
                    text=rec["text"],
                    title=rec.get("title", ""),
                    metadata=rec.get("metadata", {}),
                )
            )
    return docs


def load_text_dir(
    path: str | Path, patterns: tuple[str, ...] = ("*.md", "*.txt")
) -> list[Document]:
    """Load every matching file under ``path``; the first markdown heading becomes the title."""
    root = Path(path)
    docs = []
    for pattern in patterns:
        for file in sorted(root.rglob(pattern)):
            text = file.read_text(encoding="utf-8")
            title = file.stem
            for line in text.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            docs.append(
                Document(id=str(file.relative_to(root)), text=text, title=title,
                         metadata={"path": str(file)})
            )
    return docs


def load_corpus(path: str | Path) -> list[Document]:
    p = Path(path)
    if p.is_dir():
        return load_text_dir(p)
    if p.suffix == ".jsonl":
        return load_jsonl_corpus(p)
    raise ValueError(f"Unsupported corpus path {p}: expected a directory or a .jsonl file")
