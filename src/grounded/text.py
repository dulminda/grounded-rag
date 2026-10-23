"""Tokenisation and sentence splitting shared by the lexical and LSA retrievers."""

from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
# Split on sentence-final punctuation followed by whitespace and an uppercase letter/digit.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

# A compact English stopword list (Lucene's default set plus a few high-frequency extras).
STOPWORDS = frozenset(
    """a an and are as at be but by for if in into is it no not of on or such that the their
    then there these they this to was will with from has have had were been being which who
    whom what when where why how all any both each few more most other some than too very can
    do does did its our we you your he she his her them i me my so also""".split()
)

_stemmer = snowballstemmer.stemmer("english")


@lru_cache(maxsize=200_000)
def _stem(token: str) -> str:
    return _stemmer.stemWord(token)


def tokenize(text: str, *, stem: bool = True, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [_stem(t) for t in tokens]
    return tokens


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]
