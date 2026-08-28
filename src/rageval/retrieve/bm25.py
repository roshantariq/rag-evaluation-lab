"""Lexical retrieval over the same chunks the dense index holds.

The Phase 6 miss audit found questions that quote their target almost
verbatim and are still never retrieved: f013 names `xLSTM` and `ConvLSTM`,
m010 carries "9.4 percent" and `ECMWF-IFS`, f015 turns on `B-MSE`. A
22M-parameter sentence encoder averages those rare tokens into 256 word
pieces of surrounding prose. BM25 does the opposite - a term almost nothing
else contains carries almost all the weight.

Deliberately free of any dependency on the vector store: records are
duck-typed on `.text` and `.chunk_id` and updated with
`dataclasses.replace`, so the whole module is testable with a five-line
stub, no index, no model download and no ChromaDB. Same reason
`retrieval_metrics.py` has no retriever import.

TOKENIZATION IS THE EXPERIMENT. The pre-registered prediction for this
sweep names four questions that turn on four specific strings. Whether
`B-MSE` survives as a token, becomes `b` + `mse`, or is destroyed by an
attached comma decides that prediction on its own. So the rule is stated,
not left to a default:

    a compound emits itself AND its parts

Compounds are formed by hyphen, underscore and slash, and by case
transitions. Parts are kept only when they are at least two characters and
contain a letter - so `B-MSE` gives {b-mse, mse} rather than a bare `b`,
and `Z500` stays whole rather than leaking a `500` that would match every
"500 epochs" in the corpus. `tests/test_bm25.py` pins the behaviour on all
four strings.

No stopword list. BM25's IDF already drives ubiquitous terms to near zero,
and a hand-written list would be one more undeclared choice affecting the
outcome. k1 and b stay at the rank_bm25 defaults (1.5, 0.75) for the same
reason: this gold set has no statistical power to spend tuning them, as
sweep 1 established.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# A word starts alphanumeric and may carry internal ., _, / or -, so
# "B-MSE", "9.4", "km/h" and "state-of-the-art" survive as single units.
# Trailing punctuation is stripped afterwards: "levels." must not become a
# token distinct from "levels".
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/\-]*")
_JOINER = re.compile(r"[-_/]+")
# Case-transition splitter: an all-caps run, a capitalised word, a lowercase
# run, or a digit run. "ConvLSTM" -> Conv, LSTM.  "xLSTM" -> x, LSTM.
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def sub_tokens(raw: str) -> list[str]:
    """The parts of a compound, lowercased.

    Kept only if at least two characters and containing a letter. The
    letter requirement is what stops `Z500` from emitting `500`, which
    would match every parameter count and epoch number in the corpus.
    """
    parts: list[str] = []
    for piece in _JOINER.split(raw):
        if not piece:
            continue
        for m in _CAMEL.findall(piece):
            if len(m) >= 2 and any(ch.isalpha() for ch in m):
                parts.append(m.lower())
    return parts


def tokenize(text: str) -> list[str]:
    """Text to BM25 terms: every compound plus its parts, lowercased."""
    out: list[str] = []
    for m in _WORD.finditer(text):
        raw = m.group(0).strip("._/-")
        if not raw:
            continue
        whole = raw.lower()
        out.append(whole)
        for part in sub_tokens(raw):
            if part != whole:
                out.append(part)
    return out


class BM25Retriever:
    """BM25 Okapi over a fixed corpus of chunk records.

    `records` must be dataclass instances carrying `.text` and `.chunk_id`;
    `query` returns copies with `score` and `rank` filled in, so the result
    is interchangeable with the dense store's hits.
    """

    def __init__(self, records: Sequence, k1: float = DEFAULT_K1,
                 b: float = DEFAULT_B):
        from rank_bm25 import BM25Okapi  # imported here so the module loads

        self.records = list(records)
        if not self.records:
            raise ValueError("BM25Retriever needs a non-empty corpus")
        corpus = [tokenize(r.text) for r in self.records]
        empty = sum(1 for toks in corpus if not toks)
        if empty:
            logger.warning(
                "  %d/%d chunks tokenize to nothing and can never be "
                "retrieved lexically", empty, len(corpus))
        self.k1, self.b = k1, b
        self.bm25 = BM25Okapi(corpus, k1=k1, b=b)

    def __len__(self) -> int:
        return len(self.records)

    def query(self, text: str, k: int = 10) -> list:
        toks = tokenize(text)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        # Stable sort so equal scores fall in corpus order rather than an
        # order that changes between runs.
        order = np.argsort(-scores, kind="stable")[: max(k, 0)]
        out = []
        for rank, i in enumerate(order, 1):
            s = float(scores[i])
            # BM25 scores go to zero with no shared term, and negative for
            # a term present in more than half the corpus. Returning those
            # would hand the fusion an arbitrary ranking of non-matches.
            if s <= 0.0:
                break
            out.append(replace(self.records[int(i)], score=s, rank=rank))
        return out