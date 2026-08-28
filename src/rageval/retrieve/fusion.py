"""Combine ranked lists from retrievers that do not share a score scale.

Cosine similarity lives in [-1, 1] and saturates near the top; BM25 is
unbounded and depends on corpus statistics. Min-max normalising them onto a
common scale makes the blend depend on the depth of the candidate pool and
on whichever single outlier happens to set the maximum - two knobs nobody
declared, quietly deciding the result.

Reciprocal rank fusion avoids the question entirely by discarding the
scores and keeping only the ranks:

    score(d) = sum over lists of  1 / (k + rank(d))

k is fixed at 60, the value from Cormack, Clarke and Buettcher (2009). It
is not tuned here: sweep 1 established this gold set cannot resolve a
four-point difference, so spending its power on a fusion constant would buy
a number with no meaning. Large k flattens the lists toward equal weight;
small k lets a single first-place finish dominate.

Like `bm25.py`, this module knows nothing about the store - records are
duck-typed on `.chunk_id` and `.rank` and copied with
`dataclasses.replace`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(rankings: Sequence[Sequence], k: int = DEFAULT_RRF_K,
                           top_k: int | None = None) -> list:
    """Fuse ranked lists into one, best first.

    Each input list must already be ranked, with `.rank` counting from 1
    WITHIN that list - a global rank would make the fusion meaningless.
    """
    if k <= 0:
        raise ValueError("rrf k must be positive")

    scores: dict[str, float] = {}
    first_seen: dict[str, object] = {}
    for ranked in rankings:
        for rec in ranked:
            cid = rec.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rec.rank)
            first_seen.setdefault(cid, rec)

    # Tie-break on chunk_id, not on insertion order: two documents with
    # identical fused scores must not swap places because a dict happened
    # to be built in a different order.
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        ordered = ordered[:top_k]

    return [replace(first_seen[cid], score=score, rank=rank)
            for rank, (cid, score) in enumerate(ordered, 1)]


def provenance(fused: Sequence, sources: dict[str, Sequence]) -> dict[str, int]:
    """How many of the fused results each source contributed, and how many
    both found.

    Without this, a hybrid arm that beats dense is uninterpretable: the gain
    could be lexical recall, or it could be the fusion reordering documents
    dense already had. Counting the overlap separates those.
    """
    ids = {name: {r.chunk_id for r in ranked} for name, ranked in sources.items()}
    names = list(ids)
    counts = {f"{n} only": 0 for n in names}
    counts["both"] = 0
    for rec in fused:
        found = [n for n in names if rec.chunk_id in ids[n]]
        if len(found) == 1:
            counts[f"{found[0]} only"] += 1
        elif len(found) > 1:
            counts["both"] += 1
    return counts