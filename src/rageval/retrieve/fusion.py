"""Combine ranked lists from retrievers that do not share a score scale.

Cosine similarity lives in [-1, 1] and saturates near the top; BM25 is
unbounded and depends on corpus statistics. Min-max normalising them onto a
common scale makes the blend depend on the depth of the candidate pool and
on whichever single outlier happens to set the maximum - two knobs nobody
declared, quietly deciding the result.

Reciprocal rank fusion avoids the question entirely by discarding the
scores and keeping only the ranks:

    score(d) = sum over lists of  1 / (k + rank(d))

WHAT k DOES, measured rather than assumed. k controls how much presence in
both lists is worth relative to a first-place finish in one. At k=60 (the
value from Cormack, Clarke and Buettcher 2009) a document ranked 50th by
both scores 2/110 = 0.0182, while a document ranked FIRST by one and absent
from the other scores 1/61 = 0.0164 - so agreement beats excellence and the
fused head contains only documents both retrievers already had. Sweep 2
measured exactly that: 100.0% of the fused top 10 was "both", 0.0% unique
to either arm, and the fusion lost a question BM25 alone had recovered.

That constant was calibrated for fusing many similar-quality TREC runs,
where agreement is the useful signal. Fusing two retrievers with
near-disjoint failure modes inverts the situation: the value is the
complementary recall, and k=60 discards it.

`interleave` is the control for all of this - see its docstring.
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


def interleave(rankings: Sequence[Sequence], top_k: int | None = None) -> list:
    """Strict round-robin over ranked lists, skipping documents already taken.

    THE CONTROL ARM. RRF has a constant, and once k=60 was found to fail,
    every replacement value is a post-hoc choice made after seeing which
    way the results went. Interleaving has nothing to tune: each retriever
    simply nominates its next unseen document in turn. If it performs like
    a tuned RRF, then the conclusion "fusion helps once it stops behaving
    as an intersection" does not rest on having picked a good constant -
    and if it does not, the tuned result should be read as tuning.

    It also guarantees by construction what k=60 destroyed: each
    retriever's rank-1 document is in the fused top 2, so unique recall can
    never be filtered out.

    Order of `rankings` decides only which retriever gets position 1. That
    is an arbitrary choice affecting a single rank, so it is made
    explicitly by the caller rather than hidden here.
    """
    lists = [list(r) for r in rankings]
    seen: set[str] = set()
    out: list = []
    depth = 0
    longest = max((len(lst) for lst in lists), default=0)
    while depth < longest:
        for lst in lists:
            if depth >= len(lst):
                continue                      # this list is exhausted; skip it
            rec = lst[depth]
            if rec.chunk_id in seen:
                continue                      # the other list already offered it
            seen.add(rec.chunk_id)
            out.append(rec)
            if top_k is not None and len(out) >= top_k:
                return _renumber(out)
        depth += 1
    return _renumber(out)


def _renumber(records: Sequence) -> list:
    """Stamp 1-based ranks and a monotone decreasing score.

    The score is 1/rank rather than anything meaningful: interleaving
    produces an order, not a similarity. Making it monotone keeps any
    downstream code that sorts by score in agreement with the order here.
    """
    return [replace(rec, score=1.0 / rank, rank=rank)
            for rank, rec in enumerate(records, 1)]


def provenance(fused: Sequence, sources: dict[str, Sequence]) -> dict[str, int]:
    """How many of the fused results each source contributed, and how many
    both found.

    Without this, a hybrid arm that beats dense is uninterpretable: the gain
    could be lexical recall, or it could be the fusion reordering documents
    dense already had. Counting the overlap separates those - and it is what
    exposed the k=60 intersection behaviour.
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