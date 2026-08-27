"""Retrieval metrics scored by character-span overlap.

The gold set anchors evidence to (arxiv_id, char_start, char_end) rather
than to chunk IDs, because chunk IDs are not stable across the six chunking
strategies being ablated. So relevance here is a geometric question: does a
retrieved chunk share any characters with an evidence span from the same
paper?

Two recall-style measures are reported, and they are not redundant:

    Recall@k    was ANY evidence span retrieved
    Coverage@k  was EVERY evidence span retrieved

They are identical for single-evidence questions and diverge on
multi-evidence ones. For multi_hop that divergence is the whole story: a
system that finds the terminal passage but never the referring sentence
scores well on Recall and fails Coverage, and only Coverage exposes it.

Metrics are also reported at a character BUDGET, not only at a rank cutoff.
Comparing chunkings at fixed k is confounded: smaller chunks mean more of
them, so top-10 delivers a quarter of the text at 256 tokens that it does at
1024. A generator is limited by context, not by document count, so the
budget view is the decision-relevant one - and it handles variable-size
chunkers natively, where multiplying a median by k does not.

This module is deliberately free of any dependency on the retriever, the
vector store, or pandas, so the metrics can be tested against hand-computed
cases without building an index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

# Report at these cutoffs unless told otherwise.
DEFAULT_KS = (1, 3, 5, 10, 20)

# Character budgets, chosen to bracket the baseline: 512-token chunks at k=10
# is roughly 21k characters, so 20_000 is the like-for-like comparison point.
DEFAULT_BUDGETS = (5_000, 10_000, 20_000, 40_000)


@dataclass(frozen=True)
class Retrieved:
    """One retrieved unit, reduced to what scoring needs."""

    arxiv_id: str
    char_start: int
    char_end: int
    score: float = 0.0

    @classmethod
    def from_chunk(cls, chunk, score: float = 0.0) -> "Retrieved":
        """Adapt anything with arxiv_id/char_start/char_end (e.g. Chunk)."""
        return cls(
            arxiv_id=chunk.arxiv_id,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            score=score,
        )

    def overlaps(self, arxiv_id: str, start: int, end: int) -> bool:
        """Half-open overlap, matching Chunk.overlaps.

        A chunk ending exactly where a span begins does NOT overlap it.
        """
        if self.arxiv_id != arxiv_id:
            return False
        return self.char_start < end and start < self.char_end


@dataclass(frozen=True)
class Span:
    """One evidence span from the gold set."""

    arxiv_id: str
    char_start: int
    char_end: int

    @classmethod
    def from_evidence(cls, ev) -> "Span":
        return cls(ev.arxiv_id, ev.char_start, ev.char_end)


def spans_from_question(question) -> list[Span]:
    """Evidence spans of a GoldQuestion. Empty for unanswerable questions."""
    return [Span.from_evidence(e) for e in getattr(question, "evidence", [])]


def is_scorable(question) -> bool:
    """Unanswerable questions carry no evidence and have no retrieval target.

    They are excluded from every metric here rather than scored as zero: a
    0.0 in a results table reads as a failure, and these are not failures.
    They are evaluated in the generation phase instead.
    """
    return bool(spans_from_question(question))


def relevance_flags(retrieved: Sequence[Retrieved], spans: Sequence[Span]) -> list[bool]:
    """Per-rank relevance: does this chunk touch any evidence span?"""
    return [
        any(r.overlaps(s.arxiv_id, s.char_start, s.char_end) for s in spans)
        for r in retrieved
    ]


def spans_hit(retrieved: Sequence[Retrieved], spans: Sequence[Span]) -> set[int]:
    """Indices of evidence spans touched by at least one retrieved chunk."""
    return {
        i
        for i, s in enumerate(spans)
        if any(r.overlaps(s.arxiv_id, s.char_start, s.char_end) for r in retrieved)
    }


def recall_at_k(retrieved: Sequence[Retrieved], spans: Sequence[Span], k: int) -> float:
    """1.0 if any evidence span was retrieved within the top k."""
    if not spans:
        return float("nan")
    return 1.0 if spans_hit(retrieved[:k], spans) else 0.0


def coverage_at_k(retrieved: Sequence[Retrieved], spans: Sequence[Span], k: int) -> float:
    """Fraction of the question's evidence spans retrieved within the top k.

    Reported as a fraction rather than a 0/1 flag so a two-span question
    that finds one span is visibly half-solved rather than simply failed.
    """
    if not spans:
        return float("nan")
    return len(spans_hit(retrieved[:k], spans)) / len(spans)


def reciprocal_rank(retrieved: Sequence[Retrieved], spans: Sequence[Span]) -> float:
    """1/rank of the first relevant chunk, or 0.0 if none is relevant."""
    for i, flag in enumerate(relevance_flags(retrieved, spans)):
        if flag:
            return 1.0 / (i + 1)
    return 0.0


def _dcg(gains: Iterable[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(
    retrieved: Sequence[Retrieved],
    spans: Sequence[Span],
    k: int,
    n_relevant_total: int | None = None,
) -> float:
    """Binary-gain nDCG@k.

    `n_relevant_total` is the number of chunks in the whole index that
    overlap this question's evidence. Pass it whenever it is known: the
    ideal ranking puts min(n_relevant_total, k) relevant chunks first, and
    without it the ideal is inferred from what was actually retrieved,
    which flatters a run that retrieved nothing relevant.
    """
    if not spans:
        return float("nan")
    gains = [1.0 if f else 0.0 for f in relevance_flags(retrieved[:k], spans)]
    ideal_n = int(sum(gains)) if n_relevant_total is None else n_relevant_total
    ideal_n = min(ideal_n, k)
    if ideal_n <= 0:
        return 0.0
    return _dcg(gains) / _dcg([1.0] * ideal_n)


def count_relevant_chunks(chunks: Iterable, spans: Sequence[Span]) -> int:
    """How many chunks in the index overlap this question's evidence.

    Needed for an honest IDCG, and diagnostic in its own right: if one
    evidence span is covered by eight chunks, Recall@5 is easy for reasons
    that have nothing to do with retrieval quality.
    """
    total = 0
    for c in chunks:
        r = Retrieved.from_chunk(c)
        if any(r.overlaps(s.arxiv_id, s.char_start, s.char_end) for s in spans):
            total += 1
    return total


def take_within_budget(retrieved: Sequence[Retrieved], budget: int) -> list[Retrieved]:
    """Chunks in rank order that fit in `budget` characters.

    Packing rule: add chunks until the next one would overflow, but always
    take at least the first - a real system truncates an oversized leading
    chunk rather than returning nothing.
    """
    taken: list[Retrieved] = []
    used = 0
    for r in retrieved:
        length = r.char_end - r.char_start
        if taken and used + length > budget:
            break
        taken.append(r)
        used += length
    return taken


def recall_at_budget(retrieved, spans, budget: int) -> float:
    if not spans:
        return float("nan")
    return 1.0 if spans_hit(take_within_budget(retrieved, budget), spans) else 0.0


def coverage_at_budget(retrieved, spans, budget: int) -> float:
    if not spans:
        return float("nan")
    got = spans_hit(take_within_budget(retrieved, budget), spans)
    return len(got) / len(spans)


def evaluate_question(
    question,
    retrieved: Sequence[Retrieved],
    ks: Sequence[int] = DEFAULT_KS,
    n_relevant_total: int | None = None,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
) -> dict:
    """Metrics for one question. Returns a flat row ready for a CSV."""
    spans = spans_from_question(question)
    row: dict = {
        "id": question.id,
        "question_type": question.question_type,
        "difficulty": getattr(question, "difficulty", ""),
        "n_evidence": len(spans),
        "n_papers": len({s.arxiv_id for s in spans}),
        "n_relevant_chunks": n_relevant_total if n_relevant_total is not None else "",
        "n_retrieved": len(retrieved),
    }
    if not spans:
        row["scorable"] = False
        return row

    row["scorable"] = True
    row["mrr"] = reciprocal_rank(retrieved, spans)
    row["first_hit_rank"] = next(
        (i + 1 for i, f in enumerate(relevance_flags(retrieved, spans)) if f), None
    )
    for k in ks:
        hit = spans_hit(retrieved[:k], spans)
        row[f"recall@{k}"] = recall_at_k(retrieved, spans, k)
        row[f"coverage@{k}"] = coverage_at_k(retrieved, spans, k)
        row[f"ndcg@{k}"] = ndcg_at_k(retrieved, spans, k, n_relevant_total)
        # Which spans, not just how many. For multi_hop, evidence[0] is the
        # source paper carrying the referring sentence and evidence[1] is the
        # terminal - so this distinguishes "the chain broke" from "the chain
        # broke at the first hop", which the coverage fraction cannot.
        row[f"hit_spans@{k}"] = ";".join(str(i) for i in sorted(hit))

    # Budget view. `k@B` records how many chunks actually fitted, which is
    # the number that makes two chunkings comparable.
    for b in budgets:
        taken = take_within_budget(retrieved, b)
        got = spans_hit(taken, spans)
        row[f"recall@B{b}"] = 1.0 if got else 0.0
        row[f"coverage@B{b}"] = len(got) / len(spans)
        row[f"k@B{b}"] = len(taken)
        row[f"hit_spans@B{b}"] = ";".join(str(i) for i in sorted(got))
    return row


def aggregate(rows: Sequence[dict], ks: Sequence[int] = DEFAULT_KS,
              budgets: Sequence[int] = DEFAULT_BUDGETS) -> dict:
    """Mean of each metric over scorable rows, plus counts."""
    scored = [r for r in rows if r.get("scorable")]
    out: dict = {"n_questions": len(rows), "n_scored": len(scored)}
    if not scored:
        return out
    keys = ["mrr"] + [f"{m}@{k}" for k in ks for m in ("recall", "coverage", "ndcg")]
    keys += [f"{m}@B{b}" for b in budgets for m in ("recall", "coverage", "k")]
    for key in keys:
        vals = [r[key] for r in scored if key in r]
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    return out


def aggregate_by_type(rows: Sequence[dict], ks: Sequence[int] = DEFAULT_KS,
                      budgets: Sequence[int] = DEFAULT_BUDGETS) -> dict[str, dict]:
    """Aggregates per question_type, in a stable order."""
    order = ["factual", "comparative", "multi_hop", "ambiguous", "unanswerable"]
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["question_type"], []).append(r)
    return {t: aggregate(by_type[t], ks, budgets) for t in order if t in by_type}