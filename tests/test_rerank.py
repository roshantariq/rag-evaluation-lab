"""Tests for the cross-encoder reranking stage, with no model download.

The model is replaced by a scoring function, so these test only what this
project's code does with scores: order, truncation to top_k, tie handling,
and that provenance survives. Whether the real model scores well is the
sweep's question, not a unit test's.
"""

from __future__ import annotations

import pytest

from rageval.retrieve.rerank import Reranker
from rageval.store.chroma_store import Retrieved


def rec(cid: str, text: str, rank: int) -> Retrieved:
    return Retrieved(chunk_id=cid, arxiv_id="2101.00001", text=text,
                     char_start=rank * 100, char_end=rank * 100 + 50,
                     section="", title="", score=0.0, rank=rank)


def by_length(pairs):
    """Longer passage = higher score. Deterministic and easy to reason about."""
    return [float(len(p)) for _, p in pairs]


CANDS = [rec("a", "x", 1), rec("b", "xxx", 2), rec("c", "xx", 3)]


def test_orders_by_score_and_renumbers_ranks():
    out = Reranker("stub", max_length=1024, score_fn=by_length).rerank("q", CANDS)
    assert [h.chunk_id for h in out] == ["b", "c", "a"]
    assert [h.rank for h in out] == [1, 2, 3]
    assert [h.score for h in out] == [3.0, 2.0, 1.0]


def test_top_k_truncates_after_reordering():
    out = Reranker("stub", 1024, score_fn=by_length).rerank("q", CANDS, top_k=2)
    assert [h.chunk_id for h in out] == ["b", "c"]


def test_ties_keep_first_stage_order():
    flat = Reranker("stub", 1024, score_fn=lambda pairs: [0.5] * len(pairs))
    assert [h.chunk_id for h in flat.rerank("q", CANDS)] == ["a", "b", "c"]


def test_provenance_is_preserved():
    out = Reranker("stub", 1024, score_fn=by_length).rerank("q", CANDS)
    b = next(h for h in out if h.chunk_id == "b")
    assert (b.arxiv_id, b.char_start, b.char_end) == ("2101.00001", 200, 250)


def test_question_reaches_the_scorer():
    seen = []

    def spy(pairs):
        seen.extend(pairs)
        return [0.0] * len(pairs)

    Reranker("stub", 1024, score_fn=spy).rerank("what is CRPS?", CANDS)
    assert {q for q, _ in seen} == {"what is CRPS?"}
    assert [p for _, p in seen] == ["x", "xxx", "xx"]


def test_empty_candidates():
    assert Reranker("stub", 1024, score_fn=by_length).rerank("q", []) == []


def test_wrong_score_count_is_an_error():
    bad = Reranker("stub", 1024, score_fn=lambda pairs: [1.0])
    with pytest.raises(ValueError):
        bad.rerank("q", CANDS)
