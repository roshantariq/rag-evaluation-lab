"""Tests for span-overlap retrieval metrics.

Every expected value here is computed by hand in the comment beside it. A
silently wrong nDCG would poison every comparison in the ablation phase,
and a metric bug is invisible in aggregate output - the numbers still look
like plausible numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from rageval.evaluation.retrieval_metrics import (
    Retrieved,
    Span,
    aggregate,
    aggregate_by_type,
    coverage_at_k,
    count_relevant_chunks,
    evaluate_question,
    is_scorable,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_flags,
    spans_hit,
)


# --------------------------------------------------------------------------
# Minimal stand-ins, so these tests do not depend on the gold set or index.
# --------------------------------------------------------------------------


@dataclass
class FakeEvidence:
    arxiv_id: str
    char_start: int
    char_end: int


@dataclass
class FakeQuestion:
    id: str
    question_type: str
    evidence: list
    difficulty: str = "medium"


def r(paper: str, start: int, end: int, score: float = 0.0) -> Retrieved:
    return Retrieved(paper, start, end, score)


def s(paper: str, start: int, end: int) -> Span:
    return Span(paper, start, end)


# --------------------------------------------------------------------------
# Overlap semantics
# --------------------------------------------------------------------------


def test_overlap_is_half_open_at_the_end():
    # Chunk [0,100) ends exactly where span [100,200) begins: no shared char.
    assert not r("A", 0, 100).overlaps("A", 100, 200)


def test_overlap_is_half_open_at_the_start():
    # Chunk [200,300) begins exactly where span [100,200) ends.
    assert not r("A", 200, 300).overlaps("A", 100, 200)


def test_single_shared_character_is_an_overlap():
    assert r("A", 0, 101).overlaps("A", 100, 200)


def test_same_offsets_different_paper_is_not_a_hit():
    # The failure that produced c006-c008: internally consistent evidence
    # attached to the wrong paper. Offsets alone must never match.
    assert not r("B", 100, 200).overlaps("A", 100, 200)


def test_chunk_wholly_inside_span_counts():
    assert r("A", 120, 130).overlaps("A", 100, 200)


def test_span_wholly_inside_chunk_counts():
    assert r("A", 0, 1000).overlaps("A", 100, 200)


# --------------------------------------------------------------------------
# Recall and coverage
# --------------------------------------------------------------------------


def test_recall_and_coverage_agree_on_single_evidence():
    spans = [s("A", 100, 200)]
    got = [r("A", 150, 250)]
    assert recall_at_k(got, spans, 5) == 1.0
    assert coverage_at_k(got, spans, 5) == 1.0


def test_recall_and_coverage_diverge_on_two_spans():
    # This divergence is the reason both are reported.
    spans = [s("A", 100, 200), s("B", 300, 400)]
    got = [r("A", 150, 250)]  # only the first span is touched
    assert recall_at_k(got, spans, 5) == 1.0
    assert coverage_at_k(got, spans, 5) == 0.5


def test_coverage_reaches_one_when_both_spans_hit():
    spans = [s("A", 100, 200), s("B", 300, 400)]
    got = [r("A", 150, 250), r("B", 350, 450)]
    assert coverage_at_k(got, spans, 5) == 1.0


def test_k_truncates_before_scoring():
    spans = [s("A", 100, 200)]
    got = [r("A", 0, 10), r("A", 10, 20), r("A", 150, 250)]  # hit at rank 3
    assert recall_at_k(got, spans, 2) == 0.0
    assert recall_at_k(got, spans, 3) == 1.0


def test_duplicate_chunks_do_not_inflate_coverage():
    spans = [s("A", 100, 200), s("B", 300, 400)]
    got = [r("A", 150, 250), r("A", 160, 260), r("A", 170, 270)]
    assert coverage_at_k(got, spans, 5) == 0.5


# --------------------------------------------------------------------------
# MRR
# --------------------------------------------------------------------------


def test_reciprocal_rank_uses_the_first_hit():
    spans = [s("A", 100, 200)]
    got = [r("A", 0, 10), r("A", 150, 250), r("A", 160, 260)]
    assert reciprocal_rank(got, spans) == pytest.approx(1 / 2)


def test_reciprocal_rank_is_zero_with_no_hits():
    assert reciprocal_rank([r("A", 0, 10)], [s("A", 100, 200)]) == 0.0


# --------------------------------------------------------------------------
# nDCG
# --------------------------------------------------------------------------


def test_ndcg_is_one_for_a_perfect_ranking():
    spans = [s("A", 100, 200)]
    got = [r("A", 150, 250), r("A", 0, 10)]
    assert ndcg_at_k(got, spans, 5, n_relevant_total=1) == pytest.approx(1.0)


def test_ndcg_for_a_single_hit_at_rank_three():
    # DCG  = 1/log2(4) = 0.5
    # IDCG = 1/log2(2) = 1.0
    # nDCG = 0.5
    spans = [s("A", 100, 200)]
    got = [r("A", 0, 10), r("A", 10, 20), r("A", 150, 250)]
    assert ndcg_at_k(got, spans, 5, n_relevant_total=1) == pytest.approx(0.5)


def test_ndcg_with_two_relevant_chunks_one_found_late():
    # Two relevant chunks exist; one retrieved at rank 2.
    # DCG  = 1/log2(3)          = 0.63093
    # IDCG = 1/log2(2)+1/log2(3) = 1.63093
    spans = [s("A", 100, 200)]
    got = [r("A", 0, 10), r("A", 150, 250)]
    expected = (1 / math.log2(3)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(got, spans, 5, n_relevant_total=2) == pytest.approx(expected)


def test_ndcg_ideal_is_capped_at_k():
    # 10 relevant chunks exist but k=2, so the ideal ranking has 2, not 10.
    spans = [s("A", 100, 200)]
    got = [r("A", 150, 250), r("A", 160, 260)]
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(got, spans, 2, n_relevant_total=10) == pytest.approx(
        (1 / math.log2(2) + 1 / math.log2(3)) / idcg
    )


def test_ndcg_without_total_does_not_flatter_an_empty_run():
    # No hits and no known total: must be 0.0, not 0/0 or 1.0.
    spans = [s("A", 100, 200)]
    got = [r("A", 0, 10)]
    assert ndcg_at_k(got, spans, 5) == 0.0


# --------------------------------------------------------------------------
# Relevant-chunk counting
# --------------------------------------------------------------------------


def test_count_relevant_chunks_scans_the_whole_index():
    spans = [s("A", 100, 200)]
    chunks = [r("A", 0, 50), r("A", 90, 150), r("A", 150, 260), r("B", 100, 200)]
    assert count_relevant_chunks(chunks, spans) == 2


# --------------------------------------------------------------------------
# Question-level behaviour
# --------------------------------------------------------------------------


def test_unanswerable_questions_are_not_scorable():
    q = FakeQuestion(id="u001", question_type="unanswerable", evidence=[])
    assert not is_scorable(q)
    row = evaluate_question(q, [r("A", 0, 100)])
    assert row["scorable"] is False
    # No zeroed metrics: a 0.0 here would read as a retrieval failure.
    assert "recall@5" not in row


def test_answerable_question_row_carries_shape_metadata():
    q = FakeQuestion(
        id="m001",
        question_type="multi_hop",
        evidence=[FakeEvidence("A", 100, 200), FakeEvidence("B", 300, 400)],
        difficulty="hard",
    )
    row = evaluate_question(q, [r("A", 150, 250)], ks=(5,), n_relevant_total=3)
    assert row["n_evidence"] == 2
    assert row["n_papers"] == 2
    assert row["recall@5"] == 1.0
    assert row["coverage@5"] == 0.5
    assert row["first_hit_rank"] == 1


def test_first_hit_rank_is_none_when_nothing_relevant():
    q = FakeQuestion(
        id="f001", question_type="factual", evidence=[FakeEvidence("A", 100, 200)]
    )
    row = evaluate_question(q, [r("A", 0, 10)], ks=(5,))
    assert row["first_hit_rank"] is None
    assert row["recall@5"] == 0.0


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_aggregate_ignores_unscorable_rows():
    rows = [
        {"question_type": "factual", "scorable": True, "mrr": 1.0, "recall@5": 1.0,
         "coverage@5": 1.0, "ndcg@5": 1.0},
        {"question_type": "unanswerable", "scorable": False},
    ]
    agg = aggregate(rows, ks=(5,))
    assert agg["n_questions"] == 2
    assert agg["n_scored"] == 1
    assert agg["recall@5"] == 1.0


def test_aggregate_by_type_keeps_a_stable_order():
    rows = [
        {"question_type": "multi_hop", "scorable": True, "mrr": 0.0, "recall@5": 0.0,
         "coverage@5": 0.0, "ndcg@5": 0.0},
        {"question_type": "factual", "scorable": True, "mrr": 1.0, "recall@5": 1.0,
         "coverage@5": 1.0, "ndcg@5": 1.0},
    ]
    out = aggregate_by_type(rows, ks=(5,))
    assert list(out) == ["factual", "multi_hop"]
    assert out["factual"]["recall@5"] == 1.0
    assert out["multi_hop"]["recall@5"] == 0.0


def test_hit_spans_records_which_span_was_found():
    # Two spans, only the second retrieved: the row must say "1", not just
    # coverage 0.5, or "found the answer but never the referring sentence"
    # is indistinguishable from "found the referring sentence only".
    q = FakeQuestion(
        id="m001",
        question_type="multi_hop",
        evidence=[FakeEvidence("A", 100, 200), FakeEvidence("B", 300, 400)],
    )
    row = evaluate_question(q, [r("B", 350, 450)], ks=(5,))
    assert row["hit_spans@5"] == "1"
    assert row["coverage@5"] == 0.5


def test_hit_spans_is_sorted_and_joined():
    q = FakeQuestion(
        id="m002",
        question_type="multi_hop",
        evidence=[FakeEvidence("A", 100, 200), FakeEvidence("B", 300, 400)],
    )
    row = evaluate_question(q, [r("B", 350, 450), r("A", 150, 250)], ks=(5,))
    assert row["hit_spans@5"] == "0;1"


def test_hit_spans_is_empty_string_when_nothing_found():
    q = FakeQuestion(
        id="m003",
        question_type="multi_hop",
        evidence=[FakeEvidence("A", 100, 200), FakeEvidence("B", 300, 400)],
    )
    row = evaluate_question(q, [r("C", 0, 10)], ks=(5,))
    assert row["hit_spans@5"] == ""


def test_hit_spans_respects_k():
    q = FakeQuestion(
        id="m004",
        question_type="multi_hop",
        evidence=[FakeEvidence("A", 100, 200), FakeEvidence("B", 300, 400)],
    )
    got = [r("A", 150, 250), r("C", 0, 10), r("B", 350, 450)]
    row = evaluate_question(q, got, ks=(2, 5))
    assert row["hit_spans@2"] == "0"
    assert row["hit_spans@5"] == "0;1"