"""Tests for the chunk prose ratio and the quality filter."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rageval.chunking.quality import filter_records, prose_ratio


@dataclass
class Rec:
    chunk_id: str
    text: str


@pytest.mark.parametrize("text,expected", [
    ("the model predicts rainfall", 1.0),
    ("8 10 0.4 2 4 6", 0.0),                      # axis labels
    (". . . . 18 1.2", 0.0),                      # table-of-contents leader
    ("skill score (ACC): 0.4 0.25", 0.6),         # punctuation stripped
    ("", 0.0),
    ("an of is", 0.0),                            # words under three letters
])
def test_prose_ratio(text, expected):
    assert prose_ratio(text) == pytest.approx(expected)


def test_filter_splits_strictly_below_threshold():
    recs = [Rec("a", "words words words words"),   # 1.0
            Rec("b", "words words 1 2"),           # 0.5
            Rec("c", "1 2 3 words")]               # 0.25
    kept, removed = filter_records(recs, 0.5)
    assert [r.chunk_id for r in kept] == ["a", "b"]     # 0.5 is not below 0.5
    assert [r.chunk_id for r in removed] == ["c"]


def test_filter_at_zero_keeps_everything():
    recs = [Rec("a", "1 2 3"), Rec("b", "")]
    assert filter_records(recs, 0.0) == (recs, [])
