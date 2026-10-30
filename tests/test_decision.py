"""Tests for the shared switching-rule decision table."""

from __future__ import annotations

import pytest

from rageval.evaluation.decision import classify, decide


@pytest.mark.parametrize("lo,hi,win,expected", [
    (0.01, 0.2, 0.90, "replaces"),        # threshold inclusive
    (0.01, 0.2, 0.89, "leaning"),
    (0.0, 0.2, 0.99, "not_separable"),    # touching zero is not excluding it
    (-0.01, 0.2, 0.99, "not_separable"),
    (-0.2, -0.01, 0.0, "worse"),
    (-0.2, 0.0, 0.0, "not_separable"),    # touching zero from below
])
def test_classify(lo, hi, win, expected):
    assert classify(lo, hi, win, 0.90) == expected


def test_decide_returns_the_single_winner_or_the_control():
    assert decide({"a": "replaces", "b": "leaning"}, "ctl") == "a"
    assert decide({"a": "leaning", "b": "worse"}, "ctl") == "ctl"


def test_two_winners_is_an_error():
    with pytest.raises(ValueError):
        decide({"a": "replaces", "b": "replaces"}, "ctl")
