"""Tests for rageval.evaluation.spans, including the pandas round trip."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from rageval.evaluation.spans import count_spans, span_indices


@pytest.mark.parametrize("cell,expected", [
    ("0;1", {0, 1}),
    ("2", {2}),
    (" 0 ; 3 ", {0, 3}),
    ("0.0", {0}),            # float64 inference turns "0" into 0.0
    (0.0, {0}),
    (1, {1}),
    ("", set()),
    ("nan", set()),
    (None, set()),
    (np.nan, set()),
    ("0;0", {0}),
])
def test_span_indices(cell, expected):
    assert span_indices(cell) == expected


def test_count_spans_matches_indices():
    assert count_spans("0;1;2") == 3
    assert count_spans(np.nan) == 0
    assert count_spans(0.0) == 1


def test_survives_float_inference_in_a_real_csv():
    # A column of only "0" and blanks is read back as float64.
    csv = "id,hit_spans@10\na,0\nb,\nc,0\n"
    df = pd.read_csv(io.StringIO(csv))
    assert df["hit_spans@10"].dtype == np.float64
    assert [count_spans(c) for c in df["hit_spans@10"]] == [1, 0, 1]
