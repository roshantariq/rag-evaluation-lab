"""Tests for scripts/22_filter_result.py, on synthetic tables.

Written before any filtered run exists: each outcome the protocol declares
is built by hand and checked.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "22_filter_result.py"
spec = importlib.util.spec_from_file_location("filter_result", SCRIPT)
fr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fr)

LABEL = "budget 20,000"


def _setup(tmp_path, cis: dict, wins: dict, metric=LABEL, slots=(3, 0, 5)) -> Path:
    t = tmp_path / "tables"
    t.mkdir()
    pd.DataFrame([{"a": "bm25", "b": arm, "metric": metric, "recall_a": 0.584,
                   "recall_b": 0.584 + sum(cis[arm]) / 2, "delta": sum(cis[arm]) / 2,
                   "ci_lo": cis[arm][0], "ci_hi": cis[arm][1], "p": 0.5, "p_holm": 1.0}
                  for arm in fr.ARMS]).to_csv(t / fr.PAIRED, index=False)
    pd.DataFrame([{"run": r, "metric": metric, "win_rate": w}
                  for r, w in wins.items()]).to_csv(t / fr.STABILITY, index=False)
    for arm, s in zip(fr.ARMS, slots):
        pd.DataFrame([{"id": "q1", "scorable": True, "low_prose_top10": s},
                      {"id": "q2", "scorable": True, "low_prose_top10": 0},
                      {"id": "x9", "scorable": False, "low_prose_top10": None}]) \
            .to_csv(t / f"retrieval_{arm}.csv", index=False)
    return t


def _run(t, monkeypatch) -> pd.DataFrame:
    monkeypatch.setattr(sys, "argv", ["22", "--tables-dir", str(t)])
    assert fr.main() == 0
    return pd.read_csv(t / "filter_result.csv").set_index("arm")


NULL_CI = {a: (-0.03, 0.04) for a in fr.ARMS}
EVEN = {"bm25": 0.25, "filter030": 0.25, "filter040": 0.25, "filter050": 0.25}


@pytest.mark.parametrize("lo,hi,win,expected", [
    (0.01, 0.1, 0.90, "replaces"),
    (0.01, 0.1, 0.89, "leaning"),
    (0.0, 0.1, 0.99, "not_separable"),     # touching zero is not excluding it
    (-0.01, 0.1, 0.99, "not_separable"),
    (-0.1, -0.01, 0.0, "worse"),
])
def test_classify(lo, hi, win, expected):
    assert fr.classify(lo, hi, win) == expected


def test_constants_match_the_protocol():
    assert fr.ARMS == ("filter030", "filter040", "filter050")
    assert fr.THRESHOLDS == {"filter030": 0.3, "filter040": 0.4, "filter050": 0.5}
    assert fr.WIN_THRESHOLD == 0.90 and fr.CONTROL == "bm25"
    assert fr.LABEL == LABEL


def test_two_winners_is_an_error():
    with pytest.raises(ValueError):
        fr.decide({"filter030": "replaces", "filter040": "replaces"})


def test_all_null_keeps_bm25(tmp_path, monkeypatch):
    out = _run(_setup(tmp_path, NULL_CI, EVEN), monkeypatch)
    assert set(out["outcome"]) == {"not_separable"}
    assert out["chosen"].iloc[0] == "bm25"


def test_one_arm_replaces(tmp_path, monkeypatch):
    cis = NULL_CI | {"filter040": (0.01, 0.08)}
    wins = {"bm25": 0.02, "filter030": 0.03, "filter040": 0.93, "filter050": 0.02}
    out = _run(_setup(tmp_path, cis, wins), monkeypatch)
    assert out.loc["filter040", "outcome"] == "replaces"
    assert out["chosen"].iloc[0] == "filter040"


def test_leaning_does_not_replace(tmp_path, monkeypatch):
    cis = NULL_CI | {"filter030": (0.01, 0.08)}
    wins = {"bm25": 0.2, "filter030": 0.6, "filter040": 0.1, "filter050": 0.1}
    out = _run(_setup(tmp_path, cis, wins), monkeypatch)
    assert out.loc["filter030", "outcome"] == "leaning"
    assert out["chosen"].iloc[0] == "bm25"


def test_worse_is_reported(tmp_path, monkeypatch):
    cis = NULL_CI | {"filter050": (-0.06, -0.01)}
    out = _run(_setup(tmp_path, cis, EVEN), monkeypatch)
    assert out.loc["filter050", "outcome"] == "worse"
    assert out["chosen"].iloc[0] == "bm25"


def test_freed_slots_are_summed_over_scorable_questions(tmp_path, monkeypatch):
    out = _run(_setup(tmp_path, NULL_CI, EVEN, slots=(3, 0, 5)), monkeypatch)
    assert out.loc["filter030", "freed_top10_slots"] == 3
    assert out.loc["filter040", "questions_with_freed_slots"] == 0
    assert out.loc["filter050", "freed_top10_slots"] == 5


def test_wrong_metric_is_refused(tmp_path, monkeypatch):
    t = _setup(tmp_path, NULL_CI, EVEN, metric="k=10")
    monkeypatch.setattr(sys, "argv", ["22", "--tables-dir", str(t)])
    assert fr.main() == 1
    assert not (t / "filter_result.csv").exists()
