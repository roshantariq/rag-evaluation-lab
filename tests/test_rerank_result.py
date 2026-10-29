"""Tests for scripts/20_rerank_result.py, on synthetic tables.

Written before any reranking run, like test_paraphrase_result.py: each
outcome the protocol declares admissible is built by hand and checked.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "20_rerank_result.py"
spec = importlib.util.spec_from_file_location("rerank_result", SCRIPT)
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

REWRITTEN = [f"q{i:03d}" for i in range(20)]
UNTOUCHED = [f"u{i:03d}" for i in range(10)]


def _write_run(tables: Path, tag: str, hits: dict[str, int]):
    rows = [{"id": q, "question_type": "factual", "n_evidence": 2, "scorable": True,
             "hit_spans@B20000": ";".join(str(i) for i in range(h)) if h else ""}
            for q, h in hits.items()]
    rows.append({"id": "x999", "question_type": "unanswerable", "n_evidence": 0,
                 "scorable": False, "hit_spans@B20000": ""})
    pd.DataFrame(rows).to_csv(tables / f"retrieval_{tag}.csv", index=False)


def _setup(tmp_path, bm25, rerank, para_bm25, para_rerank, ci=(0.05, 0.20),
           win=0.95, metric="budget 20,000", drift=False):
    t = tmp_path / "tables"
    t.mkdir()
    untouched = {u: 1 for u in UNTOUCHED}
    for tag, arm in (("bm25", bm25), ("rerank", rerank),
                     ("para_bm25", para_bm25), ("para_rerank", para_rerank)):
        hits = dict(zip(REWRITTEN, arm)) | untouched
        if drift and tag == "para_rerank":
            hits["u000"] = 0
        _write_run(t, tag, hits)
    pd.DataFrame({"id": REWRITTEN}).to_csv(t / "paraphrase_overlap.csv", index=False)
    pd.DataFrame([{"a": "bm25", "b": "rerank", "metric": metric, "recall_a": 0.5,
                   "recall_b": 0.6, "delta": 0.1, "ci_lo": ci[0], "ci_hi": ci[1],
                   "p": 0.01, "p_holm": 0.02}]).to_csv(
        t / "paired_baseline_bm25_rerank_B20000.csv", index=False)
    pd.DataFrame([{"run": "rerank", "metric": metric, "win_rate": win},
                  {"run": "bm25", "metric": metric, "win_rate": 1 - win}]).to_csv(
        t / "stability_bm25_rerank_B20000.csv", index=False)
    return t


def _run(t, monkeypatch) -> pd.Series:
    monkeypatch.setattr(sys, "argv", ["20", "--tables-dir", str(t)])
    assert rr.main() == 0
    return pd.read_csv(t / "rerank_result.csv").iloc[0]


# --- the decision tables ----------------------------------------------------

@pytest.mark.parametrize("lo,hi,win,expected", [
    (0.01, 0.2, 0.95, "replaces"),
    (0.01, 0.2, 0.90, "replaces"),       # threshold is inclusive
    (0.01, 0.2, 0.89, "leaning"),
    (-0.01, 0.2, 0.99, "not_separable"),          # the CI decides first
    (0.0, 0.2, 0.99, "not_separable"),            # touching zero is not excluding it
    (-0.2, -0.01, 0.0, "worse"),
])
def test_classify_primary(lo, hi, win, expected):
    assert rr.classify_primary(lo, hi, win) == expected


@pytest.mark.parametrize("lo,hi,expected", [
    (0.01, 0.3, "survives"), (-0.1, 0.2, "inconclusive"), (-0.3, -0.01, "hurts"),
])
def test_classify_secondary(lo, hi, expected):
    assert rr.classify_secondary(lo, hi) == expected


def test_constants_match_the_protocol():
    assert rr.WIN_THRESHOLD == 0.90
    assert rr.BOOT == 10_000 and rr.PERMS == 10_000
    assert rr.SEED == 20260827
    assert rr.COL == "hit_spans@B20000"


# --- the randomization test --------------------------------------------------

def test_randomization_identical_arms_gives_p_one():
    h = np.array([1, 2, 0, 1])
    obs, p = rr.randomization_p(h, h, np.array([2] * 4), np.random.default_rng(0))
    assert obs == 0.0 and p == 1.0


def test_randomization_consistent_gain_is_significant():
    ha, hb = np.zeros(30, int), np.ones(30, int)
    obs, p = rr.randomization_p(ha, hb, np.full(30, 2), np.random.default_rng(0))
    assert obs == pytest.approx(0.5)
    assert p < 0.001


# --- end to end ---------------------------------------------------------------

def test_replaces_and_secondary_survives(tmp_path, monkeypatch):
    t = _setup(tmp_path, [0] * 20, [2] * 20, [0] * 20, [2] * 20)
    r = _run(t, monkeypatch)
    assert r["primary"] == "replaces"
    assert r["gain_paraphrase"] == pytest.approx(1.0)
    assert r["secondary"] == "survives"
    assert r["untouched_drift"] == 0


def test_secondary_cannot_change_primary(tmp_path, monkeypatch):
    # Primary tables say null; the rewritten questions show a huge gain.
    t = _setup(tmp_path, [0] * 20, [2] * 20, [0] * 20, [2] * 20, ci=(-0.02, 0.1))
    r = _run(t, monkeypatch)
    assert r["primary"] == "not_separable"
    assert r["secondary"] == "survives"


def test_secondary_hurts(tmp_path, monkeypatch):
    t = _setup(tmp_path, [1] * 20, [1] * 20, [2] * 20, [0] * 20)
    assert _run(t, monkeypatch)["secondary"] == "hurts"


def test_nondeterminism_is_reported(tmp_path, monkeypatch, capsys):
    t = _setup(tmp_path, [1] * 20, [1] * 20, [1] * 20, [1] * 20, drift=True)
    assert _run(t, monkeypatch)["untouched_drift"] == 1
    assert "DETERMINISM WARNING" in capsys.readouterr().out


def test_wrong_metric_is_refused(tmp_path, monkeypatch):
    t = _setup(tmp_path, [1] * 20, [1] * 20, [1] * 20, [1] * 20, metric="k=10")
    monkeypatch.setattr(sys, "argv", ["20", "--tables-dir", str(t)])
    assert rr.main() == 1
    assert not (t / "rerank_result.csv").exists()


def test_same_inputs_same_numbers(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    arms = [list(rng.integers(0, 3, 20)) for _ in range(4)]
    t = _setup(tmp_path, *arms)
    first, second = _run(t, monkeypatch), _run(t, monkeypatch)
    pd.testing.assert_series_equal(first, second)
