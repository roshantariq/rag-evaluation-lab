"""Tests for scripts/19_paraphrase_result.py, on synthetic runs.

The script was written before the paraphrased runs existed, so that its
analysis could not be shaped by their result. These tests are what it was
checked against instead: hand-built tables where the right verdict is
known by construction, one per outcome the protocol declares admissible,
plus the two gates (manipulation check, broken-question guard).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "19_paraphrase_result.py"
spec = importlib.util.spec_from_file_location("paraphrase_result", SCRIPT)
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

N_REWRITTEN = 20
UNTOUCHED = ["u001", "u002", "u003"]


def _ids():
    return [f"q{i:03d}" for i in range(N_REWRITTEN)]


def _write_run(tables: Path, tag: str, hits: dict[str, int], n_ev: dict[str, int]):
    """A retrieval_<tag>.csv carrying only the columns 19 reads.

    Hit cells are written as "0;1"-style index lists, and a single-index
    column becomes float under pandas - the dtype trap _count_spans guards.
    """
    rows = []
    for q, n in n_ev.items():
        cell = ";".join(str(i) for i in range(hits[q])) if hits[q] else ""
        rows.append({"id": q, "question_type": "factual", "n_evidence": n,
                     "scorable": True, "hit_spans@20": cell,
                     "hit_spans@B20000": cell})
    rows.append({"id": "x999", "question_type": "unanswerable", "n_evidence": 0,
                 "scorable": False, "hit_spans@20": "", "hit_spans@B20000": ""})
    pd.DataFrame(rows).to_csv(tables / f"retrieval_{tag}.csv", index=False)


def _setup(tmp_path: Path, orig_dense, orig_bm25, para_dense, para_bm25,
           drop=0.40, n_ev=2, drift=False):
    """Build a full synthetic input set; each arm is a per-question hit list."""
    tables, ev = tmp_path / "tables", tmp_path / "eval"
    tables.mkdir()
    ev.mkdir()
    ids = _ids()
    n = {q: n_ev for q in ids} | {u: 1 for u in UNTOUCHED}
    untouched_hits = {u: 1 for u in UNTOUCHED}
    for tag, arm in (("baseline", orig_dense), ("bm25", orig_bm25),
                     ("para_dense", para_dense), ("para_bm25", para_bm25)):
        hits = dict(zip(ids, arm)) | untouched_hits
        if drift and tag == "para_bm25":
            hits["u001"] = 0
        _write_run(tables, tag, hits, n)

    pd.DataFrame({
        "id": ids, "question_type": ["factual"] * N_REWRITTEN,
        "overlap_before": [0.8] * N_REWRITTEN,
        "overlap_after": [0.8 - drop] * N_REWRITTEN,
    }).to_csv(tables / "paraphrase_overlap.csv", index=False)

    with open(ev / "gold_questions.jsonl", "w", encoding="utf-8") as o, \
         open(ev / "gold_paraphrased.jsonl", "w", encoding="utf-8") as p:
        for q in ids + UNTOUCHED:
            rec = {"id": q, "question": f"original {q}", "evidence": []}
            o.write(json.dumps(rec) + "\n")
            if q in ids:
                rec = rec | {"question": f"rewritten {q}"}
            p.write(json.dumps(rec) + "\n")
    return tables, ev


def _run(tables: Path, ev: Path, monkeypatch) -> pd.DataFrame:
    monkeypatch.setattr(sys, "argv", ["19", "--tables-dir", str(tables),
                                      "--eval-dir", str(ev)])
    assert pr.main() == 0
    return pd.read_csv(tables / "paraphrase_result.csv").set_index("metric")


# --- the interval rule ------------------------------------------------------

@pytest.mark.parametrize("lo,hi,expected", [
    (0.01, 0.30, "supported"),
    (-0.30, -0.01, "refuted"),
    (-0.10, 0.20, "inconclusive"),
    (0.0, 0.20, "inconclusive"),     # touching zero is not excluding it
    (-0.20, 0.0, "inconclusive"),
])
def test_classify(lo, hi, expected):
    assert pr.classify(lo, hi) == expected


def test_constants_match_the_protocol():
    """If any of these move, the protocol was changed after the fact."""
    assert pr.MIN_OVERLAP_DROP == 0.25
    assert pr.GUARD_DENSE_DROP == 0.15
    assert pr.BOOT == 10_000
    assert pr.SEED == 20260827
    assert {c for _, c, _ in pr.METRICS} == {"hit_spans@B20000", "hit_spans@20"}


# --- one synthetic world per admissible outcome -----------------------------

def test_supported_when_advantage_vanishes_after_rewriting(tmp_path, monkeypatch):
    # Original: bm25 finds both spans, dense finds none -> delta +1.0.
    # Rewritten: both find one span each -> delta 0. Effect +1.0, no noise.
    t, e = _setup(tmp_path, [0] * 20, [2] * 20, [1] * 20, [1] * 20)
    res = _run(t, e, monkeypatch)
    for metric in ("budget 20,000", "k=20"):
        r = res.loc[metric]
        assert r["delta_original"] == pytest.approx(1.0)
        assert r["delta_paraphrase"] == pytest.approx(0.0)
        assert r["effect"] == pytest.approx(1.0)
        assert r["outcome"] == "supported"
        assert not r["guard_triggered"]        # dense rose, so no warning


def test_refuted_when_advantage_grows_after_rewriting(tmp_path, monkeypatch):
    t, e = _setup(tmp_path, [1] * 20, [1] * 20, [0] * 20, [2] * 20)
    res = _run(t, e, monkeypatch)
    assert set(res["outcome"]) == {"refuted"}
    assert res.loc["k=20", "effect"] == pytest.approx(-1.0)


def test_inconclusive_when_effect_is_noise(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    arms = [list(rng.integers(0, 3, 20)) for _ in range(4)]
    # Make the two deltas equal in expectation by swapping arms symmetrically.
    t, e = _setup(tmp_path, arms[0], arms[1], arms[1], arms[0])
    res = _run(t, e, monkeypatch)
    r = res.loc["budget 20,000"]
    assert r["effect_lo"] < 0 < r["effect_hi"]
    assert r["outcome"] == "inconclusive"


# --- the two gates ----------------------------------------------------------

def test_failed_manipulation_issues_no_verdict(tmp_path, monkeypatch):
    t, e = _setup(tmp_path, [0] * 20, [2] * 20, [1] * 20, [1] * 20, drop=0.20)
    res = _run(t, e, monkeypatch)
    assert set(res["outcome"]) == {"invalid"}
    assert not res["manipulation_passed"].any()
    assert res["overlap_drop"].iloc[0] == pytest.approx(0.20)


def test_drop_exactly_at_threshold_passes(tmp_path, monkeypatch):
    t, e = _setup(tmp_path, [0] * 20, [2] * 20, [1] * 20, [1] * 20, drop=0.25)
    assert _run(t, e, monkeypatch)["manipulation_passed"].all()


def test_guard_fires_when_dense_collapses(tmp_path, monkeypatch):
    # Dense falls from 1.0 to 0.0 on the rewritten questions.
    t, e = _setup(tmp_path, [2] * 20, [2] * 20, [0] * 20, [2] * 20)
    res = _run(t, e, monkeypatch)
    assert res["guard_triggered"].all()
    assert res.loc["k=20", "dense_change"] == pytest.approx(-1.0)


def test_guard_quiet_at_exactly_minus_015(tmp_path, monkeypatch):
    # 20 questions x 2 spans = 40 spans; 6 fewer dense hits is exactly -0.15.
    orig = [2] * 20
    para = [2] * 14 + [1] * 6
    t, e = _setup(tmp_path, orig, orig, para, orig)
    res = _run(t, e, monkeypatch)
    assert res.loc["k=20", "dense_change"] == pytest.approx(-0.15)
    assert not res["guard_triggered"].any()


# --- apparatus checks and reproducibility -----------------------------------

def test_drift_on_untouched_questions_is_reported(tmp_path, monkeypatch, capsys):
    t, e = _setup(tmp_path, [0] * 20, [2] * 20, [1] * 20, [1] * 20, drift=True)
    res = _run(t, e, monkeypatch)
    assert (res["apparatus_warnings"] > 0).all()
    assert "untouched question(s) scored differently" in capsys.readouterr().out


def test_check_gold_flags_extra_changes(tmp_path):
    _, e = _setup(tmp_path, [0] * 20, [2] * 20, [1] * 20, [1] * 20)
    assert pr.check_gold(e, set(_ids())) == []
    assert pr.check_gold(e, set(_ids()[:-1]))      # one rewrite undeclared


def test_same_inputs_same_numbers(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    arms = [list(rng.integers(0, 3, 20)) for _ in range(4)]
    t, e = _setup(tmp_path, *arms)
    first = _run(t, e, monkeypatch)
    second = _run(t, e, monkeypatch)
    pd.testing.assert_frame_equal(first, second)


def test_per_question_table_written(tmp_path, monkeypatch):
    t, e = _setup(tmp_path, [0] * 20, [2] * 20, [1] * 20, [1] * 20)
    _run(t, e, monkeypatch)
    q = pd.read_csv(t / "paraphrase_result_questions.csv")
    assert len(q) == N_REWRITTEN
    assert {"orig_bm25@k20", "para_dense@B20000", "overlap_after"} <= set(q.columns)
