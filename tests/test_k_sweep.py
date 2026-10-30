"""Tests for scripts/21_k_sweep.py, on synthetic tables.

Written before the script was run on the real retrieval tables, like
test_rerank_result.py: each outcome the protocol declares is built by hand
and checked.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "21_k_sweep.py"
spec = importlib.util.spec_from_file_location("k_sweep", SCRIPT)
ks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ks)

KS = (1, 3, 5, 10, 20)


def _cell(n: int) -> str:
    return ";".join(str(i) for i in range(n))


def _write(tables: Path, tag: str, found: list[dict[int, int]], n_ev=2,
           types=None) -> None:
    """found[i][k] = how many spans question i has found at k (nested by construction)."""
    rows = []
    for i, f in enumerate(found):
        row = {"id": f"q{i:03d}", "question_type": (types or ["multi_hop"] * len(found))[i],
               "n_evidence": n_ev, "scorable": True}
        row |= {f"hit_spans@{k}": _cell(f[k]) for k in KS}
        rows.append(row)
    rows.append({"id": "x999", "question_type": "unanswerable", "n_evidence": 0,
                 "scorable": False, **{f"hit_spans@{k}": "" for k in KS}})
    pd.DataFrame(rows).to_csv(tables / f"retrieval_{tag}.csv", index=False)


def _profile(first_k: int, n: int) -> dict[int, int]:
    """n spans, all first found at first_k."""
    return {k: (n if k >= first_k else 0) for k in KS}


def _run(t: Path, monkeypatch) -> pd.DataFrame:
    monkeypatch.setattr(sys, "argv", ["21", "--tables-dir", str(t)])
    assert ks.main() == 0
    return pd.read_csv(t / "k_sweep.csv")


def _tables(tmp_path: Path) -> Path:
    t = tmp_path / "tables"
    t.mkdir()
    return t


# --- the decision rule -------------------------------------------------------

def _steps(*separable: bool) -> pd.DataFrame:
    return pd.DataFrame({"k_to": KS[1:], "separable": list(separable)})


@pytest.mark.parametrize("flags,expected", [
    ((True, True, True, True), 20),
    ((False, True, True, True), 1),
    ((True, False, True, True), 3),     # stops at the first failure
    ((True, True, False, False), 5),
    ((True, True, True, False), 10),
])
def test_choose_k(flags, expected):
    assert ks.choose_k(_steps(*flags)) == expected


def test_constants_match_the_protocol():
    assert ks.KS == (1, 3, 5, 10, 20)
    assert ks.BOOT == 10_000 and ks.SEED == 20260827
    assert ks.DECIDING_ARM == "bm25"


# --- end to end -------------------------------------------------------------

def test_every_step_separable_returns_the_ceiling(tmp_path, monkeypatch):
    # 8 questions gain at each step: far above the ~4-question bar.
    found = []
    for first in KS:
        found += [_profile(first, 1) for _ in range(8)]
    t = _tables(tmp_path)
    _write(t, "bm25", found)
    out = _run(t, monkeypatch)
    assert out["separable"].all()
    assert out["chosen_k"].iloc[0] == 20
    assert bool(out["ladder_ceiling"].iloc[0])


def test_a_step_where_nothing_is_gained_has_lower_bound_exactly_zero(tmp_path, monkeypatch):
    # Gains at 1->3 and 3->5 only; nothing arrives after k=5.
    found = ([_profile(1, 1) for _ in range(10)] + [_profile(3, 1) for _ in range(10)]
             + [_profile(5, 1) for _ in range(10)] + [_profile(50, 0) for _ in range(10)])
    t = _tables(tmp_path)
    _write(t, "bm25", found)
    out = _run(t, monkeypatch).set_index("step")
    assert out.loc["5->10", "ci_lo"] == 0.0 and out.loc["5->10", "gain"] == 0.0
    assert out["chosen_k"].iloc[0] == 5
    assert not bool(out.loc["10->20", "tested"])     # reported, not tested


def test_a_single_question_gaining_is_not_separable(tmp_path, monkeypatch):
    found = [_profile(1, 2) for _ in range(30)]
    found[0] = {1: 0, 3: 2, 5: 2, 10: 2, 20: 2}
    t = _tables(tmp_path)
    _write(t, "bm25", found)
    out = _run(t, monkeypatch)
    assert out["chosen_k"].iloc[0] == 1


def test_non_nested_table_is_refused(tmp_path, monkeypatch):
    found = [_profile(1, 1) for _ in range(10)]
    found[3] = {1: 1, 3: 0, 5: 1, 10: 1, 20: 1}     # found at k=1, lost at k=3
    t = _tables(tmp_path)
    _write(t, "bm25", found)
    monkeypatch.setattr(sys, "argv", ["21", "--tables-dir", str(t)])
    assert ks.main() == 1
    assert not (t / "k_sweep.csv").exists()


def test_curve_counts_two_span_completeness(tmp_path, monkeypatch):
    # 4 two-span questions: both spans by k=5, k=10, one span only, none.
    found = [{1: 0, 3: 1, 5: 2, 10: 2, 20: 2}, {1: 0, 3: 0, 5: 1, 10: 2, 20: 2},
             {1: 1, 3: 1, 5: 1, 10: 1, 20: 1}, _profile(50, 0)]
    t = _tables(tmp_path)
    _write(t, "bm25", found, types=["comparative", "multi_hop", "multi_hop", "factual"])
    _run(t, monkeypatch)
    c = pd.read_csv(t / "k_sweep_curve.csv").set_index("k")
    assert c.loc[5, "two_span_both"] == 1 and c.loc[10, "two_span_both"] == 2
    assert c.loc[20, "two_span_n"] == 4
    assert c.loc[10, "span_recall"] == pytest.approx(5 / 8)
    assert c.loc[10, "complete_share"] == pytest.approx(0.5)
    assert c.loc[10, "recall_multi_hop"] == pytest.approx(3 / 4)
    assert np.isnan(c.loc[10, "recall_ambiguous"])


def test_reported_arms_never_change_the_decision(tmp_path, monkeypatch):
    base = [_profile(1, 1) for _ in range(20)]       # nothing gained after k=1
    better = [_profile(1, 1) for _ in range(10)] + [_profile(10, 1) for _ in range(10)]
    t = _tables(tmp_path)
    _write(t, "bm25", base)
    _write(t, "rerank", better)
    _write(t, "para_bm25", better)
    pd.DataFrame({"id": [f"q{i:03d}" for i in range(5)]}).to_csv(
        t / "paraphrase_overlap.csv", index=False)
    out = _run(t, monkeypatch)
    assert out["chosen_k"].iloc[0] == 1
    assert out["rerank_reaches_at_k"].iloc[0] == 10
    curves = pd.read_csv(t / "k_sweep_curve.csv")
    assert set(curves["subset"]) == {"all", "rewritten_20_original", "rewritten_20_reworded"}


def test_same_inputs_same_numbers(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    found = []
    for _ in range(40):
        first = int(rng.choice(KS + (50,)))
        found.append(_profile(first, int(rng.integers(1, 3))))
    t = _tables(tmp_path)
    _write(t, "bm25", found)
    pd.testing.assert_frame_equal(_run(t, monkeypatch), _run(t, monkeypatch))
