"""Tests for scripts/25_replication_result.py, on synthetic tables.

Written before the replication runs exist: each declared outcome is built
by hand and checked, and the interaction arithmetic is checked on a case
worked out by hand.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "25_replication_result.py"
spec = importlib.util.spec_from_file_location("replication_result", SCRIPT)
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

LABEL = "budget 20,000"
N = 20


def _run_table(t: Path, tag: str, hits: list[int]):
    rows = [{"id": f"q{i:02d}", "scorable": True, "n_evidence": 2,
             "hit_spans@B20000": ";".join(str(j) for j in range(h))}
            for i, h in enumerate(hits)]
    rows.append({"id": "x99", "scorable": False, "n_evidence": 0, "hit_spans@B20000": ""})
    pd.DataFrame(rows).to_csv(t / f"retrieval_{tag}.csv", index=False)


def _setup(tmp_path, hits: dict, ci=(-0.05, 0.08), win=0.6, metric=LABEL) -> Path:
    t = tmp_path / "tables"
    t.mkdir()
    for tag in (rp.CONTROL, rp.ARM) + rp.DENSE + rp.EXPLORATORY:
        _run_table(t, tag, hits.get(tag, [1] * N))
    pd.DataFrame([{"a": rp.CONTROL, "b": rp.ARM, "metric": metric, "recall_a": 0.5,
                   "recall_b": 0.52, "delta": 0.02, "ci_lo": ci[0], "ci_hi": ci[1],
                   "p": 0.5, "p_holm": 0.5}]).to_csv(t / rp.PAIRED, index=False)
    pd.DataFrame([{"run": rp.ARM, "metric": metric, "win_rate": win},
                  {"run": rp.CONTROL, "metric": metric, "win_rate": 1 - win}]) \
        .to_csv(t / rp.STABILITY, index=False)
    return t


def _run(t, monkeypatch) -> pd.Series:
    monkeypatch.setattr(sys, "argv", ["25", "--tables-dir", str(t)])
    assert rp.main() == 0
    return pd.read_csv(t / "replication_result.csv").iloc[0]


def test_constants_match_the_protocol():
    assert (rp.CONTROL, rp.ARM) == ("bm25", "bm25_chunk256")
    assert rp.DENSE == ("baseline", "chunk256")
    assert rp.EXPLORATORY == ("rerank", "rerank_chunk256")
    assert rp.WIN_THRESHOLD == 0.90 and rp.BOOT == 10_000 and rp.SEED == 20260827
    assert rp.COL == "hit_spans@B20000"


def test_null_keeps_fixed512(tmp_path, monkeypatch):
    r = _run(_setup(tmp_path, {}), monkeypatch)
    assert r["outcome"] == "not_separable" and r["chosen"] == "bm25"


def test_replaces(tmp_path, monkeypatch):
    r = _run(_setup(tmp_path, {}, ci=(0.02, 0.1), win=0.95), monkeypatch)
    assert r["outcome"] == "replaces" and r["chosen"] == "bm25_chunk256"


def test_leaning_keeps_fixed512(tmp_path, monkeypatch):
    r = _run(_setup(tmp_path, {}, ci=(0.02, 0.1), win=0.7), monkeypatch)
    assert r["outcome"] == "leaning" and r["chosen"] == "bm25"


def test_interaction_arithmetic(tmp_path, monkeypatch):
    # 20 questions x 2 spans = 40 spans.
    # BM25: 256 gains 1 span on 8 questions -> +8/40 = +0.20
    # dense: 256 gains 1 span on 2 questions -> +2/40 = +0.05
    # interaction = 0.20 - 0.05 = +0.15
    hits = {"bm25": [1] * N, "bm25_chunk256": [2] * 8 + [1] * 12,
            "baseline": [1] * N, "chunk256": [2] * 2 + [1] * 18}
    r = _run(_setup(tmp_path, hits), monkeypatch)
    assert r["bm25_effect"] == pytest.approx(0.20)
    assert r["dense_effect"] == pytest.approx(0.05)
    assert r["interaction"] == pytest.approx(0.15)
    assert r["interaction_lo"] <= 0.15 <= r["interaction_hi"]


def test_reported_estimates_cannot_change_the_decision(tmp_path, monkeypatch):
    hits = {"bm25_chunk256": [2] * N, "rerank_chunk256": [2] * N}   # huge reported gains
    r = _run(_setup(tmp_path, hits, ci=(-0.01, 0.2), win=0.99), monkeypatch)
    assert r["chosen"] == "bm25"


def test_wrong_metric_is_refused(tmp_path, monkeypatch):
    t = _setup(tmp_path, {}, metric="k=10")
    monkeypatch.setattr(sys, "argv", ["25", "--tables-dir", str(t)])
    assert rp.main() == 1
    assert not (t / "replication_result.csv").exists()
