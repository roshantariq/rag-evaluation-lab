"""Tests for scripts/24_chunker_result.py, on synthetic tables.

Written before any chunker run exists: each outcome the protocol declares
is built by hand and checked.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "24_chunker_result.py"
spec = importlib.util.spec_from_file_location("chunker_result", SCRIPT)
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

LABEL = "budget 20,000"
NULL_CI = {a: (-0.04, 0.05) for a in cr.ARMS}
EVEN = {"bm25": 0.25, **{a: 0.25 for a in cr.ARMS}}


def _setup(tmp_path, cis=NULL_CI, wins=EVEN, metric=LABEL) -> Path:
    t = tmp_path / "tables"
    t.mkdir()
    pd.DataFrame([{"a": "bm25", "b": arm, "metric": metric, "recall_a": 0.584,
                   "recall_b": 0.584 + sum(cis[arm]) / 2, "delta": sum(cis[arm]) / 2,
                   "ci_lo": cis[arm][0], "ci_hi": cis[arm][1], "p": 0.5, "p_holm": 1.0}
                  for arm in cr.ARMS]).to_csv(t / cr.PAIRED, index=False)
    pd.DataFrame([{"run": r, "metric": metric, "win_rate": w}
                  for r, w in wins.items()]).to_csv(t / cr.STABILITY, index=False)
    for arm in cr.ARMS:
        pd.DataFrame({"chunk_id": ["x:0-100", "x:101-300"], "arxiv_id": ["x", "x"],
                      "char_start": [0, 101], "char_end": [100, 300]}) \
            .to_csv(t / cr.chunk_file(arm), index=False)
    return t


def _run(t, monkeypatch) -> pd.DataFrame:
    monkeypatch.setattr(sys, "argv", ["24", "--tables-dir", str(t)])
    assert cr.main() == 0
    return pd.read_csv(t / "chunker_result.csv").set_index("arm")


def test_constants_match_the_protocol():
    assert cr.ARMS == ("chunk_section", "chunk_recursive", "chunk_semantic")
    assert cr.TARGETS == {"chunk_section": 680, "chunk_recursive": 650, "chunk_semantic": 520}
    assert cr.WIN_THRESHOLD == 0.90 and cr.CONTROL == "bm25" and cr.LABEL == LABEL


def test_chunk_file_names():
    assert cr.chunk_file("chunk_semantic") == "chunks_semantic.csv"


def test_all_null_keeps_fixed512(tmp_path, monkeypatch):
    out = _run(_setup(tmp_path), monkeypatch)
    assert set(out["outcome"]) == {"not_separable"}
    assert out["chosen"].iloc[0] == "bm25"


def test_one_chunker_replaces(tmp_path, monkeypatch):
    cis = NULL_CI | {"chunk_section": (0.02, 0.12)}
    wins = {"bm25": 0.02, "chunk_section": 0.95, "chunk_recursive": 0.02,
            "chunk_semantic": 0.01}
    out = _run(_setup(tmp_path, cis, wins), monkeypatch)
    assert out.loc["chunk_section", "outcome"] == "replaces"
    assert out["chosen"].iloc[0] == "chunk_section"


def test_leaning_and_worse_keep_fixed512(tmp_path, monkeypatch):
    cis = NULL_CI | {"chunk_recursive": (0.01, 0.1), "chunk_semantic": (-0.1, -0.01)}
    wins = {"bm25": 0.3, "chunk_section": 0.1, "chunk_recursive": 0.6, "chunk_semantic": 0.0}
    out = _run(_setup(tmp_path, cis, wins), monkeypatch)
    assert out.loc["chunk_recursive", "outcome"] == "leaning"
    assert out.loc["chunk_semantic", "outcome"] == "worse"
    assert out["chosen"].iloc[0] == "bm25"


def test_chunk_set_size_is_reported(tmp_path, monkeypatch):
    out = _run(_setup(tmp_path), monkeypatch)
    assert out.loc["chunk_section", "n_chunks"] == 2
    assert out.loc["chunk_section", "median_chars"] == 149.5


def test_wrong_metric_is_refused(tmp_path, monkeypatch):
    t = _setup(tmp_path, metric="k=10")
    monkeypatch.setattr(sys, "argv", ["24", "--tables-dir", str(t)])
    assert cr.main() == 1
    assert not (t / "chunker_result.csv").exists()
