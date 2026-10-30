"""Tests for rageval.provenance: the ledger row, and the dirty flag's edges."""

from __future__ import annotations

import csv
import hashlib

import pytest

from rageval import provenance as pv
from rageval.config import PROJECT_ROOT


def fake_git(sha: str | None, status: str | None):
    def run(*args):
        return sha if args[0] == "rev-parse" else status
    return run


def read_ledger(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_clean_tree_is_not_dirty(monkeypatch):
    monkeypatch.setattr(pv, "_git", fake_git("abc123\n", ""))
    assert pv.tree_state() == ("abc123", "False", "")


def test_own_outputs_do_not_make_the_tree_dirty(monkeypatch):
    status = " M results/tables/retrieval_bm25.csv\n?? results/tables/provenance.csv\n"
    monkeypatch.setattr(pv, "_git", fake_git("abc123\n", status))
    own = [PROJECT_ROOT / "results/tables/retrieval_bm25.csv",
           PROJECT_ROOT / "results/tables/provenance.csv"]
    assert pv.tree_state(exclude=own) == ("abc123", "False", "")


def test_another_table_is_an_input_and_counts(monkeypatch):
    # scripts 13-28 read other tables, so a modified one is a dirty input
    status = " M results/tables/retrieval_bm25.csv\n M results/tables/retrieval_baseline.csv\n"
    monkeypatch.setattr(pv, "_git", fake_git("abc123\n", status))
    own = [PROJECT_ROOT / "results/tables/retrieval_bm25.csv"]
    assert pv.tree_state(exclude=own) == ("abc123", "True", "results/tables/retrieval_baseline.csv")


def test_untracked_and_renamed_paths_count(monkeypatch):
    status = "?? scripts/new_helper.py\nR  src/a.py -> src/b.py\n"
    monkeypatch.setattr(pv, "_git", fake_git("abc123\n", status))
    assert pv.tree_state() == ("abc123", "True", "scripts/new_helper.py;src/b.py")


@pytest.mark.parametrize("sha, status", [(None, ""), ("abc\n", None), (None, None)])
def test_no_git_answer_reads_unknown_not_clean(monkeypatch, sha, status):
    monkeypatch.setattr(pv, "_git", fake_git(sha, status))
    assert pv.tree_state() == ("unknown", "unknown", "")


def test_long_dirty_list_is_capped_and_says_so(monkeypatch):
    status = "".join(f" M src/f{i}.py\n" for i in range(pv.MAX_DIRTY_PATHS + 5))
    monkeypatch.setattr(pv, "_git", fake_git("abc\n", status))
    _, dirty, paths = pv.tree_state()
    assert dirty == "True" and paths.endswith(";+5 more")


def test_record_appends_beside_the_table(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "_git", fake_git("abc123\n", " M config/baseline.yaml\n"))
    gold = tmp_path / "gold.jsonl"
    gold.write_text('{"id": "f001"}\n', encoding="utf-8")
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pv.record([a, b], gold=[gold], argv=["scripts/x.py", "--tag", "two words"])
    pv.record([a], argv=["scripts/x.py"])
    rows = read_ledger(tmp_path / "provenance.csv")
    assert [r["table"] for r in rows] == ["a.csv", "b.csv", "a.csv"]
    first = rows[0]
    assert tuple(first) == pv.COLUMNS
    assert first["git_sha"] == "abc123" and first["dirty"] == "True"
    assert first["dirty_paths"] == "config/baseline.yaml"
    assert first["gold_sha256"] == hashlib.sha256(gold.read_bytes()).hexdigest()
    assert first["command"].endswith('--tag "two words"')
    assert first["source"] == "recorded"
    assert rows[2]["gold_files"] == "n/a" and rows[2]["gold_sha256"] == "n/a"
