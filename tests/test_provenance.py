"""Tests for rageval.provenance: the ledger row, and the dirty flag's edges."""

from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess

import pytest

from rageval import provenance as pv
from rageval.config import PROJECT_ROOT


def fake_git(sha: str | None, changed: str | None, untracked: str | None = ""):
    def run(*args):
        return {"rev-parse": sha, "diff": changed, "ls-files": untracked}[args[0]]
    return run


def read_ledger(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_clean_tree_is_not_dirty(monkeypatch):
    monkeypatch.setattr(pv, "_git", fake_git("abc123\n", ""))
    assert pv.tree_state() == ("abc123", "False", "")


def test_own_outputs_do_not_make_the_tree_dirty(monkeypatch):
    monkeypatch.setattr(pv, "_git", fake_git("abc123\n", "results/tables/retrieval_bm25.csv\n",
                                             "results/tables/provenance.csv\n"))
    own = [PROJECT_ROOT / "results/tables/retrieval_bm25.csv",
           PROJECT_ROOT / "results/tables/provenance.csv"]
    assert pv.tree_state(exclude=own) == ("abc123", "False", "")


def test_another_table_is_an_input_and_counts(monkeypatch):
    # scripts 13-28 read other tables, so a modified one is a dirty input
    changed = "results/tables/retrieval_bm25.csv\nresults/tables/retrieval_baseline.csv\n"
    monkeypatch.setattr(pv, "_git", fake_git("abc123\n", changed))
    own = [PROJECT_ROOT / "results/tables/retrieval_bm25.csv"]
    assert pv.tree_state(exclude=own) == ("abc123", "True", "results/tables/retrieval_baseline.csv")


def test_untracked_files_count(monkeypatch):
    monkeypatch.setattr(pv, "_git", fake_git("abc123\n", "src/a.py\n", "scripts/new_helper.py\n"))
    assert pv.tree_state() == ("abc123", "True", "src/a.py;scripts/new_helper.py")


@pytest.mark.parametrize("sha, changed, untracked", [
    (None, "", ""), ("abc\n", None, ""), ("abc\n", "", None), (None, None, None)])
def test_no_git_answer_reads_unknown_not_clean(monkeypatch, sha, changed, untracked):
    monkeypatch.setattr(pv, "_git", fake_git(sha, changed, untracked))
    assert pv.tree_state() == ("unknown", "unknown", "")


def test_long_dirty_list_is_capped_and_says_so(monkeypatch):
    changed = "".join(f"src/f{i}.py\n" for i in range(pv.MAX_DIRTY_PATHS + 5))
    monkeypatch.setattr(pv, "_git", fake_git("abc\n", changed))
    _, dirty, paths = pv.tree_state()
    assert dirty == "True" and paths.endswith(";+5 more")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_crlf_rewrite_is_clean_but_a_content_change_is_dirty(tmp_path, monkeypatch):
    """Real git: pandas on Windows rewrites tables with CRLF and identical content."""
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / ".gitattributes").write_text("* text=auto eol=lf\n")
    table = tmp_path / "t.csv"
    table.write_bytes(b"a,b\n1,2\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    monkeypatch.setattr(pv, "PROJECT_ROOT", tmp_path)

    table.write_bytes(b"a,b\r\n1,2\r\n")
    assert pv.tree_state()[1:] == ("False", "")
    table.write_bytes(b"a,b\r\n1,3\r\n")
    assert pv.tree_state()[1:] == ("True", "t.csv")


def test_record_appends_beside_the_table(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "_git", fake_git("abc123\n", "config/baseline.yaml\n"))
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
    assert first["source"] == "recorded" and first["note"] == ""
    assert rows[2]["gold_files"] == "n/a" and rows[2]["gold_sha256"] == "n/a"


def test_ef_search_is_recorded_and_defaults_to_na(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "_git", fake_git("abc123\n", ""))
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pv.record([a], argv=["scripts/x.py"], ef_search=500)
    pv.record([b], argv=["scripts/x.py"])
    rows = read_ledger(tmp_path / "provenance.csv")
    assert [r["ef_search"] for r in rows] == ["500", "n/a"]


def test_a_ledger_with_other_columns_is_refused(tmp_path, monkeypatch):
    # appending a row with a column the header lacks would shift every field
    monkeypatch.setattr(pv, "_git", fake_git("abc123\n", ""))
    old = tmp_path / "provenance.csv"
    old.write_text(",".join(pv.COLUMNS[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="misalign"):
        pv.record([tmp_path / "a.csv"], argv=["scripts/x.py"])
