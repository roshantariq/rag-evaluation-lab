"""Tests for script 30's comparison: which field differs is what locates a cause."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "extraction_hashes", Path(__file__).parents[1] / "scripts" / "30_extraction_hashes.py")
eh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eh)


def write(dirpath: Path, name: str, text: str, title: str = "t") -> None:
    dirpath.mkdir(exist_ok=True)
    (dirpath / f"{name}.json").write_text(
        json.dumps({"arxiv_id": name, "title": title, "text": text}), encoding="utf-8")


def test_identical_runs_match_on_every_field(tmp_path):
    for d in ("a", "b"):
        write(tmp_path / d, "x1", "same text")
    df = eh.compare(eh.scan(tmp_path / "a", None), eh.scan(tmp_path / "b", None))
    assert df.interim_match.tolist() == ["True"] and df.text_match.tolist() == ["True"]
    assert df.pdf_match.tolist() == ["n/a"]


def test_metadata_change_keeps_text_match(tmp_path):
    # a changed title alters the file but not what the gold spans anchor to
    write(tmp_path / "a", "x1", "same text", title="old")
    write(tmp_path / "b", "x1", "same text", title="new")
    row = eh.compare(eh.scan(tmp_path / "a", None), eh.scan(tmp_path / "b", None)).iloc[0]
    assert (row.interim_match, row.text_match) == ("False", "True")


def test_missing_and_extra_files_are_reported(tmp_path):
    write(tmp_path / "a", "x1", "t")
    write(tmp_path / "b", "x2", "t")
    df = eh.compare(eh.scan(tmp_path / "a", None), eh.scan(tmp_path / "b", None))
    assert set(zip(df.interim_file, df.in_reference, df.in_new)) == {
        ("x1.json", True, False), ("x2.json", False, True)}


def test_pdf_hash_is_compared_when_present(tmp_path):
    write(tmp_path / "a", "x1", "t")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "x1.pdf").write_bytes(b"%PDF-1")
    ref = eh.scan(tmp_path / "a", tmp_path / "raw")
    (tmp_path / "raw" / "x1.pdf").write_bytes(b"%PDF-2")
    row = eh.compare(ref, eh.scan(tmp_path / "a", tmp_path / "raw")).iloc[0]
    assert (row.text_match, row.pdf_match) == ("True", "False")
