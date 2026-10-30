"""Tests for 27_label_worksheet.py on a synthetic answers table."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "27_label_worksheet.py"
STRATEGIES = ("naive", "abstain", "abstain_confidence")


@pytest.fixture
def ws(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("ws27", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("LABEL_SHEET", "CHECK_SHEET"):
        monkeypatch.setattr(mod, name, tmp_path / getattr(mod, name).name)
    for name in ("LABEL_KEY", "CHECK_KEY"):
        monkeypatch.setattr(mod, name, tmp_path / getattr(mod, name).name)
    monkeypatch.setattr(mod, "TABLES_DIR", tmp_path)

    class Q:
        def __init__(self, qid, qt):
            self.id, self.question_type = qid, qt
            self.question, self.reference_answer = f"Question {qid}?", f"Reference for {qid}."

    gold = [Q("u001", "unanswerable"), Q("u002", "unanswerable"),
            Q("f001", "factual"), Q("a001", "ambiguous")]
    monkeypatch.setattr(mod, "load_gold", lambda path: gold)

    rows = []
    for q in gold:
        for s in STRATEGIES:
            answer = f"Answer to {q.id} by the model [1]."
            if s == "abstain_confidence":
                answer += "\n\nConfidence: high"
            refused = q.id == "f001" and s == "abstain"
            rows.append({"id": q.id, "question_type": q.question_type, "strategy": s,
                         "answer": "INSUFFICIENT CONTEXT" if refused else answer,
                         "abstained": str(refused), "n_citations": "0" if refused else "1"})
    answers = tmp_path / "answers.csv"
    pd.DataFrame(rows).to_csv(answers, index=False)
    mod._answers = answers
    return mod


def fill(path: Path, who: str, label_for) -> None:
    lines, check = path.read_text(encoding="utf-8").splitlines(), ""
    for i, line in enumerate(lines):
        if line.startswith("check:"):
            check = line.split(":", 1)[1].strip()
        if line.startswith(f"{who}:"):
            lines[i] = f"{who}: {label_for(check)}"
    path.write_text("\n".join(lines), encoding="utf-8")


def labels(check):
    return {"refusal": "REFUSAL", "ambiguity": "SINGLE"}.get(check, "ABSTAIN")


def test_worksheet_hides_strategy_and_confidence(ws):
    assert ws.build(ws._answers) == 0
    text = ws.LABEL_SHEET.read_text(encoding="utf-8")
    body = text.split("=== item", 1)[1]
    assert "Confidence: high" not in body
    for s in ("naive", "abstain_confidence", "strategy"):
        assert s not in body
    assert text.count("=== item") == 6


def test_shuffle_is_fixed_by_the_seed(ws, tmp_path):
    ws.build(ws._answers)
    first = ws.LABEL_KEY.read_text(encoding="utf-8")
    for p in (ws.LABEL_SHEET, ws.CHECK_SHEET):
        p.unlink()
    ws.build(ws._answers)
    assert ws.LABEL_KEY.read_text(encoding="utf-8") == first


def test_check_sheet_covers_refusals_and_every_ambiguous_answer(ws):
    ws.build(ws._answers)
    key = pd.read_csv(ws.CHECK_KEY)
    assert sorted(key[key.check == "refusal"]["id"]) == ["f001"]
    assert len(key[key.check == "ambiguity"]) == 3


def test_build_refuses_to_overwrite_labels(ws):
    ws.build(ws._answers)
    assert ws.build(ws._answers) == 2


def test_read_refuses_until_the_owner_has_labelled(ws):
    ws.build(ws._answers)
    for p in (ws.LABEL_SHEET, ws.CHECK_SHEET):
        fill(p, "claude", labels)
    assert ws.read() == 2
    assert not (ws.TABLES_DIR / "generation_labels.csv").exists()


def test_read_rejects_a_label_outside_the_rubric(ws):
    ws.build(ws._answers)
    for p in (ws.LABEL_SHEET, ws.CHECK_SHEET):
        fill(p, "claude", labels)
        fill(p, "owner", labels)
    fill(ws.LABEL_SHEET, "owner", lambda c: "PARTIAL")
    assert ws.read() == 2


def test_read_joins_the_key_and_the_owner_is_final(ws):
    ws.build(ws._answers)
    for p in (ws.LABEL_SHEET, ws.CHECK_SHEET):
        fill(p, "claude", labels)
        fill(p, "owner", labels)
    fill(ws.LABEL_SHEET, "owner", lambda c: "hallucinate")
    assert ws.read() == 0
    out = pd.read_csv(ws.TABLES_DIR / "generation_labels.csv")
    assert set(out["final"]) == {"HALLUCINATE"} and not out["agree"].any()
    assert sorted(out["strategy"].unique()) == sorted(STRATEGIES)
    checks = pd.read_csv(ws.TABLES_DIR / "generation_checks.csv")
    assert set(checks["check"]) == {"refusal", "ambiguity"}
