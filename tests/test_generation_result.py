"""Tests for 28_generation_result.py: pinned constants, rule, end to end."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "28_generation_result.py"
spec = importlib.util.spec_from_file_location("gr28", SCRIPT)
gr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gr)
STRATEGIES = gr.STRATEGIES


def test_preregistered_constants():
    assert gr.FLOOR == 2 and gr.N_UNANSWERABLE == 16
    assert gr.BOOT == 10_000 and gr.SEED == 20260827
    assert gr.DECIDING_PAIR == ("naive", "abstain")
    assert gr.JUDGE_NOISE == 0.083 and gr.N_ANSWERS == 222


@pytest.mark.parametrize("naive, lo, hi, expected", [
    (2, 0.10, 0.30, "floor"),          # floor wins even with a clear interval
    (0, -0.1, 0.1, "floor"),
    (3, 0.0625, 0.25, "reduces"),
    (3, 0.0, 0.25, "not_separable"),   # zero on the boundary is not excluded
    (5, -0.25, -0.0625, "increases"),
])
def test_outcome_table(naive, lo, hi, expected):
    assert gr.classify(naive, lo, hi) == expected


def test_mcnemar_counts_discordant_questions():
    a, b = np.array([1, 1, 1, 0, 0]), np.array([0, 0, 1, 1, 0])
    only_a, only_b, p = gr.mcnemar(a, b)
    assert (only_a, only_b) == (2, 1) and p == 1.0


def test_paired_ci_is_reproducible_and_exact_when_constant():
    a, b = np.ones(16), np.zeros(16)
    est, lo, hi = gr.paired_ci(a, b, gr.draws_for(16))
    assert (est, lo, hi) == (1.0, 1.0, 1.0)
    r = np.random.default_rng(1).integers(0, 2, 16).astype(float)
    assert gr.paired_ci(r, b, gr.draws_for(16)) == gr.paired_ci(r, b, gr.draws_for(16))


def synthetic(tmp_path: Path, naive_h: int, abstain_h: int) -> Path:
    answers, labels, checks, faith = [], [], [], []
    for i in range(16):
        qid = f"u{i:03d}"
        for s in STRATEGIES:
            h = (s == "naive" and i < naive_h) or (s != "naive" and i < abstain_h)
            answers.append({"id": qid, "question_type": "unanswerable", "strategy": s,
                            "evidence": "unanswerable", "answer": "x\nConfidence: low",
                            "abstained": str(not h), "n_citations": 0, "cited_on_gold": 0,
                            "gold_spans_in_context": "", "gold_spans_cited": "",
                            "out_of_range_citations": 0, "served_model": "m-2024-07-18"})
            lab = "HALLUCINATE" if h else "ABSTAIN"
            labels.append({"item": len(labels) + 1, "id": qid, "strategy": s,
                           "claude": lab, "owner": lab, "final": lab, "agree": "True",
                           "note": ""})
    for i in range(4):
        qid, ev = f"f{i:03d}", "complete" if i < 2 else "incomplete"
        for s in STRATEGIES:
            refused = s == "abstain" and i == 3
            answers.append({"id": qid, "question_type": "factual", "strategy": s,
                            "evidence": ev, "answer": "y [1]\nConfidence: high",
                            "abstained": str(refused), "n_citations": 0 if refused else 1,
                            "cited_on_gold": 0 if refused else 1,
                            "gold_spans_in_context": "0", "gold_spans_cited": "" if refused else "0",
                            "out_of_range_citations": 0, "served_model": "m-2024-07-18"})
            if not refused:
                faith.append({"id": qid, "strategy": s, "evidence": ev, "key": "k",
                              "n_statements": 4, "n_supported": 3, "faithfulness": 0.75,
                              "judge_served_model": "j"})
    for s in STRATEGIES:
        checks.append({"item": len(checks) + 1, "id": "f003", "strategy": s,
                       "check": "ambiguity", "claude": "SINGLE", "owner": "SINGLE",
                       "final": "SINGLE", "agree": "True", "note": ""})
    checks.append({"item": len(checks) + 1, "id": "f003", "strategy": "abstain",
                   "check": "refusal", "claude": "REFUSAL", "owner": "REFUSAL",
                   "final": "REFUSAL", "agree": "True", "note": ""})
    for name, rows in (("generation_answers", answers), ("generation_labels", labels),
                       ("generation_checks", checks), ("generation_faithfulness", faith)):
        pd.DataFrame(rows).to_csv(tmp_path / f"{name}.csv", index=False)
    return tmp_path


def run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["28", "--tables", str(tmp_path)])
    assert gr.main() == 0
    return capsys.readouterr().out, pd.read_csv(tmp_path / "generation_result.csv")


def test_end_to_end_reduces(tmp_path, monkeypatch, capsys):
    out, res = run(synthetic(tmp_path, naive_h=8, abstain_h=1), monkeypatch, capsys)
    assert "OUTCOME: reduces" in out
    row = res[(res.measure == "hallucination_diff") & (res.arm == "naive-abstain")].iloc[0]
    assert row["value"] == pytest.approx(7 / 16) and row["lo"] > 0


def test_end_to_end_floor(tmp_path, monkeypatch, capsys):
    out, _ = run(synthetic(tmp_path, naive_h=2, abstain_h=0), monkeypatch, capsys)
    assert "OUTCOME: floor" in out


def test_end_to_end_not_separable(tmp_path, monkeypatch, capsys):
    out, _ = run(synthetic(tmp_path, naive_h=4, abstain_h=3), monkeypatch, capsys)
    assert "OUTCOME: not_separable" in out


def test_refusals_split_by_evidence_and_faithfulness_within_noise(tmp_path, monkeypatch, capsys):
    _, res = run(synthetic(tmp_path, naive_h=8, abstain_h=1), monkeypatch, capsys)
    inc = res[(res.measure == "refusal_rate_incomplete") & (res.arm == "abstain")].iloc[0]
    assert inc["count"] == 1 and inc["n"] == 2
    diff = res[res.measure == "diff_both_answered"].iloc[0]
    assert diff["outcome"] == "within_judge_noise"
