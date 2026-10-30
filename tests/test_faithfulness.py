"""Tests for resumable, fail-loud faithfulness judging. No network calls."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from rageval.evaluation import faithfulness as fa


class FakeMetric:
    """Stands in for ragas Faithfulness's two steps."""

    def __init__(self, statements, verdicts=None, meter=None, served="gpt-4o-2024-08-06",
                 fail=None):
        self.statements, self.verdicts = statements, verdicts or []
        self.meter, self.served, self.fail = meter, served, fail
        self.calls = 0

    def _use(self):
        self.calls += 1
        if self.fail:
            raise self.fail
        if self.meter:
            self.meter.record(self.served, 1000, 50)

    async def _create_statements(self, row, callbacks=None):
        self._use()
        return SimpleNamespace(statements=list(self.statements))

    async def _create_verdicts(self, row, statements, callbacks=None):
        self._use()
        return SimpleNamespace(statements=[SimpleNamespace(verdict=v) for v in self.verdicts])


def run(metric, meter=None, key="k1"):
    return asyncio.run(fa.judge(metric, key, "q?", "an answer [1]", ["ctx"], meter))


def test_judgment_keeps_its_working_and_served_version():
    meter = fa.UsageMeter()
    j = run(FakeMetric(["s1", "s2", "s3"], [1, 0, 1], meter), meter)
    assert (j.n_statements, j.n_supported, j.verdicts) == (3, 2, [1, 0, 1])
    assert j.score == pytest.approx(2 / 3)
    assert j.served_model == "gpt-4o-2024-08-06"
    assert meter.calls == 2 and meter.tokens_used == 2100


def test_no_claims_is_recorded_not_scored_and_not_a_failure():
    m = FakeMetric([])
    j = run(m)
    assert j.n_statements == 0 and j.score is None and m.calls == 1


def test_a_failed_call_propagates_instead_of_becoming_nan():
    with pytest.raises(RuntimeError):
        run(FakeMetric(["s"], [1], fail=RuntimeError("insufficient_quota")))


def test_verdict_count_mismatch_is_refused():
    with pytest.raises(ValueError):
        run(FakeMetric(["s1", "s2"], [1]))


def test_checkpoint_resumes_across_processes(tmp_path):
    path = tmp_path / "f.jsonl"
    first = fa.Checkpoint(path)
    first.add(run(FakeMetric(["s"], [1]), key="a"))
    second = fa.Checkpoint(path)
    assert "a" in second and "b" not in second
    assert second.done["a"].n_supported == 1


def test_checkpoint_refuses_an_incomplete_judgment(tmp_path):
    ckpt = fa.Checkpoint(tmp_path / "f.jsonl")
    with pytest.raises(ValueError):
        ckpt.add(fa.Judgment(key="x", n_statements=2, n_supported=1,
                             statements=["s1"], verdicts=[1]))
    assert not (tmp_path / "f.jsonl").exists()


def test_checkpoint_never_holds_a_nan(tmp_path):
    ckpt = fa.Checkpoint(tmp_path / "f.jsonl")
    ckpt.add(run(FakeMetric([]), key="empty"))
    line = (tmp_path / "f.jsonl").read_text(encoding="utf-8")
    assert "NaN" not in line and json.loads(line)["n_statements"] == 0


def test_answer_key_changes_with_answer_and_judge():
    k = fa.answer_key("f001", "naive", "text", "gpt-4o-2024-08-06")
    assert k != fa.answer_key("f001", "naive", "text2", "gpt-4o-2024-08-06")
    assert k != fa.answer_key("f001", "naive", "text", "gpt-4o-2024-11-20")
    assert k == fa.answer_key("f001", "naive", "text", "gpt-4o-2024-08-06")


def test_budget_stops_a_real_chat_model_before_any_request(monkeypatch):
    """raise_error must make LangChain re-raise; otherwise the budget is decoration."""
    from langchain_openai import ChatOpenAI

    meter = fa.UsageMeter(max_total_tokens=10)
    chat = ChatOpenAI(model="gpt-4o-2024-08-06", api_key="sk-test-not-real",
                      base_url="http://127.0.0.1:9", max_retries=0, callbacks=[meter])
    with pytest.raises(fa.JudgeBudgetExceeded):
        asyncio.run(chat.ainvoke("x" * 300))
    assert meter.calls == 0


def test_budget_stop_survives_the_ragas_wrapper(monkeypatch):
    """RAGAS must not turn the budget stop into a NaN or a retry."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9")
    meter = fa.UsageMeter(max_total_tokens=10)
    metric = fa.make_metric("gpt-4o-2024-08-06", meter)
    with pytest.raises(fa.JudgeBudgetExceeded):
        asyncio.run(fa.judge(metric, "k", "What is FSS?", "FSS is a score [1].", ["ctx"], meter))
    assert meter.calls == 0


@pytest.mark.parametrize("answer", [
    "It takes two days [2]. Confidence: high",
    "It takes two days [2]. \n\nConfidence: high",
    "It takes two days [2].\nconfidence: Medium.",
    "It takes two days [2].\nConfidence: low  ",
])
def test_confidence_line_is_not_judged(answer):
    assert fa.strip_confidence(answer) == "It takes two days [2]."


def test_answers_without_the_line_are_untouched():
    text = "Confidence intervals are reported [3]. The score is high [4]."
    assert fa.strip_confidence(text) == text
