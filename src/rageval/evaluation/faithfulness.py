"""RAGAS faithfulness, judged resumably and failing loudly.

Implements the Phase 7 protocol's judging rules (docs/phase7-protocol.md,
amendment of 30 October 2026):

* Every score is appended to a checkpoint the moment it exists, so a run
  stopped by the $10 hard spend cap resumes without re-judging anything.
* A failed call stops the run. RAGAS turns failures into NaN by default; a
  NaN written as a score would make a capped run look finished and be
  skipped on resume. Nothing non-finite is ever written as a score.
* "The answer made no claims" is not a failure and not a score: it is
  recorded with n_statements = 0 and a blank score, and counted.
* Every judge call passes through UsageMeter, which refuses a call that
  could pass the token budget and records the version that replied.

The two RAGAS steps (statements, then verdicts) are called directly rather
than through `single_turn_ascore`, so the statements and per-statement
verdicts are kept for audit, not just the ratio.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler


class JudgeBudgetExceeded(RuntimeError):
    """Raised instead of a judge call that could pass the token budget."""


class UsageMeter(BaseCallbackHandler):
    """LangChain callback: token budget before each call, usage after it.

    `raise_error` makes LangChain re-raise from the callback instead of
    logging and carrying on, which is what lets the budget stop a run.
    """

    raise_error = True

    def __init__(self, max_total_tokens: int | None = None,
                 completion_allowance: int = 1500):
        self.max_total_tokens = max_total_tokens
        self.completion_allowance = completion_allowance
        self.tokens_used = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.served_models: set[str] = set()

    def check(self, prompt_chars: int) -> None:
        if self.max_total_tokens is None:
            return
        estimate = prompt_chars // 3 + self.completion_allowance
        if self.tokens_used + estimate > self.max_total_tokens:
            raise JudgeBudgetExceeded(
                f"judge budget {self.max_total_tokens:,} would be passed: "
                f"{self.tokens_used:,} used in {self.calls} calls, next estimated "
                f"at {estimate:,}. Scored answers are in the checkpoint.")

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.check(sum(len(str(m.content)) for batch in messages for m in batch))

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.check(sum(len(p) for p in prompts))

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.calls += 1
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)
        self.tokens_used = self.prompt_tokens + self.completion_tokens
        if model:
            self.served_models.add(model)

    def on_llm_end(self, response, **kwargs):
        out = response.llm_output or {}
        usage = out.get("token_usage") or {}
        self.record(out.get("model_name", ""), usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0))


@dataclass
class Judgment:
    key: str
    n_statements: int
    n_supported: int
    statements: list[str] = field(default_factory=list)
    verdicts: list[int] = field(default_factory=list)
    served_model: str = ""   # ";"-joined if one judgment saw more than one

    @property
    def score(self) -> float | None:
        """Supported share of statements; None when the answer made no claim."""
        return self.n_supported / self.n_statements if self.n_statements else None


_CONFIDENCE_LINE = re.compile(r"\s*Confidence:\s*(high|medium|low)\s*\.?\s*$", re.IGNORECASE)


def strip_confidence(answer: str) -> str:
    """Drop abstain_confidence's trailing self-rating before judging.

    The dry run showed the judge turning "Confidence: high" into the claim
    "the confidence level is high" and marking it unsupported, a penalty
    only one strategy pays. The line is a rating of the answer, not a
    claim about the papers; labelling removes it for the same reason.
    """
    return _CONFIDENCE_LINE.sub("", answer).rstrip()


def answer_key(qid: str, strategy: str, answer: str, judge_model: str) -> str:
    """Identifies one judgment. A changed answer or judge is a new key."""
    import hashlib

    blob = json.dumps([qid, strategy, answer, judge_model], sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


class Checkpoint:
    """Append-only JSONL of finished judgments, one line per answer."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, Judgment] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        j = Judgment(**json.loads(line))
                        self.done[j.key] = j

    def __contains__(self, key: str) -> bool:
        return key in self.done

    def add(self, j: Judgment) -> None:
        validate(j)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(j.__dict__, ensure_ascii=False) + "\n")
        self.done[j.key] = j


def validate(j: Judgment) -> None:
    """Refuse anything that is not a complete, finite judgment."""
    if j.n_statements != len(j.statements) or j.n_statements != len(j.verdicts):
        raise ValueError(f"{j.key}: {j.n_statements} statements but "
                         f"{len(j.statements)} texts and {len(j.verdicts)} verdicts")
    if any(v not in (0, 1) for v in j.verdicts):
        raise ValueError(f"{j.key}: verdicts must be 0 or 1, got {j.verdicts}")
    if j.n_supported != sum(j.verdicts):
        raise ValueError(f"{j.key}: n_supported does not match the verdicts")
    if j.score is not None and not math.isfinite(j.score):
        raise ValueError(f"{j.key}: non-finite score")


async def judge(metric, key: str, question: str, answer: str,
                contexts: list[str], meter: UsageMeter | None = None) -> Judgment:
    """Run RAGAS's two faithfulness steps and keep their working.

    Any exception from the judge propagates: there is no NaN path.
    """
    seen_before = set(meter.served_models) if meter else set()
    if meter:
        meter.served_models.clear()
    row = {"user_input": question, "response": answer, "retrieved_contexts": contexts}
    statements = (await metric._create_statements(row, callbacks=None)).statements
    if not statements:
        j = Judgment(key=key, n_statements=0, n_supported=0)
    else:
        verdicts = (await metric._create_verdicts(row, statements, callbacks=None)).statements
        if len(verdicts) != len(statements):
            raise ValueError(f"{key}: judge returned {len(verdicts)} verdicts for "
                             f"{len(statements)} statements")
        v = [1 if int(x.verdict) else 0 for x in verdicts]
        j = Judgment(key=key, n_statements=len(statements), n_supported=sum(v),
                     statements=list(statements), verdicts=v)
    if meter:
        j.served_model = ";".join(sorted(meter.served_models))
        meter.served_models |= seen_before
    return j


def make_metric(judge_model: str, meter: UsageMeter, max_tokens: int = 4096):
    """RAGAS Faithfulness over a pinned OpenAI model, metered."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    chat = ChatOpenAI(model=judge_model, temperature=0.0, max_tokens=max_tokens,
                      callbacks=[meter])
    return Faithfulness(llm=LangchainLLMWrapper(chat))
