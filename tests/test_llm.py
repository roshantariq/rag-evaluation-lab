"""Tests for LLMClient's cache, token budget and served-model record.

A fake OpenAI client stands in for the API, so nothing here makes a call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rageval.generate.llm import LLMClient, TokenBudgetExceeded


class FakeOpenAI:
    """Mimics client.chat.completions.create and counts the requests."""

    def __init__(self, served="gpt-4o-mini-2024-07-18", prompt_tokens=1000,
                 completion_tokens=100, finish_reason="stop"):
        self.requests = []
        self.served = served
        self.usage = SimpleNamespace(prompt_tokens=prompt_tokens,
                                     completion_tokens=completion_tokens)
        self.finish_reason = finish_reason
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        msg = SimpleNamespace(content=f"answer {len(self.requests)}")
        return SimpleNamespace(model=self.served, usage=self.usage,
                               choices=[SimpleNamespace(message=msg,
                                                        finish_reason=self.finish_reason)])


def client(tmp_path=None, **kw) -> tuple[LLMClient, FakeOpenAI]:
    fake = FakeOpenAI()
    llm = LLMClient(model="gpt-4o-mini-2024-07-18", max_tokens=700,
                    cache_path=(tmp_path / "r.sqlite") if tmp_path else None, **kw)
    llm._client = fake
    return llm, fake


def test_served_model_and_finish_reason_are_recorded(tmp_path):
    llm, _ = client(tmp_path)
    r = llm.complete("sys", "user")
    assert r.served_model == "gpt-4o-mini-2024-07-18" and r.finish_reason == "stop"


def test_served_model_survives_the_cache(tmp_path):
    llm, fake = client(tmp_path)
    llm.complete("sys", "user")
    again = llm.complete("sys", "user")
    assert again.cached and again.served_model == "gpt-4o-mini-2024-07-18"
    assert len(fake.requests) == 1


def test_a_resumed_run_pays_only_for_new_requests(tmp_path):
    first, _ = client(tmp_path)
    first.complete("sys", "q1")
    second, fake = client(tmp_path)          # a new process, same cache
    second.complete("sys", "q1")
    second.complete("sys", "q2")
    assert len(fake.requests) == 1 and second.calls_made == 1


def test_budget_refuses_before_the_call_is_made():
    llm, fake = client(max_total_tokens=2500)
    llm.complete("s", "u")                   # 1,100 used
    with pytest.raises(TokenBudgetExceeded):
        llm.complete("s", "x" * 3000)        # estimate 1,000 + 700 would pass 2,500
    assert len(fake.requests) == 1 and llm.tokens_used == 1100


def test_cached_answers_do_not_spend_the_budget(tmp_path):
    llm, _ = client(tmp_path)
    llm.complete("s", "u")
    capped, fake = client(tmp_path, max_total_tokens=1)
    assert capped.complete("s", "u").cached
    assert fake.requests == []


def test_no_budget_means_no_limit():
    llm, fake = client()
    for i in range(5):
        llm.complete("s", f"u{i}")
    assert len(fake.requests) == 5


def test_entries_written_before_this_field_still_load(tmp_path):
    import json
    import sqlite3
    import time

    from rageval.generate.llm import ResponseCache

    path = tmp_path / "r.sqlite"
    cache = ResponseCache(path)
    key = ResponseCache.key("m", 0.0, "s", "u")
    old = {"text": "t", "model": "m", "prompt_tokens": 1, "completion_tokens": 1,
           "latency_s": 0.1}
    sqlite3.connect(path).execute("INSERT INTO responses VALUES (?, ?, ?)",
                                  (key, json.dumps(old), time.time())).connection.commit()
    hit = cache.get(key)
    assert hit.text == "t" and hit.served_model == ""
