"""Provider-agnostic LLM client with response caching.

Two reasons the provider is abstracted. First, Azure OpenAI is the
deployment this mirrors professionally, and the swap should be a config
change rather than a rewrite. Second, RAGAS evaluation multiplies calls by
roughly five per answer, so a re-run without caching is a real bill.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key       TEXT PRIMARY KEY,
    payload   TEXT NOT NULL,
    created   REAL NOT NULL
)
"""


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ResponseCache:
    """Keyed on the full request, so any prompt change is a cache miss."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(_SCHEMA)
        self.conn.commit()

    @staticmethod
    def key(model: str, temperature: float, system: str, user: str) -> str:
        blob = json.dumps([model, temperature, system, user], sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> LLMResponse | None:
        row = self.conn.execute(
            "SELECT payload FROM responses WHERE key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        return LLMResponse(**data, cached=True)

    def put(self, key: str, resp: LLMResponse) -> None:
        payload = json.dumps({
            "text": resp.text,
            "model": resp.model,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "latency_s": resp.latency_s,
        })
        self.conn.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?)", (key, payload, time.time())
        )
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]


class LLMClient:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 700,
        cache_path: Path | None = None,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = ResponseCache(cache_path) if cache_path else None
        self._client = None
        self.calls_made = 0
        self.tokens_used = 0

    @property
    def client(self):
        """Constructed lazily so a fully cached run needs no API key at all."""
        if self._client is None:
            if self.provider == "openai":
                from openai import OpenAI

                key = os.getenv("OPENAI_API_KEY")
                if not key:
                    raise RuntimeError(
                        "OPENAI_API_KEY is not set. Copy .env.example to .env "
                        "and add your key, or run against a warm cache."
                    )
                self._client = OpenAI(api_key=key)
            elif self.provider == "azure":
                from openai import AzureOpenAI

                self._client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                )
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        return self._client

    def complete(self, system: str, user: str) -> LLMResponse:
        key = None
        if self.cache:
            key = ResponseCache.key(self.model, self.temperature, system, user)
            hit = self.cache.get(key)
            if hit:
                return hit

        t0 = time.perf_counter()
        raw = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        resp = LLMResponse(
            text=(raw.choices[0].message.content or "").strip(),
            model=self.model,
            prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
            completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
            cached=False,
            latency_s=round(time.perf_counter() - t0, 3),
        )
        self.calls_made += 1
        self.tokens_used += resp.total_tokens
        if self.cache and key:
            self.cache.put(key, resp)
        return resp