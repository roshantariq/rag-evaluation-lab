"""End-to-end RAG: retrieve, generate, and record what happened.

Every answer carries the retrieval hits and parsed citations that produced
it, so faithfulness and citation accuracy can be scored later without
re-running generation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from rageval.generate.prompts import ABSTENTION_SENTINEL, build_prompt

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class RAGAnswer:
    question: str
    answer: str
    strategy: str
    k: int
    hits: list = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    abstained: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    retrieval_s: float = 0.0
    generation_s: float = 0.0

    def to_row(self) -> dict:
        """Flat record for results tables - excludes the bulky hit texts."""
        return {
            "question": self.question,
            "answer": self.answer,
            "strategy": self.strategy,
            "k": self.k,
            "abstained": self.abstained,
            "n_citations": len(self.citations),
            "cited_chunk_ids": "|".join(self.cited_chunk_ids),
            "retrieved_chunk_ids": "|".join(h.chunk_id for h in self.hits),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached": self.cached,
            "retrieval_s": round(self.retrieval_s, 3),
            "generation_s": round(self.generation_s, 3),
        }


def parse_citations(answer: str, n_hits: int) -> list[int]:
    """Extract [n] markers, de-duplicated, ordered, and range-checked.

    A model citing [7] when five passages were supplied is a real failure
    mode; dropping it silently would hide it from citation accuracy.
    """
    seen, out = set(), []
    for m in _CITATION.finditer(answer):
        idx = int(m.group(1))
        if 1 <= idx <= n_hits and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def detect_abstention(answer: str) -> bool:
    stripped = answer.strip().strip('".')
    return stripped.upper().startswith(ABSTENTION_SENTINEL)


class RAGPipeline:
    def __init__(self, store, encoder, llm, k: int = 5, strategy: str = "abstain"):
        self.store = store
        self.encoder = encoder
        self.llm = llm
        self.k = k
        self.strategy = strategy

    def retrieve(self, question: str, k: int | None = None) -> list:
        import time

        t0 = time.perf_counter()
        hits = self.store.query(self.encoder.encode_query(question), k=k or self.k)
        return hits, time.perf_counter() - t0

    def answer(self, question: str, k: int | None = None,
               strategy: str | None = None) -> RAGAnswer:
        import time

        k = k or self.k
        strategy = strategy or self.strategy

        hits, retrieval_s = self.retrieve(question, k)
        system, user = build_prompt(question, hits, strategy)

        t0 = time.perf_counter()
        resp = self.llm.complete(system, user)
        generation_s = time.perf_counter() - t0

        citations = parse_citations(resp.text, len(hits))
        return RAGAnswer(
            question=question,
            answer=resp.text,
            strategy=strategy,
            k=k,
            hits=hits,
            citations=citations,
            cited_chunk_ids=[hits[i - 1].chunk_id for i in citations],
            abstained=detect_abstention(resp.text),
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cached=resp.cached,
            retrieval_s=retrieval_s,
            generation_s=generation_s,
        )