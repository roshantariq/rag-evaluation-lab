"""Tests for citation parsing, abstention detection and prompt construction.

All run against a stub LLM, so the suite needs no API key and costs nothing.
"""

from dataclasses import dataclass

import pytest

from rageval.generate.prompts import ABSTENTION_SENTINEL, build_prompt
from rageval.pipeline import RAGPipeline, detect_abstention, parse_citations


@dataclass
class Hit:
    chunk_id: str
    arxiv_id: str
    text: str
    title: str = "Paper"
    char_start: int = 0
    char_end: int = 100
    section: str = ""
    score: float = 0.9
    rank: int = 1


HITS = [Hit(f"id{i}", f"paper{i}", f"Passage {i} about ConvLSTM.", rank=i) for i in range(1, 6)]


class TestCitationParsing:
    def test_extracts_markers(self):
        assert parse_citations("ConvLSTM uses convolutions [1], beating ROVER [3].", 5) == [1, 3]

    def test_deduplicates_preserving_order(self):
        assert parse_citations("[2] then [1] then [2] again", 5) == [2, 1]

    def test_out_of_range_citation_is_dropped(self):
        # Citing [7] when five passages were supplied is a real failure mode.
        assert parse_citations("As shown in [7] and [2].", 5) == [2]

    def test_zero_is_not_a_valid_citation(self):
        assert parse_citations("see [0] and [1]", 5) == [1]

    def test_no_citations(self):
        assert parse_citations("No citations here.", 5) == []


class TestAbstentionDetection:
    @pytest.mark.parametrize("text", [
        ABSTENTION_SENTINEL,
        f'"{ABSTENTION_SENTINEL}"',
        f"{ABSTENTION_SENTINEL}.",
        ABSTENTION_SENTINEL.lower(),
    ])
    def test_sentinel_forms_detected(self, text):
        assert detect_abstention(text)

    def test_real_answer_not_flagged(self):
        assert not detect_abstention("ConvLSTM replaces matrix multiplication [1].")

    def test_mid_answer_mention_not_flagged(self):
        assert not detect_abstention("The context is insufficient context here, but [1] says X.")


class TestPromptStrategies:
    def test_naive_omits_the_abstention_instruction(self):
        assert ABSTENTION_SENTINEL not in build_prompt("q", HITS, "naive")[0]

    def test_abstain_includes_it(self):
        assert ABSTENTION_SENTINEL in build_prompt("q", HITS, "abstain")[0]

    def test_confidence_variant_differs_only_by_confidence(self):
        assert "Confidence" in build_prompt("q", HITS, "abstain_confidence")[0]
        assert "Confidence" not in build_prompt("q", HITS, "abstain")[0]

    def test_context_is_numbered_for_citation(self):
        _, user = build_prompt("What is FSS?", HITS, "abstain")
        assert "[1] (paper1" in user and "[5] (paper5" in user
        assert "What is FSS?" in user

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            build_prompt("q", HITS, "nonsense")

    def test_empty_retrieval_is_handled(self):
        assert "(no passages retrieved)" in build_prompt("q", [], "abstain")[1]


class TestPipeline:
    class StubStore:
        def query(self, emb, k=5):
            return HITS[:k]

    class StubEncoder:
        def encode_query(self, q):
            return [0.0]

    class StubLLM:
        def complete(self, system, user):
            from rageval.generate.llm import LLMResponse
            if "unanswerable" in user:
                return LLMResponse(text=ABSTENTION_SENTINEL, model="stub")
            return LLMResponse(text="ConvLSTM uses convolutions [1], beating ROVER [3].",
                               model="stub", prompt_tokens=120, completion_tokens=18)

    def pipe(self, **kw):
        return RAGPipeline(self.StubStore(), self.StubEncoder(), self.StubLLM(), **kw)

    def test_citations_map_to_chunk_ids(self):
        a = self.pipe(k=5).answer("How does ConvLSTM work?")
        assert a.citations == [1, 3]
        assert a.cited_chunk_ids == ["id1", "id3"]

    def test_abstention_recorded(self):
        a = self.pipe(k=5).answer("something unanswerable")
        assert a.abstained and a.citations == []

    def test_k_override(self):
        a = self.pipe(k=5).answer("How does ConvLSTM work?", k=3)
        assert a.k == 3 and len(a.hits) == 3

    def test_row_is_flat_and_complete(self):
        row = self.pipe(k=5).answer("How does ConvLSTM work?").to_row()
        assert row["retrieved_chunk_ids"].count("|") == 4
        assert row["n_citations"] == 2