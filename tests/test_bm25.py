"""Tests for lexical retrieval and rank fusion.

The tokenizer tests are not decoration. Sweep 2 carries a pre-registered
prediction naming four questions that hinge on four specific strings
(`xLSTM`, `ConvLSTM`, `ECMWF-IFS`, `B-MSE`, "9.4 percent"). A silent change
to tokenization would decide that prediction without anyone noticing, so
each of those strings has a test that names the question it belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rageval.retrieve.bm25 import BM25Retriever, sub_tokens, tokenize
from rageval.retrieve.fusion import provenance, reciprocal_rank_fusion


@dataclass
class Rec:
    """Minimal stand-in for the store's Retrieved - the retriever must not
    care which class it is handed."""
    chunk_id: str
    text: str = ""
    score: float = 0.0
    rank: int = 0


# --- tokenizer: the four strings the prediction rests on -------------------

def test_hyphenated_acronym_keeps_whole_and_parts():
    # f015 turns on B-MSE. Splitting to "b" + "mse" alone would lose the
    # compound; keeping only "b-mse" would miss a paper writing "B MSE".
    toks = tokenize("the B-MSE loss")
    assert "b-mse" in toks
    assert "mse" in toks
    assert "b" not in toks  # single characters are noise, not signal


def test_camel_case_splits_and_survives_whole():
    # f013 names xLSTM and ConvLSTM and sits at oracle rank 1.
    assert set(tokenize("xLSTM")) == {"xlstm", "lstm"}
    assert set(tokenize("ConvLSTM")) == {"convlstm", "conv", "lstm"}


def test_convlstm_and_conv_lstm_share_terms():
    # The corpus spells it both ways; they must be able to match.
    a, b = set(tokenize("ConvLSTM")), set(tokenize("Conv-LSTM"))
    assert {"conv", "lstm"} <= a & b


def test_institutional_acronym_splits_on_hyphen():
    # m010 quotes "ECMWF-IFS" near-verbatim from its target passage.
    toks = set(tokenize("ECMWF-IFS"))
    assert {"ecmwf-ifs", "ecmwf", "ifs"} == toks


def test_decimal_number_is_one_token_and_does_not_leak_digits():
    # m010 also carries "9.4 percent".
    assert tokenize("9.4 percent") == ["9.4", "percent"]


def test_variable_code_stays_whole():
    # Z500 must not emit "500", which would match every epoch count and
    # parameter total in the corpus.
    assert tokenize("Z500") == ["z500"]
    assert tokenize("T2m") == ["t2m"]


# --- tokenizer: the mundane failures that break lexical matching -----------

def test_trailing_punctuation_is_stripped():
    # "levels." must be the same term as "levels", or a sentence-final
    # match is silently lost.
    assert tokenize("37 vertical levels.") == ["37", "vertical", "levels"]


def test_parenthesised_and_quoted_terms_match_bare_ones():
    assert tokenize('("GraphCast")') == tokenize("GraphCast")


def test_case_is_folded():
    assert tokenize("RMSE") == tokenize("rmse")


def test_empty_and_symbol_only_text_yield_no_terms():
    assert tokenize("") == []
    assert tokenize("   ≤ √ ± —  ") == []


def test_sub_tokens_drops_pure_digit_parts():
    # Digit runs are split from letter runs, then dropped for carrying no
    # lexical signal: "500" would match every epoch count in the corpus.
    assert sub_tokens("A-500") == []
    assert sub_tokens("A-500nm") == ["nm"]


# --- BM25 ------------------------------------------------------------------

CORPUS = [
    Rec("c1", "The ConvLSTM network extends the fully connected LSTM to have "
               "convolutional structures in both the input-to-state and "
               "state-to-state transitions."),
    Rec("c2", "We train the model with a balanced mean squared error, B-MSE, "
               "which weights rare heavy rainfall more strongly."),
    Rec("c3", "GraphCast forecasts at 0.25 degree resolution across 37 "
               "vertical levels of the atmosphere."),
    Rec("c4", "Skill is evaluated against the ECMWF-IFS operational system "
               "and improves by 9.4 percent."),
    Rec("c5", "The encoder processes the input sequence and the decoder "
               "produces the forecast for the next time step."),
]


@pytest.fixture(scope="module")
def bm25():
    return BM25Retriever(CORPUS)


def test_rare_term_retrieves_its_own_chunk_first(bm25):
    assert bm25.query("what is B-MSE", k=3)[0].chunk_id == "c2"
    assert bm25.query("ECMWF-IFS", k=3)[0].chunk_id == "c4"


def test_ranks_are_one_based_and_contiguous(bm25):
    hits = bm25.query("ConvLSTM convolutional structures", k=5)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_scores_are_descending(bm25):
    scores = [h.score for h in bm25.query("levels resolution forecast", k=5)]
    assert scores == sorted(scores, reverse=True)


def test_no_shared_term_returns_nothing(bm25):
    # Zero-score results are withheld rather than returned in arbitrary
    # order - the fusion would otherwise treat noise as a ranking.
    assert bm25.query("photovoltaic cryptocurrency", k=5) == []


def test_query_of_only_symbols_returns_nothing(bm25):
    assert bm25.query("≤≥±", k=5) == []


def test_returned_records_are_copies(bm25):
    bm25.query("B-MSE", k=1)
    assert CORPUS[1].rank == 0 and CORPUS[1].score == 0.0


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        BM25Retriever([])


# --- reciprocal rank fusion ------------------------------------------------

def ranked(*ids):
    return [Rec(cid, rank=i) for i, cid in enumerate(ids, 1)]


def test_agreement_beats_a_single_first_place():
    # b is second on both lists; a is first on one and absent from the other.
    # 2/62 > 1/61, so consensus wins - the property RRF exists for.
    fused = reciprocal_rank_fusion([ranked("a", "b"), ranked("c", "b")])
    assert fused[0].chunk_id == "b"


def test_document_found_by_one_list_still_appears():
    fused = reciprocal_rank_fusion([ranked("a"), ranked("b")])
    assert {r.chunk_id for r in fused} == {"a", "b"}


def test_ranks_are_renumbered_from_one():
    fused = reciprocal_rank_fusion([ranked("a", "b", "c")])
    assert [r.rank for r in fused] == [1, 2, 3]


def test_ties_break_deterministically_not_by_dict_order():
    # Same rank in symmetric lists: order must not depend on which list was
    # walked first.
    one = reciprocal_rank_fusion([ranked("b"), ranked("a")])
    two = reciprocal_rank_fusion([ranked("a"), ranked("b")])
    assert [r.chunk_id for r in one] == [r.chunk_id for r in two] == ["a", "b"]


def test_top_k_truncates_after_fusing_not_before():
    fused = reciprocal_rank_fusion([ranked("a", "b"), ranked("c", "b")], top_k=1)
    assert len(fused) == 1 and fused[0].chunk_id == "b"


def test_large_k_flattens_and_small_k_sharpens():
    lists = [ranked("a", "b"), ranked("c", "b")]
    # With k=1, a's single first place (1/2) beats b's two seconds (2/3)? No:
    # 2/3 > 1/2, so b still wins; but the gap narrows as k grows.
    small = reciprocal_rank_fusion(lists, k=1)
    large = reciprocal_rank_fusion(lists, k=1000)
    gap = lambda f: f[0].score - f[1].score  # noqa: E731
    assert gap(small) > gap(large)


def test_rrf_k_must_be_positive():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([ranked("a")], k=0)


def test_empty_input_fuses_to_nothing():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# --- provenance ------------------------------------------------------------

def test_provenance_separates_lexical_gain_from_reordering():
    dense, sparse = ranked("a", "b"), ranked("b", "c")
    fused = reciprocal_rank_fusion([dense, sparse])
    counts = provenance(fused, {"dense": dense, "bm25": sparse})
    assert counts == {"dense only": 1, "bm25 only": 1, "both": 1}