"""Tests for gold set validation.

The span/quote check is the safety net for twelve hours of manual work:
a mistyped offset here silently corrupts every retrieval metric downstream.
"""

from rageval.evaluation.gold import Evidence, GoldQuestion, validate_question, validate_set

TEXT = ("The ConvLSTM extends the fully connected LSTM to have convolutional structures "
        "in both the input-to-state and state-to-state transitions. " * 3)
GC = "GraphCast uses a multi-mesh graph representation."
TEXTS = {"1506.04214v2": TEXT, "2212.12794v2": GC}
QUOTE = TEXT[4:40]


def q(**kw):
    base = dict(id="q", question="Q?", question_type="factual", reference_answer="A",
                evidence=[Evidence("1506.04214v2", 4, 40, QUOTE)], verified=True)
    return GoldQuestion(**{**base, **kw})


class TestSpanVerification:
    def test_matching_quote_passes(self):
        assert validate_question(q(), TEXTS) == []

    def test_mismatched_quote_is_caught(self):
        bad = q(evidence=[Evidence("1506.04214v2", 4, 40, "COMPLETELY DIFFERENT")])
        assert any("quote does not match" in p for p in validate_question(bad, TEXTS))

    def test_off_by_one_is_caught(self):
        # The realistic error: a quote typed by hand rather than copied.
        bad = q(evidence=[Evidence("2212.12794v2", 0, 20, "GraphCast uses a mult")])
        assert any("quote does not match" in p for p in validate_question(bad, TEXTS))

    def test_span_past_end_of_document(self):
        bad = q(evidence=[Evidence("1506.04214v2", 0, 999999, "x")])
        assert any("outside document" in p for p in validate_question(bad, TEXTS))

    def test_inverted_span(self):
        bad = q(evidence=[Evidence("1506.04214v2", 100, 50, "x")])
        assert any("inverted" in p for p in validate_question(bad, TEXTS))

    def test_unknown_paper(self):
        bad = q(evidence=[Evidence("9999.99999v1", 0, 10, "x")])
        assert any("not in corpus" in p for p in validate_question(bad, TEXTS))

    def test_whitespace_differences_tolerated(self):
        loose = q(evidence=[Evidence("1506.04214v2", 4, 40, "  " + " ".join(QUOTE.split()) + " ")])
        assert validate_question(loose, TEXTS) == []


class TestStructuralRules:
    def test_unanswerable_must_have_no_evidence(self):
        bad = q(question_type="unanswerable")
        assert any("has evidence" in p for p in validate_question(bad, TEXTS))

    def test_unanswerable_without_evidence_is_valid(self):
        assert validate_question(q(question_type="unanswerable", evidence=[]), TEXTS) == []

    def test_answerable_must_have_evidence(self):
        assert any("no evidence" in p for p in validate_question(q(evidence=[]), TEXTS))

    def test_multi_hop_needs_two_spans(self):
        assert any("at least two" in p
                   for p in validate_question(q(question_type="multi_hop"), TEXTS))

    def test_multi_hop_needs_two_papers(self):
        bad = q(question_type="multi_hop",
                evidence=[Evidence("1506.04214v2", 4, 40, QUOTE),
                          Evidence("1506.04214v2", 0, 10, TEXT[0:10])])
        assert any("one paper" in p for p in validate_question(bad, TEXTS))

    def test_unverified_is_flagged(self):
        assert any("not marked verified" in p for p in validate_question(q(verified=False), TEXTS))


class TestSetLevel:
    def test_duplicate_ids_caught(self):
        r = validate_set([q(id="a"), q(id="a", question="different")], TEXTS)
        assert any("duplicate question id" in p for p in r["per_question"]["a"])

    def test_duplicate_text_caught_ignoring_case_and_space(self):
        r = validate_set([q(id="a", question="How does X work?"),
                          q(id="b", question="  how does x WORK?  ")], TEXTS)
        assert any("duplicate question text" in p for p in r["per_question"]["b"])

    def test_counts_and_coverage(self):
        r = validate_set([q(id="a", question="First question?"),
                          q(id="b", question="Second question?",
                            question_type="unanswerable", evidence=[])], TEXTS)
        assert r["counts"]["factual"] == 1 and r["counts"]["unanswerable"] == 1
        assert r["n_ok"] == 2