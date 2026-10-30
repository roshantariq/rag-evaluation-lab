"""Tests for the section-aware, recursive and semantic chunkers.

The two properties everything rests on are checked on randomised documents
for every chunker: the span invariant (each chunk is an exact slice of the
text, trimmed to word edges) and tiling (every word lands in exactly one
chunk, in order). Then each chunker's own rule is checked on its own.
"""

from __future__ import annotations

import random

import pytest

from rageval.chunking.base import chunk_document, word_spans
from rageval.chunking.strategies import (
    LINE, PARAGRAPH, SENTENCE, WORD, WordIndex, recursive_chunks, section_chunks,
    semantic_cuts, sentence_segments,
)

count = lambda s: int(len(s.split()) * 1.35) or 1  # noqa: E731 - offline tokenizer

VOCAB = "radar nowcasting precipitation ERA5 skill RMSE model forecast the of".split()


def random_doc(rng: random.Random) -> tuple[str, list[int]]:
    """Sentences, occasional line and paragraph breaks, and section starts."""
    parts, starts, pos = [], [], 0
    for s in range(rng.randint(1, 60)):
        sent = " ".join(rng.choice(VOCAB) for _ in range(rng.randint(1, 30)))
        sent = sent[0].upper() + sent[1:] + rng.choice([".", ".", "?", ""])
        sep = rng.choice([" ", " ", " ", "\n", "\n\n"])
        if rng.random() < 0.1:
            starts.append(pos)
        parts.append(sent + sep)
        pos += len(sent) + len(sep)
    return "".join(parts), starts


def assert_tiles(text: str, spans: list[tuple[int, int]]):
    words = word_spans(text)
    covered = []
    for a, b in spans:
        assert text[a:b] and text[a:b].strip() == text[a:b]
        covered += [w for w in words if a <= w[0] and w[1] <= b]
    assert covered == words, "every word exactly once, in order"
    assert all(b1 <= a2 for (_, b1), (a2, _) in zip(spans, spans[1:]))


@pytest.mark.parametrize("target", [8, 40, 200])
def test_every_chunker_tiles_random_documents(target):
    rng = random.Random(target)
    for _ in range(80):
        text, starts = random_doc(rng)
        assert_tiles(text, section_chunks(text, starts, target, target // 4, count))
        assert_tiles(text, recursive_chunks(text, target, count))
        idx, segs = sentence_segments(text, target, count)
        costs = [idx.cost(a, b) for a, b in segs]
        dist = [rng.random() for _ in range(len(segs) - 1)]
        groups = semantic_cuts(costs, dist, target // 2, target * 3 // 2)
        assert_tiles(text, [idx.char_span(segs[g0][0], segs[g1 - 1][1]) for g0, g1 in groups])


@pytest.mark.parametrize("fn", [
    lambda t: section_chunks(t, [0], 50, 12, count),
    lambda t: recursive_chunks(t, 50, count),
])
def test_empty_and_whitespace(fn):
    assert fn("") == [] and fn("  \n ") == []


# --- boundary strength ------------------------------------------------------

@pytest.mark.parametrize("text,word,expected", [
    # `word` is the index of the word after the boundary being checked
    ("Rain fell. The model", 2, SENTENCE),
    ("Rain fell? The model", 2, SENTENCE),
    ("as in Fig. The model", 3, WORD),        # abbreviation
    ("Smith et al. The model", 3, WORD),
    ("by J. Smith", 2, WORD),                 # an initial, not a sentence end
    ("rain fell. the model", 2, WORD),        # next word lower case
    ("rain fell\nThe model", 2, LINE),
    ("rain fell\n\nThe model", 2, PARAGRAPH),
])
def test_boundary_levels(text, word, expected):
    assert WordIndex(text, count).levels[word] == expected


# --- section-aware ------------------------------------------------------------

def test_section_chunks_never_cross_a_section():
    a = "Alpha words here. " * 30
    b = "Beta words here. " * 30
    text = a + b
    spans = section_chunks(text, [0, len(a)], 25, 6, count)
    assert all(e <= len(a) or s >= len(a) for s, e in spans)


def test_tiny_section_joins_the_next():
    head = "Methods "
    body = "Gamma words here. " * 30
    text = "Intro words here. " * 30 + head + body
    start_head = text.index(head)
    spans = section_chunks(text, [0, start_head, start_head + len(head)], 25, 6, count)
    assert not any(text[s:e] == "Methods" for s, e in spans)
    assert any(text[s:e].startswith("Methods Gamma") for s, e in spans)


def test_long_section_windows_are_balanced():
    text = " ".join(f"w{i}" for i in range(1000))
    idx = WordIndex(text, count)
    sizes = [idx.cost(*_words(idx, s, e)) for s, e in section_chunks(text, [0], 100, 25, count)]
    assert max(sizes) - min(sizes) <= 2


def _words(idx, s, e):
    starts = [a for a, _ in idx.spans]
    return starts.index(s), [b for _, b in idx.spans].index(e) + 1


# --- recursive ----------------------------------------------------------------

def test_recursive_prefers_sentence_ends():
    text = " ".join(f"Sentence {i} has a few more words in it." for i in range(40))
    for s, e in recursive_chunks(text, 30, count):
        assert text[s:e].endswith(".")


def test_recursive_respects_the_limit_where_it_can():
    text = " ".join(f"Sentence {i} has a few more words in it." for i in range(40))
    idx = WordIndex(text, count)
    for s, e in recursive_chunks(text, 30, count):
        assert idx.cost(*_words(idx, s, e)) <= 30


def test_recursive_without_boundaries_falls_back_to_words():
    text = " ".join(f"w{i}" for i in range(300))
    assert_tiles(text, recursive_chunks(text, 40, count))


# --- semantic -----------------------------------------------------------------

def test_semantic_cuts_at_the_largest_distance_in_the_window():
    costs = [10] * 10
    dist = [0.1] * 9
    dist[5] = 0.9                       # between segments 5 and 6
    assert semantic_cuts(costs, dist, 30, 80)[0] == (0, 6)


def test_semantic_ignores_a_large_distance_outside_the_window():
    costs = [10] * 10
    dist = [0.1] * 9
    dist[0] = 0.99                      # would make a 10-token chunk
    first = semantic_cuts(costs, dist, 30, 80)[0]
    assert first != (0, 1) and 3 <= first[1] <= 8


def test_semantic_falls_back_when_a_segment_jumps_the_window():
    costs = [10, 200, 10, 10]
    assert semantic_cuts(costs, [0.5, 0.5, 0.5], 30, 80)[0] == (0, 2)


def test_semantic_distance_count_is_checked():
    with pytest.raises(ValueError):
        semantic_cuts([10, 10], [0.1, 0.2], 5, 15)


def test_long_sentences_are_split_for_the_embedder():
    text = " ".join(f"n{i}" for i in range(500)) + ". Short one."
    idx, segs = sentence_segments(text, 64, count)
    assert all(idx.cost(a, b) <= 64 for a, b in segs)


# --- dispatch -------------------------------------------------------------------

DOC = {"arxiv_id": "x", "title": "t", "published": "",
       "text": "Intro text here. " * 40 + "Method text here. " * 40,
       "sections": [{"heading": "Intro", "text": "Intro text here. " * 40},
                    {"heading": "Method", "text": "Method text here. " * 40}]}


@pytest.mark.parametrize("strategy", ["section_512", "recursive_512", "fixed_512"])
def test_chunk_document_dispatch_round_trips(strategy):
    chunks = chunk_document(DOC, strategy, target_tokens=30, count_tokens=count)
    assert chunks and all(DOC["text"][c.char_start:c.char_end] == c.text for c in chunks)
    assert all(c.strategy == strategy for c in chunks)


def test_section_dispatch_uses_the_section_boundary():
    chunks = chunk_document(DOC, "section_512", target_tokens=30, count_tokens=count)
    cut = DOC["text"].index("Method")
    assert all(c.char_end <= cut or c.char_start >= cut for c in chunks)


def test_semantic_dispatch_is_refused():
    with pytest.raises(ValueError):
        chunk_document(DOC, "semantic", target_tokens=30, count_tokens=count)


def test_fixed_dispatch_is_unchanged():
    from rageval.chunking.base import fixed_size_chunks
    chunks = chunk_document(DOC, "fixed_512", target_tokens=30, count_tokens=count)
    assert [(c.char_start, c.char_end) for c in chunks] == \
        list(fixed_size_chunks(DOC["text"], 30, 0, count))
