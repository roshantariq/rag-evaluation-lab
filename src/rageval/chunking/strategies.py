"""Section-aware, recursive and semantic chunkers.

Each differs from `fixed_size_chunks` in one respect: where the cuts fall.

    section    never cut across a section heading; split long sections
               into near-equal windows
    recursive  cut at the strongest boundary available near the size
               limit: paragraph break, then line break, then sentence end,
               then word
    semantic   cut at the sentence boundary, inside a size window, where
               adjacent sentences are least similar

Every chunker cuts only between whitespace-delimited words, as the fixed
chunker does, so `text[char_start:char_end] == chunk.text` holds by
construction and no chunk begins or ends mid-word. Every chunker also tiles
the document: no overlap and no word left out. Sizes are counted with the
same per-word token costs as the fixed chunker, so a target means the same
thing in all four.

Size is controlled deliberately. Sweep 1 showed that comparing chunkers of
different sizes measures size, and Chroma's chunking report found the same
of semantic chunkers whose threshold is relative. The protocol sets each
chunker's target so its median chunk length matches fixed_512 on the
corpus (docs/chunker-protocol.md).

The semantic chunker is split in two so the cut logic is testable without
a model: `sentence_segments` produces the units to embed, the caller embeds
them, and `semantic_cuts` chooses cuts from the costs and distances.
"""

from __future__ import annotations

import bisect
import math
import re
from typing import Callable, Sequence

from rageval.chunking.base import word_spans

# Boundary strength between two adjacent words, strongest first.
PARAGRAPH, LINE, SENTENCE, WORD = 0, 1, 2, 3

# Tokens ending in a full stop that do not end a sentence in this corpus.
_ABBREV = frozenset({
    "al.", "fig.", "figs.", "eq.", "eqs.", "e.g.", "i.e.", "vs.", "cf.", "sec.",
    "secs.", "tab.", "no.", "ref.", "refs.", "approx.", "resp.", "ch.", "app.",
    "dr.", "mr.", "ms.", "st.", "prof.", "eqn.", "def.", "thm.", "vol.", "pp.",
})
_SENT_END = re.compile(r"[.!?][\"'”’)\]]*$")


class WordIndex:
    """Word spans, cumulative token costs and boundary strengths for one text."""

    def __init__(self, text: str, count_tokens: Callable[[str], int]):
        self.text = text
        self.spans = word_spans(text)
        self.prefix = [0]
        for a, b in self.spans:
            self.prefix.append(self.prefix[-1] + max(count_tokens(text[a:b]), 1))
        # levels[k] is the strength of the boundary before word k (k >= 1).
        self.levels = [PARAGRAPH] + [self._level(k) for k in range(1, len(self.spans))]

    def __len__(self) -> int:
        return len(self.spans)

    def cost(self, i: int, j: int) -> int:
        """Tokens in words i..j-1."""
        return self.prefix[j] - self.prefix[i]

    def char_span(self, i: int, j: int) -> tuple[int, int]:
        return self.spans[i][0], self.spans[j - 1][1]

    def _level(self, k: int) -> int:
        gap = self.text[self.spans[k - 1][1]:self.spans[k][0]]
        if gap.count("\n") >= 2:
            return PARAGRAPH
        if "\n" in gap:
            return LINE
        prev = self.text[self.spans[k - 1][0]:self.spans[k - 1][1]]
        nxt = self.text[self.spans[k][0]:self.spans[k][1]]
        core = prev.rstrip(".!?\"'”’)]")
        if (_SENT_END.search(prev) and prev.lower() not in _ABBREV
                and len(core) > 1 and nxt[:1].isupper()):
            return SENTENCE
        return WORD


def _balanced(idx: WordIndex, i: int, j: int, target: int) -> list[tuple[int, int]]:
    """Split words i..j-1 into the fewest near-equal windows of at most ~target."""
    total = idx.cost(i, j)
    n = max(1, math.ceil(total / target))
    cuts = [i]
    for k in range(1, n):
        b = bisect.bisect_left(idx.prefix, idx.prefix[i] + k * total / n, cuts[-1] + 1, j)
        if cuts[-1] < b < j:
            cuts.append(b)
    cuts.append(j)
    return list(zip(cuts, cuts[1:]))


def _to_chars(idx: WordIndex, ranges) -> list[tuple[int, int]]:
    return [idx.char_span(i, j) for i, j in ranges if j > i]


# --- section-aware ----------------------------------------------------------

def section_chunks(text: str, section_starts: Sequence[int], target_tokens: int,
                   min_tokens: int, count_tokens: Callable[[str], int]) -> list[tuple[int, int]]:
    """Chunks that never span two sections.

    A section shorter than `min_tokens` (a heading with little or no body)
    is joined to the section after it, or to the one before if it is last;
    otherwise it would become a tiny chunk that BM25's length normalisation
    favours for reasons unrelated to content. Each resulting region is
    split into near-equal windows of at most about `target_tokens`.
    """
    idx = WordIndex(text, count_tokens)
    if not len(idx):
        return []
    starts = [a for a, _ in idx.spans]
    bounds = sorted({0, len(idx)} | {bisect.bisect_left(starts, s) for s in section_starts})
    bounds = [b for b in bounds if 0 <= b <= len(idx)]
    regions = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]

    merged: list[list[int]] = []
    pending = None
    for a, b in regions:
        a = pending if pending is not None else a
        if idx.cost(a, b) < min_tokens:
            pending = a
            continue
        pending = None
        merged.append([a, b])
    if pending is not None:
        if merged:
            merged[-1][1] = len(idx)
        else:
            merged.append([pending, len(idx)])

    out = []
    for a, b in merged:
        out += _balanced(idx, a, b, target_tokens)
    return _to_chars(idx, out)


# --- recursive ----------------------------------------------------------------

def recursive_chunks(text: str, target_tokens: int,
                     count_tokens: Callable[[str], int]) -> list[tuple[int, int]]:
    """Recursive splitting on boundary strength, then greedy merging.

    The span-level equivalent of the common RecursiveCharacterTextSplitter,
    with a sentence level added: this corpus keeps few line breaks
    (median 12 paragraph breaks per paper), so without it the splitter
    would fall straight to word windows and be the fixed chunker under
    another name.
    """
    idx = WordIndex(text, count_tokens)
    if not len(idx):
        return []

    def split(i: int, j: int, level: int) -> list[tuple[int, int]]:
        if idx.cost(i, j) <= target_tokens or j - i == 1:
            return [(i, j)]
        cuts = [k for k in range(i + 1, j) if idx.levels[k] <= level]
        if not cuts:
            return split(i, j, level + 1)
        pieces = list(zip([i] + cuts, cuts + [j]))
        out: list[tuple[int, int]] = []
        for a, b in pieces:
            out += [(a, b)] if idx.cost(a, b) <= target_tokens else split(a, b, level + 1)
        merged = [out[0]]
        for a, b in out[1:]:
            if idx.cost(merged[-1][0], b) <= target_tokens:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        return merged

    return _to_chars(idx, split(0, len(idx), PARAGRAPH))


# --- semantic -----------------------------------------------------------------

def sentence_segments(text: str, max_segment_tokens: int,
                      count_tokens: Callable[[str], int]) -> tuple[WordIndex, list[tuple[int, int]]]:
    """The units a semantic chunker embeds: sentences, with long runs split.

    A "sentence" longer than `max_segment_tokens` (a table or equation with
    no full stop) is split into near-equal windows, so no unit overflows
    the embedder's input and is silently truncated.
    """
    idx = WordIndex(text, count_tokens)
    if not len(idx):
        return idx, []
    cuts = [k for k in range(1, len(idx)) if idx.levels[k] <= SENTENCE]
    segs = []
    for a, b in zip([0] + cuts, cuts + [len(idx)]):
        segs += _balanced(idx, a, b, max_segment_tokens) \
            if idx.cost(a, b) > max_segment_tokens else [(a, b)]
    return idx, segs


def semantic_cuts(costs: Sequence[int], distances: Sequence[float],
                  min_tokens: int, max_tokens: int) -> list[tuple[int, int]]:
    """Group consecutive segments into chunks, cutting where topics shift.

    `distances[e]` is the embedding distance between segments e and e+1.
    From each chunk start, every cut leaving a chunk of min_tokens to
    max_tokens is a candidate, and the one with the largest distance wins;
    ties go to the earliest. If no cut lands in the window (one long
    segment jumps over it), the first cut reaching min_tokens is taken.
    Returns (first, last + 1) segment ranges.
    """
    m = len(costs)
    if len(distances) != max(m - 1, 0):
        raise ValueError(f"{len(distances)} distances for {m} segments")
    prefix = [0]
    for c in costs:
        prefix.append(prefix[-1] + c)
    out, pos = [], 0
    while pos < m:
        if prefix[m] - prefix[pos] <= max_tokens:
            out.append((pos, m))
            break
        best, best_d, fallback = None, -math.inf, None
        for e in range(pos + 1, m):
            size = prefix[e] - prefix[pos]
            if size >= min_tokens and fallback is None:
                fallback = e
            if size > max_tokens:
                break
            if size >= min_tokens and distances[e - 1] > best_d:
                best, best_d = e, distances[e - 1]
        cut = best or fallback or m
        out.append((pos, cut))
        pos = cut
    return out
