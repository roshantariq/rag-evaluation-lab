"""Chunking with exact character-span provenance.

Every chunk records where it came from in the source document as a
(char_start, char_end) span. This is what makes the gold evaluation set
survive re-chunking: chunk IDs differ across strategies, but a passage's
character span does not, so relevance can be judged by span overlap for
any chunker.

Boundaries are therefore derived from real word positions in the text,
never from token-decode arithmetic - the offsets are exact by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterator

_WORD = re.compile(r"\S+")

# Rough tokens-per-word for English technical prose. Used only when tiktoken
# is unavailable; chunk sizes shift slightly, spans stay exact either way.
_TOKENS_PER_WORD = 1.35


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit, with provenance back to the source document."""

    arxiv_id: str
    text: str
    char_start: int
    char_end: int
    section: str
    strategy: str
    title: str = ""
    published: str = ""

    @property
    def chunk_id(self) -> str:
        """Deterministic and span-derived, so it is stable across rebuilds."""
        return f"{self.arxiv_id}:{self.char_start}-{self.char_end}"

    def overlaps(self, start: int, end: int) -> bool:
        """True when this chunk shares any characters with an evidence span."""
        return self.char_start < end and start < self.char_end

    def overlap_fraction(self, start: int, end: int) -> float:
        """Share of the evidence span this chunk covers."""
        span = max(end - start, 1)
        return max(0, min(self.char_end, end) - max(self.char_start, start)) / span


def get_token_counter() -> Callable[[str], int]:
    """Return a token counter, preferring tiktoken but never requiring it.

    tiktoken downloads its encoding on first use, so an offline fresh clone
    would otherwise fail here. The word-based estimate keeps the pipeline
    runnable; it changes chunk sizes slightly and nothing else.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s))
    except Exception:  # noqa: BLE001 - offline, missing package, download failure
        return lambda s: int(len(s.split()) * _TOKENS_PER_WORD)


def word_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of every whitespace-delimited token."""
    return [m.span() for m in _WORD.finditer(text)]


def fixed_size_chunks(
    text: str,
    target_tokens: int = 512,
    overlap_tokens: int = 0,
    count_tokens: Callable[[str], int] | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield (char_start, char_end) spans of roughly target_tokens each.

    Boundaries land on word edges, so no chunk begins or ends mid-word.
    """
    count_tokens = count_tokens or get_token_counter()
    spans = word_spans(text)
    if not spans:
        return

    costs = [max(count_tokens(text[a:b]), 1) for a, b in spans]

    i = 0
    n = len(spans)
    while i < n:
        total = 0
        j = i
        while j < n and total < target_tokens:
            total += costs[j]
            j += 1

        yield spans[i][0], spans[j - 1][1]

        if j >= n:
            break

        if overlap_tokens > 0:
            back = 0
            k = j
            while k > i + 1 and back < overlap_tokens:
                k -= 1
                back += costs[k]
            i = k
        else:
            i = j


def chunk_document(
    doc: dict,
    strategy: str = "fixed_512",
    target_tokens: int = 512,
    overlap_tokens: int = 0,
    count_tokens: Callable[[str], int] | None = None,
) -> list[Chunk]:
    """Chunk one extracted document, attaching provenance to every piece.

    `doc` is one of the JSON files written by scripts/02_extract_text.py.
    """
    text = doc["text"]
    count_tokens = count_tokens or get_token_counter()

    # Map char offset -> owning section heading, for metadata only.
    bounds: list[tuple[int, int, str]] = []
    cursor = 0
    for sec in doc.get("sections", []):
        found = text.find(sec["text"][:120], cursor) if sec["text"] else -1
        if found >= 0:
            bounds.append((found, found + len(sec["text"]), sec["heading"]))
            cursor = found
    section_at = _section_lookup(bounds)

    return [
        Chunk(
            arxiv_id=doc["arxiv_id"],
            text=text[start:end],
            char_start=start,
            char_end=end,
            section=section_at(start),
            strategy=strategy,
            title=doc.get("title", ""),
            published=doc.get("published", ""),
        )
        for start, end in fixed_size_chunks(text, target_tokens, overlap_tokens, count_tokens)
    ]


def _section_lookup(bounds: list[tuple[int, int, str]]) -> Callable[[int], str]:
    def lookup(pos: int) -> str:
        for start, end, heading in bounds:
            if start <= pos < end:
                return heading
        return "(unknown)"

    return lookup