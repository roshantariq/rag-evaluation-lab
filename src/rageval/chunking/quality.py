"""Chunk prose ratio, and the filter the Phase 6 ablation tests.

Phase 2b found that low-prose chunks fall into four kinds a single threshold
cannot separate: results tables and equations (valuable), and chart axis
labels and table-of-contents leaders (noise). It removed only base64 and
deferred everything else to a measured ablation (docs/chunk-filter-
protocol.md) instead of a silent filter. This module is that filter.

`prose_ratio` is the score `scripts/diagnose_chunks.py` wrote to
`results/tables/chunk_quality.csv`, moved here unchanged so the diagnostic
and the harness cannot drift apart.
"""

from __future__ import annotations

import re
from typing import Sequence

WORDLIKE = re.compile(r"^[A-Za-z][A-Za-z'-]{2,}$")


def prose_ratio(text: str) -> float:
    """Share of tokens that are ordinary words, not numbers or symbols."""
    tokens = text.split()
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if WORDLIKE.match(t.strip(".,;:()[]%")))
    return hits / len(tokens)


def filter_records(records: Sequence, min_ratio: float) -> tuple[list, list]:
    """Split records into (kept, removed) at `min_ratio`.

    A chunk is removed when its prose ratio is strictly below the threshold,
    the same comparison diagnose_chunks.py used to report the counts.
    """
    kept, removed = [], []
    for r in records:
        (removed if prose_ratio(r.text) < min_ratio else kept).append(r)
    return kept, removed
