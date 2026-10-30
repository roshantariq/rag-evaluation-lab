"""Read hit_spans cells back from a results CSV, surviving pandas dtype guessing.

The harness writes the gold span indices each question retrieved as a
string such as "0;1", or a blank when nothing was found. A column holding
only "0" and blanks is inferred as float64, so "0" comes back as 0.0, and a
bare isdigit() then silently counts nothing. That produced one wrong
budget table before it was caught (CLAUDE.md, bug museum).

Every script that reads hit_spans goes through these two functions. There
were six private copies before they were hoisted here; one fix landing in
five of them is the failure that hoisting prevents.
"""

from __future__ import annotations

import pandas as pd


def span_indices(cell) -> set[int]:
    """The gold span indices in one hit_spans cell."""
    if cell is None:
        return set()
    try:
        if pd.isna(cell):
            return set()
    except (TypeError, ValueError):
        pass
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return set()
    out = set()
    for part in s.split(";"):
        part = part.strip()
        if part.endswith(".0"):
            part = part[:-2]
        if part.isdigit():
            out.add(int(part))
    return out


def count_spans(cell) -> int:
    """How many gold spans one hit_spans cell records."""
    return len(span_indices(cell))
