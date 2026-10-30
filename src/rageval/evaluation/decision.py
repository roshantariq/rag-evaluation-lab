"""The switching rule's decision table, shared by every sweep that applies it.

    replaces       paired CI excludes zero, positive, AND win rate >= threshold
    leaning        CI excludes zero, positive, win rate below it
    not_separable  CI includes zero ("not separable" is not "equal")
    worse          CI excludes zero, negative

"Excludes zero" is strict: a bound of exactly 0.000 does not exclude it.
"not_separable" rather than "null", because pandas reads the string "null"
in a CSV back as a missing value (bug museum, sweep 3).

The threshold is passed in, not defaulted, so each protocol script states
the value it pre-registered.
"""

from __future__ import annotations


def classify(lo: float, hi: float, win_rate: float, threshold: float) -> str:
    if hi < 0:
        return "worse"
    if lo > 0:
        return "replaces" if win_rate >= threshold else "leaning"
    return "not_separable"


def decide(outcomes: dict[str, str], control: str) -> str:
    """The one arm that replaces the control, or the control if none does.

    Win rates over one set of half-split selections sum to at most 1, so
    at most one arm can clear a threshold above 0.5. Two winners means the
    tables are inconsistent, and that is an error, not a tie to break.
    """
    winners = [a for a, o in outcomes.items() if o == "replaces"]
    if len(winners) > 1:
        raise ValueError(f"more than one arm clears the win-rate threshold: {winners}")
    return winners[0] if winners else control
