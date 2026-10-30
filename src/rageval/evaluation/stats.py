"""Exact McNemar and Holm correction, shared by every script that tests pairs.

Hoisted from scripts/16_paired_test.py when Phase 7 needed the same tests,
so the two cannot drift apart. No scipy: the exact binomial is a few lines
of `math.comb`.
"""

from __future__ import annotations

from math import comb


def binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial p for b successes in b+c trials at p=0.5.

    This is McNemar's exact test. With no discordant pairs the runs are
    indistinguishable and p is 1 by definition.
    """
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)          # enforce monotonicity
        adjusted[i] = min(1.0, running)
    return adjusted
