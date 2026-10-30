"""Paired significance tests between ablation runs, at span level.

The stability analysis (15) answers "would the same arm be selected on
other questions?". This answers the other half of the switching rule: "is
the difference larger than sampling noise?". Both must pass before an arm
replaces the baseline.

Two instruments, because they fail differently:

  McNemar exact   Every span is scored by BOTH runs on the SAME question,
                  so the runs are paired and an unpaired test would throw
                  away that structure and lose power. McNemar looks only at
                  the DISCORDANT spans - found by one run, missed by the
                  other - because spans both runs found, or both missed,
                  carry no information about which is better. Exact
                  binomial rather than the chi-square approximation, which
                  is unreliable below ~25 discordant pairs and this gold
                  set produces fewer than that regularly.

  Cluster         The CI on the difference in span recall, resampling
  bootstrap       QUESTIONS rather than spans. Spans within a question are
                  correlated - a two-span question usually has both spans
                  in the same paper, often in adjacent chunks - so
                  resampling spans independently would treat 101 correlated
                  observations as 101 independent ones and report an
                  interval far too narrow.

  Holm            With three arms there are three pairwise tests, and Phase
  correction      6 plans about seventeen runs. Testing many pairs at
                  alpha=0.05 makes a spurious "significant" result likely
                  by construction. Holm-Bonferroni controls the
                  family-wise error rate without assuming independence, and
                  is uniformly more powerful than plain Bonferroni.

No scipy: the exact binomial is a few lines of `math.comb`, and adding a
dependency for it would be worse than writing it.

Usage:
    python scripts/16_paired_test.py baseline bm25 hybrid
    python scripts/16_paired_test.py baseline bm25 --k 10

Writes one row per pair to results/tables/paired_<tags>_<metric>.csv,
e.g. paired_baseline_bm25_B20000.csv or _k10 (override with --out).
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from rageval.config import TABLES_DIR
from rageval.evaluation.spans import span_indices


def binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial p for b successes in b+c trials at p=0.5.

    This is McNemar's exact test. With no discordant pairs the runs are
    indistinguishable on every span and p is 1 by definition.
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


def load_run(tag: str, col: str, tables_dir: Path):
    path = tables_dir / f"retrieval_{tag}.csv"
    if not path.exists():
        print(f"  missing: {path}", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    if col not in df.columns:
        print(f"  {tag}: no column {col}", file=sys.stderr)
        return None
    scored = df[df["scorable"] == True]  # noqa: E712 - pandas mask
    return {
        str(r["id"]): (int(r["n_evidence"]), span_indices(r[col]))
        for _, r in scored.iterrows()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tags", nargs="+", help="Run tags to compare (>=2).")
    parser.add_argument("--budget", type=int, default=20_000)
    parser.add_argument("--k", type=int, default=None,
                        help="Test at fixed k instead of a character budget.")
    parser.add_argument("--boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--tables-dir", default=None)
    parser.add_argument("--out", default=None,
                        help="CSV path (default results/tables/paired_<tags>_<metric>.csv).")
    args = parser.parse_args()

    if len(args.tags) < 2:
        print("Need at least two tags.", file=sys.stderr)
        return 1

    tables_dir = Path(args.tables_dir) if args.tables_dir else TABLES_DIR
    if args.k is not None:
        col, label = f"hit_spans@{args.k}", f"k={args.k}"
    else:
        col, label = f"hit_spans@B{args.budget}", f"budget {args.budget:,}"
    # Filename suffix, following the oracle_*_k10 / _k50 convention, so a
    # run at another budget or k does not overwrite this one.
    metric = f"k{args.k}" if args.k is not None else f"B{args.budget}"

    runs = {}
    for tag in args.tags:
        r = load_run(tag, col, tables_dir)
        if r is not None:
            runs[tag] = r
    if len(runs) < 2:
        print("Fewer than two runs loaded.", file=sys.stderr)
        return 1

    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    if not ids:
        print("No questions in common.", file=sys.stderr)
        return 1

    # n_evidence belongs to the gold set, not to a run. Disagreement means
    # the runs were scored against different gold files and every number
    # below would be meaningless.
    tags = list(runs)
    n_ev = {q: runs[tags[0]][q][0] for q in ids}
    for tag in tags:
        bad = [q for q in ids if runs[tag][q][0] != n_ev[q]]
        if bad:
            print(f"  ERROR: {tag} disagrees on n_evidence for {bad}. "
                  f"Runs were scored against different gold sets.", file=sys.stderr)
            return 1

    # Span-level hit vectors, grouped by question so the bootstrap can
    # resample whole questions.
    per_q = {tag: [np.array([i in runs[tag][q][1] for i in range(n_ev[q])])
                   for q in ids] for tag in tags}
    sizes = np.array([n_ev[q] for q in ids])
    total_spans = int(sizes.sum())

    def recall(tag: str, idx: np.ndarray) -> float:
        hit = sum(int(per_q[tag][i].sum()) for i in idx)
        tot = int(sizes[idx].sum())
        return hit / tot if tot else float("nan")

    all_idx = np.arange(len(ids))

    print(f"\n{'=' * 84}")
    print(f"PAIRED TESTS   span level, {label}")
    print(f"{'=' * 84}")
    print(f"  questions {len(ids)}   spans {total_spans}   "
          f"bootstrap {args.boot:,} resamples of questions   seed {args.seed}\n")
    print(f"  {'run':<14}{'span recall':>13}")
    for tag in tags:
        print(f"  {tag:<14}{recall(tag, all_idx):>13.3f}")

    rng = np.random.default_rng(args.seed)
    boot_idx = rng.integers(0, len(ids), size=(args.boot, len(ids)))

    results = []
    for a, b in combinations(tags, 2):
        hits_a = np.concatenate(per_q[a])
        hits_b = np.concatenate(per_q[b])
        n_b = int(np.sum(hits_a & ~hits_b))   # a found, b missed
        n_c = int(np.sum(~hits_a & hits_b))   # b found, a missed
        p = binom_two_sided(n_b, n_c)

        diff = np.empty(args.boot)
        for j in range(args.boot):
            idx = boot_idx[j]
            diff[j] = recall(b, idx) - recall(a, idx)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        results.append({
            "a": a, "b": b, "delta": recall(b, all_idx) - recall(a, all_idx),
            "lo": lo, "hi": hi, "b_only": n_b, "c_only": n_c,
            "discordant": n_b + n_c, "p": p,
        })

    for r, adj in zip(results, holm([r["p"] for r in results])):
        r["p_holm"] = adj

    print(f"\n{'-' * 84}")
    print("PAIRWISE   delta is the second run minus the first")
    print(f"{'-' * 84}")
    print(f"  {'comparison':<24}{'delta':>9}{'95% CI':>20}{'disc':>7}"
          f"{'split':>10}{'p':>9}{'p Holm':>9}")
    for r in results:
        ci = f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"
        split = f"{r['b_only']}/{r['c_only']}"
        print(f"  {r['a'] + ' vs ' + r['b']:<24}{r['delta']:>+9.3f}{ci:>20}"
              f"{r['discordant']:>7}{split:>10}{r['p']:>9.4f}{r['p_holm']:>9.4f}")
    print("\n  'split' is spans the FIRST run found and the second missed, over")
    print("  the reverse. Concordant spans are excluded by design: they carry")
    print("  no information about which run is better.")

    print(f"\n{'-' * 84}")
    print("READING")
    print(f"{'-' * 84}")
    for r in results:
        excludes = (r["lo"] > 0) or (r["hi"] < 0)
        sig = r["p_holm"] < 0.05
        if excludes and sig:
            verdict = (f"{r['b']} differs from {r['a']}: CI excludes zero and "
                       f"Holm p = {r['p_holm']:.4f}.")
        elif excludes != sig:
            verdict = (f"{r['a']} vs {r['b']}: the two instruments disagree "
                       f"(CI {'excludes' if excludes else 'includes'} zero, "
                       f"Holm p = {r['p_holm']:.4f}). Treat as undecided.")
        else:
            verdict = (f"{r['a']} vs {r['b']}: not separable at this sample "
                       f"size ({r['discordant']} discordant spans).")
        print(f"  {verdict}")
    print("\n  This is only half the switching rule. An arm replaces the baseline")
    print("  when its CI excludes zero AND its half-split win rate from")
    print("  15_selection_stability.py is decisively above chance.")
    print("  Holm-adjusted p is the one to read: these pairs were all tested")
    print("  together, and an uncorrected p invites the multiplicity it ignores.\n")

    # --- the table the build log cites ---------------------------------------
    rows = [{
        "a": r["a"],
        "b": r["b"],
        "metric": label,
        "recall_a": recall(r["a"], all_idx),
        "recall_b": recall(r["b"], all_idx),
        "delta": r["delta"],
        "ci_lo": r["lo"],
        "ci_hi": r["hi"],
        "a_only": r["b_only"],      # first run found, second missed
        "b_only": r["c_only"],      # second run found, first missed
        "discordant": r["discordant"],
        "p": r["p"],
        "p_holm": r["p_holm"],
        "n_questions": len(ids),
        "n_spans": total_spans,
        "boot": args.boot,
        "seed": args.seed,
    } for r in results]
    out_path = Path(args.out) if args.out else (
        TABLES_DIR / f"paired_{'_'.join(tags)}_{metric}.csv")
    # Unlink first: a failed write must not leave the previous run's table
    # in place reading as this run's result.
    out_path.unlink(missing_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  -> {out_path}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())