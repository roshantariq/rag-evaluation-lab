"""Ask whether the winner of an ablation would still win on other questions.

A paired test answers "is A better than B on these 58 questions?".  This
answers a different and, for a sweep, more urgent question:

    if the gold set had been a different 58 questions, would the same
    configuration have been selected?

Phase 6 scores ~17 configurations on ONE question set.  Whichever comes out
top is top partly because it is better and partly because it suited these
particular questions - and those questions are in the set every single time.
Reporting the winner's score then reports its luck along with its merit.
That is the winner's curse.  It has nothing to do with training (nothing here
is trained); it is purely an artifact of selecting on the same data you
report on, and it grows with the number of configurations compared.

Procedure - repeated stratified half-splits:

  1. split the scorable questions in half, stratified by question_type so
     each half keeps the 20/14/18/6 mix.  Unstratified, a split could land
     15 of the 18 multi-hop questions on one side and the comparison would
     be measuring question mix rather than configuration.
  2. on half A, select the winner by span recall - the same rule used for
     real decisions.
  3. on half B - questions that had no say in the choice - record whether
     that configuration still wins, and what it scores.
  4. repeat, both directions per split.

Two numbers come out:

  WIN FREQUENCY  how often each configuration is selected.  Near 50/50 for
                 two configurations means the "winner" of the sweep is a
                 coin flip and should be written up as one.
  OPTIMISM       mean (selection-half score - held-out score) for whichever
                 configuration was selected.  A direct estimate of how many
                 recall points the reported winner is inflated by.

Why not a permanent held-out set: at n=58 a 30% holdout is 17 questions,
whose CI on recall is roughly +/-0.12 - too wide to separate any
configuration from any other.  That trades a third of the gold set for a
number that says nothing.  Repeated splitting uses every question in both
roles across thousands of draws instead.

Scoring is span-level (spans found / spans available) rather than
question-level, because that is the unit that compares cleanly against the
oracle ceiling and does not reward multi-evidence questions for having two
chances to register a hit.

Usage:
    python scripts/15_selection_stability.py baseline chunk256 chunk1024
    python scripts/15_selection_stability.py baseline chunk256 --budget 10000
    python scripts/15_selection_stability.py baseline chunk256 --k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rageval.config import TABLES_DIR


def _count_spans(cell) -> int:
    """Count span indices in a hit_spans cell, surviving pandas dtype guessing.

    A column holding only "0" and blanks is inferred as float64, so "0" comes
    back as 0.0.  Counting with a bare isdigit() then silently returns zero
    for every row.  This mirrors the guard in 13_compare_runs.py; if a third
    script needs it, hoist it into rageval rather than copying again.
    """
    if cell is None:
        return 0
    try:
        if pd.isna(cell):
            return 0
    except (TypeError, ValueError):
        pass
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return 0
    n = 0
    for part in s.split(";"):
        part = part.strip()
        if part.endswith(".0"):
            part = part[:-2]
        if part.isdigit():
            n += 1
    return n


def load_run(tag: str, col: str, tables_dir: Path) -> pd.DataFrame | None:
    """Per-question hit counts for one run, indexed by question id."""
    path = tables_dir / f"retrieval_{tag}.csv"
    if not path.exists():
        print(f"  missing: {path}", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    if col not in df.columns:
        print(f"  {tag}: no column {col}", file=sys.stderr)
        return None
    scored = df[df["scorable"] == True].copy()  # noqa: E712 - pandas mask
    out = pd.DataFrame(
        {
            "question_type": scored["question_type"].to_numpy(),
            "n_evidence": scored["n_evidence"].astype(int).to_numpy(),
            "hits": [_count_spans(c) for c in scored[col]],
        },
        index=pd.Index(scored["id"].to_numpy(), name="id"),
    )
    if (out["hits"] > out["n_evidence"]).any():
        bad = out[out["hits"] > out["n_evidence"]].index.tolist()
        print(f"  {tag}: more hits than spans for {bad} - check the harness",
              file=sys.stderr)
    return out


def stratified_half(types: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask selecting half of each question type.

    An odd-sized stratum is cut randomly high or low, so the extra question
    does not always land on the same side.
    """
    mask = np.zeros(len(types), dtype=bool)
    for t in np.unique(types):
        idx = np.flatnonzero(types == t)
        rng.shuffle(idx)
        cut = len(idx) // 2
        if len(idx) % 2 and rng.random() < 0.5:
            cut += 1
        mask[idx[:cut]] = True
    return mask


def span_recall(hits: np.ndarray, n_ev: np.ndarray, mask: np.ndarray) -> float:
    total = n_ev[mask].sum()
    return float(hits[mask].sum() / total) if total else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tags", nargs="+", help="Run tags to compare (>=2).")
    parser.add_argument("--budget", type=int, default=20_000,
                        help="Character budget to select on (default 20000).")
    parser.add_argument("--k", type=int, default=None,
                        help="Select at fixed k instead of a character budget.")
    parser.add_argument("--splits", type=int, default=2000,
                        help="Number of half-splits; each contributes two "
                             "selections (A picks, B judges, then swapped).")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--tables-dir", default=None,
                        help="Override results/tables (for testing).")
    args = parser.parse_args()

    if len(args.tags) < 2:
        print("Need at least two tags to have anything to select between.",
              file=sys.stderr)
        return 1

    tables_dir = Path(args.tables_dir) if args.tables_dir else TABLES_DIR
    if args.k is not None:
        col, label = f"hit_spans@{args.k}", f"k={args.k}"
    else:
        col, label = f"hit_spans@B{args.budget}", f"budget {args.budget:,}"

    runs: dict[str, pd.DataFrame] = {}
    for tag in args.tags:
        df = load_run(tag, col, tables_dir)
        if df is not None:
            runs[tag] = df
    if len(runs) < 2:
        print("Fewer than two runs loaded; nothing to select between.",
              file=sys.stderr)
        return 1

    # --- align: only questions present in every run can be compared --------
    common = set.intersection(*(set(df.index) for df in runs.values()))
    for tag, df in runs.items():
        extra = set(df.index) - common
        if extra:
            print(f"  note: {tag} has {len(extra)} question(s) no other run "
                  f"scored; dropped ({', '.join(sorted(extra))})",
                  file=sys.stderr)
    ids = sorted(common)
    if not ids:
        print("No questions in common.", file=sys.stderr)
        return 1

    ref = runs[args.tags[0]].loc[ids]
    types = ref["question_type"].to_numpy()
    n_ev = ref["n_evidence"].to_numpy()

    # n_evidence is a property of the gold set, not of the run.  If two runs
    # disagree, one of them was scored against a different gold file and the
    # whole comparison is void.
    for tag, df in runs.items():
        other = df.loc[ids, "n_evidence"].to_numpy()
        if not np.array_equal(other, n_ev):
            where = [ids[i] for i in np.flatnonzero(other != n_ev)]
            print(f"  ERROR: {tag} disagrees on n_evidence for {where}. "
                  f"Runs were scored against different gold sets.",
                  file=sys.stderr)
            return 1

    tags = list(runs)
    hits = {t: runs[t].loc[ids, "hits"].to_numpy() for t in tags}
    all_mask = np.ones(len(ids), dtype=bool)
    full = {t: span_recall(hits[t], n_ev, all_mask) for t in tags}

    print(f"\n{'=' * 82}")
    print(f"SELECTION STABILITY   selecting on span recall at {label}")
    print(f"{'=' * 82}")
    counts = pd.Series(types).value_counts().to_dict()
    mix = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    print(f"  questions {len(ids)} ({mix})")
    print(f"  spans     {int(n_ev.sum())}")
    print(f"  splits    {args.splits} x 2 directions = {2 * args.splits} selections")
    print(f"  seed      {args.seed}\n")

    print(f"  {'run':<14}{'full-set score':>16}")
    for t in tags:
        print(f"  {t:<14}{full[t]:>16.3f}")

    # --- the resampling ----------------------------------------------------
    rng = np.random.default_rng(args.seed)
    wins = {t: 0 for t in tags}
    sel_scores = {t: [] for t in tags}   # score on the half that selected it
    out_scores = {t: [] for t in tags}   # score on the fresh half
    optimism: list[float] = []
    margins: list[float] = []
    reselected = 0
    ties = 0
    n_sel = 0

    for _ in range(args.splits):
        a = stratified_half(types, rng)
        for sel, hold in ((a, ~a), (~a, a)):
            s = np.array([span_recall(hits[t], n_ev, sel) for t in tags])
            h = np.array([span_recall(hits[t], n_ev, hold) for t in tags])
            if np.isnan(s).any() or np.isnan(h).any():
                continue  # a half with no spans at all; cannot select
            best = np.flatnonzero(s == s.max())
            if len(best) > 1:
                ties += 1
            # Random tie-break: taking the first would quietly hand every tie
            # to whichever tag was typed first on the command line.
            w = int(rng.choice(best))
            tag = tags[w]

            wins[tag] += 1
            n_sel += 1
            sel_scores[tag].append(s[w])
            out_scores[tag].append(h[w])
            optimism.append(s[w] - h[w])
            ordered = np.sort(s)[::-1]
            margins.append(float(ordered[0] - ordered[1]))
            if h[w] == h.max():
                reselected += 1

    if not n_sel:
        print("\nNo usable splits.", file=sys.stderr)
        return 1

    # --- results -----------------------------------------------------------
    print(f"\n{'-' * 82}")
    print("WIN FREQUENCY   how often each run is selected on half the questions")
    print(f"{'-' * 82}")
    print(f"  {'run':<14}{'wins':>8}{'win rate':>11}{'selection half':>17}"
          f"{'held-out half':>16}{'optimism':>11}")
    for t in sorted(tags, key=lambda x: -wins[x]):
        rate = wins[t] / n_sel
        if wins[t]:
            ms, mo = float(np.mean(sel_scores[t])), float(np.mean(out_scores[t]))
            print(f"  {t:<14}{wins[t]:>8}{rate:>10.1%}{ms:>17.3f}{mo:>16.3f}"
                  f"{ms - mo:>+11.3f}")
        else:
            print(f"  {t:<14}{0:>8}{0.0:>10.1%}{'-':>17}{'-':>16}{'-':>11}")
    print("\n  'selection half' and 'held-out half' are conditional on being")
    print("  selected, so they are not comparable across rows - read them")
    print("  against each other within a row.")

    top = max(tags, key=lambda t: wins[t])
    top_rate = wins[top] / n_sel

    print(f"\n{'-' * 82}")
    print("THE PRICE OF SELECTING AND REPORTING ON THE SAME QUESTIONS")
    print(f"{'-' * 82}")
    opt = np.array(optimism)
    lo, hi = np.percentile(opt, [2.5, 97.5])
    print(f"  mean optimism of the winner     {opt.mean():+.3f}  "
          f"(95% of splits {lo:+.3f} to {hi:+.3f})")
    print(f"  winner also wins held-out half  {reselected / n_sel:.1%}")
    print(f"  mean winning margin on a half   {float(np.mean(margins)):.3f}")
    if ties:
        print(f"  exact ties (broken at random)   {ties} of {n_sel}")
    print("\n  Optimism is the average gap between the winner's score on the")
    print("  questions that chose it and its score on questions that did not.")
    print("  Subtract it from any 'best configuration' figure before reporting.")

    print(f"\n{'-' * 82}")
    print("READING")
    print(f"{'-' * 82}")
    if top_rate < 0.60:
        verdict = (f"COIN FLIP. '{top}' is selected {top_rate:.0%} of the time - "
                   f"barely above chance for\n  {len(tags)} configurations. Carry the "
                   f"baseline forward and record the sweep as a null\n  result. Whatever "
                   f"led on the full set led by luck of the question draw.")
    elif top_rate < 0.90:
        verdict = (f"LEANING, NOT DECISIVE. '{top}' wins {top_rate:.0%} of splits. "
                   f"Worth noting as a\n  direction, but not enough to change the "
                   f"baseline on its own - pair it with a\n  paired CI that excludes "
                   f"zero before switching.")
    else:
        verdict = (f"ROBUST. '{top}' wins {top_rate:.0%} of splits, so the choice does "
                   f"not depend on\n  which questions happened to be in the gold set.")
    print(f"  {verdict}")
    print("\n  Rule for the rest of Phase 6: claim a winner only when the paired CI")
    print("  excludes zero AND win frequency is decisively above chance. Otherwise")
    print("  carry the baseline forward rather than chasing the highest number.")
    print("\n  Half-splits have less power than the full set by construction - that")
    print("  is the point, not a defect. A configuration that cannot survive being")
    print("  chosen on 29 questions was never separable on 58.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())