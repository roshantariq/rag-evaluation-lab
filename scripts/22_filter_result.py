"""Apply the chunk-filter ablation's pre-registered rule.

Implements docs/chunk-filter-protocol.md and nothing else. Written and
tested before any filtered run exists, like 19, 20 and 21. The constants
that decide the outcome are module constants, not flags.

WHAT IT READS

    paired_bm25_filter030_filter040_filter050_B20000.csv     from 16
    stability_bm25_filter030_filter040_filter050_B20000.csv  from 15
    retrieval_filter030.csv, _040, _050                      from 10

WHAT IT DECIDES

  Each filtered arm against unfiltered BM25, span recall at the 20k budget:
    replaces       CI excludes zero, positive, AND win rate >= 90%
    leaning        CI excludes zero, positive, win rate < 90%
    not_separable  CI includes zero (not "equal")
    worse          CI excludes zero, negative

  The win rate is each arm's share of the half-split selections among all
  four runs, so at most one arm can reach 90%. If one arm replaces, it
  becomes the baseline filter; otherwise BM25 stays unfiltered.

WHAT IT REPORTS AND DOES NOT DECIDE

  * Holm p across the six pairs 16 tests.
  * Slots the filter could free: unfiltered top-10 positions held by the
    chunks each threshold removes (the harness's low_prose_top10 column).

Writes results/tables/filter_result.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from rageval.config import TABLES_DIR
from rageval.evaluation import decision
from rageval.provenance import record

# --- pre-registered in docs/chunk-filter-protocol.md; do not change --------
CONTROL = "bm25"
ARMS = ("filter030", "filter040", "filter050")
THRESHOLDS = {"filter030": 0.3, "filter040": 0.4, "filter050": 0.5}
WIN_THRESHOLD = 0.90
LABEL = "budget 20,000"
PAIRED = "paired_bm25_filter030_filter040_filter050_B20000.csv"
STABILITY = "stability_bm25_filter030_filter040_filter050_B20000.csv"


def classify(lo: float, hi: float, win_rate: float) -> str:
    """The protocol's four-way table, identical to sweep 3's."""
    return decision.classify(lo, hi, win_rate, WIN_THRESHOLD)


def decide(outcomes: dict[str, str]) -> str:
    """The arm that replaces unfiltered BM25, or CONTROL if none does."""
    return decision.decide(outcomes, CONTROL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", default=None, help="Override results/tables (for testing).")
    args = parser.parse_args()

    tables = Path(args.tables_dir) if args.tables_dir else TABLES_DIR
    out_path = tables / "filter_result.csv"
    out_path.unlink(missing_ok=True)

    needed = [PAIRED, STABILITY] + [f"retrieval_{a}.csv" for a in ARMS]
    missing = [n for n in needed if not (tables / n).exists()]
    if missing:
        print(f"  missing: {missing}. Run 10, 16 and 15 first.", file=sys.stderr)
        return 1

    paired = pd.read_csv(tables / PAIRED)
    stab = pd.read_csv(tables / STABILITY)
    if set(paired["metric"]) != {LABEL} or set(stab["metric"]) != {LABEL}:
        print(f"  tables were not computed at the pre-registered {LABEL!r}.", file=sys.stderr)
        return 1

    rows, outcomes = [], {}
    for arm in ARMS:
        p = paired[(paired["a"] == CONTROL) & (paired["b"] == arm)]
        s = stab[stab["run"] == arm]
        if len(p) != 1 or len(s) != 1:
            print(f"  no single {CONTROL}-vs-{arm} row in the paired or stability table.",
                  file=sys.stderr)
            return 1
        p, s = p.iloc[0], s.iloc[0]
        outcomes[arm] = classify(p["ci_lo"], p["ci_hi"], s["win_rate"])
        run = pd.read_csv(tables / f"retrieval_{arm}.csv")
        scored = run[run["scorable"] == True]  # noqa: E712 - pandas mask
        slots = (int(scored["low_prose_top10"].sum())
                 if "low_prose_top10" in scored.columns else None)
        rows.append({
            "arm": arm, "threshold": THRESHOLDS[arm], "metric": LABEL,
            "recall_bm25": p["recall_a"], "recall_arm": p["recall_b"],
            "delta": p["delta"], "ci_lo": p["ci_lo"], "ci_hi": p["ci_hi"],
            "p_holm": p["p_holm"], "win_rate": s["win_rate"],
            "outcome": outcomes[arm], "freed_top10_slots": slots,
            "questions_with_freed_slots": (int((scored["low_prose_top10"] > 0).sum())
                                           if slots is not None else None),
        })

    try:
        chosen = decide(outcomes)
    except ValueError as e:
        print(f"  {e}", file=sys.stderr)
        return 1
    control_win = float(stab.loc[stab["run"] == CONTROL, "win_rate"].iloc[0]) \
        if (stab["run"] == CONTROL).any() else float("nan")

    print(f"\n{'=' * 84}\nCHUNK-QUALITY FILTER   pre-registered in docs/chunk-filter-protocol.md\n"
          f"{'=' * 84}")
    print(f"  {'arm':<11}{'bm25':>7}{'arm':>7}{'delta':>8}{'95% CI':>19}{'Holm p':>8}"
          f"{'win':>7}{'freed slots':>13}   outcome")
    for r in rows:
        freed = "-" if r["freed_top10_slots"] is None else \
            f"{r['freed_top10_slots']} in {r['questions_with_freed_slots']}q"
        print(f"  {r['arm']:<11}{r['recall_bm25']:>7.3f}{r['recall_arm']:>7.3f}"
              f"{r['delta']:>+8.3f}   [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
              f"{r['p_holm']:>8.4f}{r['win_rate']:>7.1%}{freed:>13}   {r['outcome']}")
    print(f"  unfiltered bm25 win rate {control_win:.1%}")
    if chosen == CONTROL:
        print("\n  NO FILTER REPLACES THE BASELINE. BM25 stays unfiltered.")
    else:
        print(f"\n  {chosen.upper()} REPLACES THE BASELINE: prose filter at "
              f"{THRESHOLDS[chosen]}.")

    pd.DataFrame(rows).assign(chosen=chosen, control_win_rate=control_win) \
        .to_csv(out_path, index=False)
    record([out_path])
    print(f"\n  -> {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
