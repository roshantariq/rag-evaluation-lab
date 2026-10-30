"""Apply the chunker sweep's pre-registered rule.

Implements docs/chunker-protocol.md and nothing else. Written and tested
before any chunker run exists, like 19-22. The constants that decide the
outcome are module constants, not flags.

WHAT IT READS

    paired_bm25_chunk_section_chunk_recursive_chunk_semantic_B20000.csv     from 16
    stability_bm25_chunk_section_chunk_recursive_chunk_semantic_B20000.csv  from 15
    chunks_section.csv, chunks_recursive.csv, chunks_semantic.csv            from 23

WHAT IT DECIDES

  Each chunker against fixed_512 (the `bm25` run), span recall at the 20k
  budget, by the shared decision table (rageval.evaluation.decision):
  replaces / leaning / not_separable / worse, win-rate threshold 90%. The
  win rate is each arm's share of half-split selections among all four
  runs. If one chunker replaces, it becomes the baseline chunking;
  otherwise fixed_512 stays.

WHAT IT REPORTS AND DOES NOT DECIDE

  * Holm p across the six pairs 16 tests.
  * Each chunk set's size: chunk count and median characters, so a reader
    can see the size match the protocol required.

Writes results/tables/chunker_result.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from rageval.config import TABLES_DIR
from rageval.evaluation import decision
from rageval.provenance import record

# --- pre-registered in docs/chunker-protocol.md; do not change -------------
CONTROL = "bm25"
ARMS = ("chunk_section", "chunk_recursive", "chunk_semantic")
TARGETS = {"chunk_section": 680, "chunk_recursive": 650, "chunk_semantic": 520}
WIN_THRESHOLD = 0.90
LABEL = "budget 20,000"
PAIRED = "paired_bm25_chunk_section_chunk_recursive_chunk_semantic_B20000.csv"
STABILITY = "stability_bm25_chunk_section_chunk_recursive_chunk_semantic_B20000.csv"


def chunk_file(arm: str) -> str:
    return f"chunks_{arm.removeprefix('chunk_')}.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", default=None, help="Override results/tables (for testing).")
    args = parser.parse_args()

    tables = Path(args.tables_dir) if args.tables_dir else TABLES_DIR
    out_path = tables / "chunker_result.csv"
    out_path.unlink(missing_ok=True)

    needed = [PAIRED, STABILITY] + [chunk_file(a) for a in ARMS]
    missing = [n for n in needed if not (tables / n).exists()]
    if missing:
        print(f"  missing: {missing}. Run 23, 10, 16 and 15 first.", file=sys.stderr)
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
        outcomes[arm] = decision.classify(p["ci_lo"], p["ci_hi"], s["win_rate"], WIN_THRESHOLD)
        chunks = pd.read_csv(tables / chunk_file(arm))
        chars = chunks["char_end"] - chunks["char_start"]
        rows.append({
            "arm": arm, "target_tokens": TARGETS[arm], "metric": LABEL,
            "n_chunks": len(chunks), "median_chars": float(chars.median()),
            "recall_fixed512": p["recall_a"], "recall_arm": p["recall_b"],
            "delta": p["delta"], "ci_lo": p["ci_lo"], "ci_hi": p["ci_hi"],
            "p_holm": p["p_holm"], "win_rate": s["win_rate"], "outcome": outcomes[arm],
        })

    try:
        chosen = decision.decide(outcomes, CONTROL)
    except ValueError as e:
        print(f"  {e}", file=sys.stderr)
        return 1
    ctl = stab[stab["run"] == CONTROL]
    control_win = float(ctl["win_rate"].iloc[0]) if len(ctl) else float("nan")

    print(f"\n{'=' * 90}\nCHUNKER SWEEP   pre-registered in docs/chunker-protocol.md\n{'=' * 90}")
    print(f"  {'arm':<17}{'chunks':>7}{'median':>8}{'fixed':>7}{'arm':>7}{'delta':>8}"
          f"{'95% CI':>19}{'Holm p':>8}{'win':>7}   outcome")
    for r in rows:
        print(f"  {r['arm']:<17}{r['n_chunks']:>7}{r['median_chars']:>8.0f}"
              f"{r['recall_fixed512']:>7.3f}{r['recall_arm']:>7.3f}{r['delta']:>+8.3f}"
              f"   [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]{r['p_holm']:>8.4f}"
              f"{r['win_rate']:>7.1%}   {r['outcome']}")
    print(f"  fixed_512 (bm25) win rate {control_win:.1%}")
    if chosen == CONTROL:
        print("\n  NO CHUNKER REPLACES fixed_512. The baseline chunking stays.")
    else:
        print(f"\n  {chosen.upper()} REPLACES fixed_512 as the baseline chunking.")

    pd.DataFrame(rows).assign(chosen=chosen, control_win_rate=control_win) \
        .to_csv(out_path, index=False)
    record([out_path])
    print(f"\n  -> {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
