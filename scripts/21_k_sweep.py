"""Apply sweep 4's pre-registered rule: how many chunks Phase 7 is fed.

Implements docs/k-sweep-protocol.md and nothing else. Written and tested
before it was run on the real tables, like 19 and 20. The constants that
decide the outcome are module constants, not flags.

WHAT IT READS

    retrieval_bm25.csv         the baseline; decides
    retrieval_rerank.csv       exploratory since sweep 3; reported only
    retrieval_para_bm25.csv    the paraphrased gold set; reported only
    paraphrase_overlap.csv     which 20 questions were rewritten

No new retrieval run is needed: the harness scores every k on the ladder
from one ranked list, so the k values are nested prefixes of that list.
The script checks this, because the rule relies on it: a span found at
k=5 must also be found at k=10.

WHAT IT DECIDES

  For each step 1->3, 3->5, 5->10, 10->20 under BM25, the gain in span
  recall with a cluster-bootstrap CI over questions. Steps are tested in
  order and testing stops at the first step whose CI does not exclude zero
  (fixed-sequence testing, so no Holm adjustment). The chosen k is the
  upper end of the last separable step. If every step is separable, the
  chosen k is 20, the top of the ladder, and that is reported as gains not
  exhausted rather than a plateau.

  Gains are nested, so a question's gain is never negative and the CI's
  lower bound is either exactly 0 or above it. "Excludes zero" means
  strictly above.

WHAT IT REPORTS AND DOES NOT DECIDE

  * The CIs of steps after the stopping point.
  * Two-span completeness, complete coverage and per-type recall at each k.
  * The same curve for the 20 rewritten questions, original and reworded.
  * Reranked BM25's curve, and the smallest k at which it reaches BM25's
    recall at the chosen k.

Writes results/tables/k_sweep.csv and results/tables/k_sweep_curve.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rageval.config import TABLES_DIR
from rageval.evaluation.spans import span_indices
from rageval.provenance import record

# --- pre-registered in docs/k-sweep-protocol.md; do not change -------------
KS = (1, 3, 5, 10, 20)
BOOT = 10_000
SEED = 20260827
DECIDING_ARM = "bm25"
TYPES = ("factual", "comparative", "multi_hop", "ambiguous")


def load_run(tag: str, tables_dir: Path) -> pd.DataFrame | None:
    """Scorable rows with one hit count per k, after checking nesting."""
    path = tables_dir / f"retrieval_{tag}.csv"
    if not path.exists():
        print(f"  missing: {path}", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    cols = [f"hit_spans@{k}" for k in KS]
    absent = [c for c in cols if c not in df.columns]
    if absent:
        print(f"  {tag}: no column(s) {absent}", file=sys.stderr)
        return None
    scored = df[df["scorable"] == True].copy()  # noqa: E712 - pandas mask
    scored.index = scored["id"].astype(str)
    found = {k: [span_indices(c) for c in scored[f"hit_spans@{k}"]] for k in KS}
    for small, large in zip(KS, KS[1:]):
        broken = [q for q, a, b in zip(scored.index, found[small], found[large])
                  if not a <= b]
        if broken:
            raise ValueError(f"{tag}: spans found at k={small} but not at k={large} "
                             f"for {broken[:5]}: the k columns are not nested")
    for k in KS:
        scored[f"h{k}"] = [len(s) for s in found[k]]
    scored["n_ev"] = scored["n_evidence"].astype(int)
    return scored


def step_table(run: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Each step's gain and CI, with the fixed-sequence decision applied."""
    n_ev = run["n_ev"].to_numpy()
    total = n_ev.sum()
    draws = rng.integers(0, len(run), size=(BOOT, len(run)))
    rows, still_testing = [], True
    for small, large in zip(KS, KS[1:]):
        d = (run[f"h{large}"] - run[f"h{small}"]).to_numpy()
        boot = d[draws].sum(axis=1) / n_ev[draws].sum(axis=1)
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        separable = lo > 0
        rows.append({
            "step": f"{small}->{large}", "k_from": small, "k_to": large,
            "recall_from": run[f"h{small}"].sum() / total,
            "recall_to": run[f"h{large}"].sum() / total,
            "gain": d.sum() / total, "ci_lo": lo, "ci_hi": hi,
            "spans_gained": int(d.sum()), "questions_gaining": int((d > 0).sum()),
            "separable": separable, "tested": still_testing,
        })
        still_testing = still_testing and separable
    return pd.DataFrame(rows)


def choose_k(steps: pd.DataFrame) -> int:
    """Upper end of the last separable step, stopping at the first failure."""
    chosen = KS[0]
    for _, s in steps.iterrows():
        if not s["separable"]:
            break
        chosen = int(s["k_to"])
    return chosen


def curve(run: pd.DataFrame, arm: str, subset: str) -> list[dict]:
    out = []
    n_ev = run["n_ev"]
    two = run[n_ev == 2]
    for k in KS:
        h = run[f"h{k}"]
        row = {"arm": arm, "subset": subset, "k": k, "n_questions": len(run),
               "n_spans": int(n_ev.sum()),
               "span_recall": h.sum() / n_ev.sum(),
               "complete_share": float((h == n_ev).mean()),
               "two_span_n": len(two),
               "two_span_both": int((two[f"h{k}"] == 2).sum())}
        for t in TYPES:
            sub = run[run["question_type"] == t]
            row[f"recall_{t}"] = (sub[f"h{k}"].sum() / sub["n_ev"].sum()
                                  if len(sub) else np.nan)
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", default=None, help="Override results/tables (for testing).")
    args = parser.parse_args()

    tables = Path(args.tables_dir) if args.tables_dir else TABLES_DIR
    out_path = tables / "k_sweep.csv"
    curve_path = tables / "k_sweep_curve.csv"
    out_path.unlink(missing_ok=True)
    curve_path.unlink(missing_ok=True)

    try:
        base = load_run(DECIDING_ARM, tables)
        if base is None:
            return 1
        rerank = load_run("rerank", tables) if (tables / "retrieval_rerank.csv").exists() else None
        para = (load_run("para_bm25", tables)
                if (tables / "retrieval_para_bm25.csv").exists() else None)
    except ValueError as e:
        print(f"  {e}", file=sys.stderr)
        return 1

    rng = np.random.default_rng(SEED)
    steps = step_table(base, rng)
    chosen = choose_k(steps)
    ceiling = chosen == KS[-1]

    rows = curve(base, DECIDING_ARM, "all")
    reach = None
    if rerank is not None:
        rows += curve(rerank, "rerank", "all")
        target = base[f"h{chosen}"].sum() / base["n_ev"].sum()
        hits = [k for k in KS if rerank[f"h{k}"].sum() / rerank["n_ev"].sum() >= target]
        reach = hits[0] if hits else None
    overlap_path = tables / "paraphrase_overlap.csv"
    if para is not None and overlap_path.exists():
        rewritten = sorted(set(pd.read_csv(overlap_path)["id"].astype(str)) & set(para.index))
        rows += curve(base.loc[rewritten], DECIDING_ARM, "rewritten_20_original")
        rows += curve(para.loc[rewritten], DECIDING_ARM, "rewritten_20_reworded")
    curves = pd.DataFrame(rows)

    # --- report -------------------------------------------------------------
    print(f"\n{'=' * 84}\nSWEEP 4 - VALUE OF k   pre-registered in docs/k-sweep-protocol.md\n"
          f"{'=' * 84}")
    print(f"  {DECIDING_ARM}: {len(base)} questions, {int(base['n_ev'].sum())} spans; "
          f"k columns checked nested\n")
    print(f"  {'step':<8} {'recall':>15} {'gain':>8} {'95% CI':>18} {'spans':>6} "
          f"{'questions':>10}   decision")
    for _, s in steps.iterrows():
        verdict = ("separable" if s["separable"] else "NOT separable") if s["tested"] \
            else "not tested (reported only)"
        print(f"  {s['step']:<8} {s['recall_from']:.3f} -> {s['recall_to']:.3f} "
              f"{s['gain']:+8.3f}   [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}] "
              f"{s['spans_gained']:>6} {s['questions_gaining']:>10}   {verdict}")
    print(f"\n  CHOSEN k = {chosen}")
    if ceiling:
        print("  Every step was separable: k=20 is the top of the ladder, not a plateau.\n"
              "  Retrieval gains are not exhausted at the largest k Phase 7 can afford.")

    print(f"\n{'-' * 84}\nREPORTED ONLY\n{'-' * 84}")
    show = curves[["arm", "subset", "k", "span_recall", "complete_share",
                   "two_span_both", "two_span_n"]]
    for (arm, subset), g in show.groupby(["arm", "subset"], sort=False):
        print(f"\n  {arm} / {subset}")
        for _, r in g.iterrows():
            print(f"    k={r['k']:<3} recall {r['span_recall']:.3f}   complete "
                  f"{r['complete_share']:.3f}   two-span both {r['two_span_both']}/"
                  f"{r['two_span_n']}")
    if rerank is not None:
        print(f"\n  rerank reaches bm25's recall at k={chosen} by k="
              f"{reach if reach is not None else 'never on the ladder'}")

    steps.assign(chosen_k=chosen, ladder_ceiling=ceiling,
                 rerank_reaches_at_k=reach, boot=BOOT, seed=SEED).to_csv(out_path, index=False)
    curves.to_csv(curve_path, index=False)
    record([out_path, curve_path])
    print(f"\n  -> {out_path}\n  -> {curve_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
