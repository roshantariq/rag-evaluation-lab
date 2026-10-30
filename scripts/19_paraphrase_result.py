"""Score the paraphrase validity test against its pre-registered criterion.

Implements docs/paraphrase-protocol.md and nothing else. Written before
the paraphrased runs exist, on purpose: an analysis script written after
the data can be shaped by the result. Every constant that decides the
outcome is fixed below as a module constant, not a command-line flag,
because a flag is an invitation to try another value.

WHAT IS COMPARED

On the rewritten questions only (the ids in paraphrase_overlap.csv, written
by `18_paraphrase_worksheet.py --build`), span recall for four runs:

    original phrasing     dense = retrieval_baseline.csv  (Phase 5 dense)
                          bm25  = retrieval_bm25.csv
    rewritten phrasing    dense = retrieval_para_dense.csv
                          bm25  = retrieval_para_bm25.csv

    delta_original   = bm25 - dense, original phrasing
    delta_paraphrase = bm25 - dense, rewritten phrasing
    effect           = delta_original - delta_paraphrase

at the 20k character budget and at k=20, as the protocol names both. The
95% CI comes from a cluster bootstrap over questions: 10,000 resamples,
seed 20260827. All four recalls are computed on the SAME resampled
questions in each draw, so the pairing survives resampling.

THE CRITERION (copied from the protocol, not reinterpreted)

    CI on effect excludes zero, positive  -> confound explanation SUPPORTED.
        delta_paraphrase becomes the honest effect size; the aggregate
        +0.168 is reported as inflated by authoring, with the amount.
    CI on effect includes zero            -> INCONCLUSIVE, not negative.
        Aggregate reported with the sweep 2 tercile caveat retained; this
        test reported as underpowered, not as support for either side.
    CI on effect excludes zero, negative  -> confound explanation REFUTED.
        The bound lifts; +0.168 stands as an estimate of retrieval quality.

The protocol names both the 20k budget and k=20 without ranking them, so
both are scored and both verdicts are printed. If they disagree, this
script says so and does not pick one.

TWO GATES, ALSO PRE-REGISTERED

  Manipulation check  mean IDF-weighted overlap must fall by >= 0.25. Below
                      that the test is invalid: no verdict is issued.
  Guard               if dense span recall on the rewritten questions falls
                      by more than 0.15 against the same questions'
                      originals, the rewrites may have broken the questions.
                      Re-read every rewrite against rule 1, and report the
                      warning whatever the primary result says.

One apparatus check that is NOT in the protocol and decides nothing: the
questions that were not rewritten are byte-identical in both gold files,
so the paraphrased runs must score them exactly as the original runs did.
If they do not, the two pairs of runs were made against different indexes
or configurations, and the comparison is suspect. It is printed as a
warning and recorded in the table.

Writes results/tables/paraphrase_result.csv (one row per metric) and
results/tables/paraphrase_result_questions.csv (per-question hits, for
re-reading rewrites if the guard fires).

Usage:
    python scripts/19_paraphrase_result.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rageval.config import EVAL_DIR, TABLES_DIR
from rageval.evaluation.spans import count_spans

# --- pre-registered in docs/paraphrase-protocol.md; do not change ----------
MIN_OVERLAP_DROP = 0.25      # manipulation check
GUARD_DENSE_DROP = 0.15      # broken-question warning
BOOT = 10_000
SEED = 20260827
# Recall is a ratio of span counts, so a change of exactly 0.15 can compute
# as -0.15000000000000002 and fire a rule that says "more than 0.15". The
# tolerance only absorbs float error; it is far below one span (1/40).
EPS = 1e-9
METRICS = (("B20000", "hit_spans@B20000", "budget 20,000"),
           ("k20", "hit_spans@20", "k=20"))


def load_run(tag: str, tables_dir: Path) -> pd.DataFrame | None:
    """Scorable rows of one run, indexed by question id."""
    path = tables_dir / f"retrieval_{tag}.csv"
    if not path.exists():
        print(f"  missing: {path}", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    for _, col, _ in METRICS:
        if col not in df.columns:
            print(f"  {tag}: no column {col} - was it run with --k-max 50?",
                  file=sys.stderr)
            return None
    scored = df[df["scorable"] == True].copy()  # noqa: E712 - pandas mask
    return scored.set_index(scored["id"].astype(str))


def span_recall(hits: np.ndarray, n_ev: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Span recall over the questions in each row of idx (2-D) or in idx (1-D)."""
    return hits[idx].sum(axis=-1) / n_ev[idx].sum(axis=-1)


def classify(lo: float, hi: float) -> str:
    """The protocol's three-way interval rule."""
    if lo > 0:
        return "supported"
    if hi < 0:
        return "refuted"
    return "inconclusive"


READING = {
    "supported": (
        "CONFOUND EXPLANATION SUPPORTED. The advantage shrinks when vocabulary\n"
        "  is removed. Report delta_paraphrase as the honest effect size, and the\n"
        "  aggregate +0.168 as inflated by authoring, by the measured amount."),
    "inconclusive": (
        "INCONCLUSIVE - not negative. Report the aggregate with the sweep 2\n"
        "  tercile caveat retained, and this test as underpowered rather than\n"
        "  as support for either side."),
    "refuted": (
        "CONFOUND EXPLANATION REFUTED. The advantage is larger without shared\n"
        "  vocabulary. The bound lifts; +0.168 stands as an estimate of\n"
        "  retrieval quality."),
}


def analyse(hits: dict[str, np.ndarray], n_ev: np.ndarray,
            draws: np.ndarray) -> dict[str, float]:
    """Point estimates and bootstrap CIs for one metric.

    `hits` maps orig_dense / orig_bm25 / para_dense / para_bm25 to per-question
    hit counts, aligned with n_ev. `draws` is (BOOT, n) question indices,
    shared by all four arms so each resample stays paired.
    """
    every = np.arange(len(n_ev))
    point = {k: float(span_recall(v, n_ev, every)) for k, v in hits.items()}
    boot = {k: span_recall(v, n_ev, draws) for k, v in hits.items()}

    d_orig = boot["orig_bm25"] - boot["orig_dense"]
    d_para = boot["para_bm25"] - boot["para_dense"]
    effect = d_orig - d_para

    out = dict(point)
    out["delta_original"] = point["orig_bm25"] - point["orig_dense"]
    out["delta_paraphrase"] = point["para_bm25"] - point["para_dense"]
    out["effect"] = out["delta_original"] - out["delta_paraphrase"]
    for name, arr in (("delta_original", d_orig), ("delta_paraphrase", d_para),
                      ("effect", effect)):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[f"{name}_lo"], out[f"{name}_hi"] = float(lo), float(hi)
    out["dense_change"] = point["para_dense"] - point["orig_dense"]
    return out


def _load_jsonl(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        return {d["id"]: d for d in (json.loads(line) for line in fh if line.strip())}


def check_gold(eval_dir: Path, rewritten: set[str]) -> list[str]:
    """Confirm the paraphrased gold differs from the original only in the
    question text of the rewritten ids, as --build promises."""
    orig_p, para_p = eval_dir / "gold_questions.jsonl", eval_dir / "gold_paraphrased.jsonl"
    if not para_p.exists():
        return [f"{para_p} not found; cannot confirm which questions were rewritten"]
    orig, para = _load_jsonl(orig_p), _load_jsonl(para_p)
    problems = []
    if set(orig) != set(para):
        problems.append("question ids differ between the two gold files")
    changed = {q for q in set(orig) & set(para) if orig[q]["question"] != para[q]["question"]}
    if changed != rewritten:
        problems.append(f"question text differs for {sorted(changed)}, "
                        f"but paraphrase_overlap.csv lists {sorted(rewritten)}")
    for q in set(orig) & set(para):
        a = {k: v for k, v in orig[q].items() if k != "question"}
        b = {k: v for k, v in para[q].items() if k != "question"}
        if a != b:
            problems.append(f"{q}: fields other than the question differ")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--orig-dense", default="baseline",
                        help="Tag of the dense run on the original questions.")
    parser.add_argument("--orig-bm25", default="bm25",
                        help="Tag of the BM25 run on the original questions.")
    parser.add_argument("--para-dense", default="para_dense")
    parser.add_argument("--para-bm25", default="para_bm25")
    parser.add_argument("--tables-dir", default=None,
                        help="Override results/tables (for testing).")
    parser.add_argument("--eval-dir", default=None,
                        help="Override data/eval (for testing).")
    parser.add_argument("--out", default=None,
                        help="CSV path (default results/tables/paraphrase_result.csv).")
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir) if args.tables_dir else TABLES_DIR
    eval_dir = Path(args.eval_dir) if args.eval_dir else EVAL_DIR
    out_path = Path(args.out) if args.out else tables_dir / "paraphrase_result.csv"
    q_path = out_path.with_name(out_path.stem + "_questions.csv")
    # Unlink first: if this run stops early, the previous run's verdict must
    # not stay on disk reading as this one's.
    out_path.unlink(missing_ok=True)
    q_path.unlink(missing_ok=True)

    # --- the manipulation check ------------------------------------------
    ov_path = tables_dir / "paraphrase_overlap.csv"
    if not ov_path.exists():
        print(f"Missing {ov_path}. Run 18_paraphrase_worksheet.py --build first.",
              file=sys.stderr)
        return 1
    ov = pd.read_csv(ov_path)
    rewritten = sorted(ov["id"].astype(str))
    overlap_drop = float(-(ov["overlap_after"] - ov["overlap_before"]).mean())
    manipulation_ok = overlap_drop >= MIN_OVERLAP_DROP - EPS

    gold_problems = check_gold(eval_dir, set(rewritten))

    tags = {"orig_dense": args.orig_dense, "orig_bm25": args.orig_bm25,
            "para_dense": args.para_dense, "para_bm25": args.para_bm25}
    runs = {k: load_run(t, tables_dir) for k, t in tags.items()}
    if any(r is None for r in runs.values()):
        return 1

    missing = {k: sorted(set(rewritten) - set(r.index)) for k, r in runs.items()}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        print(f"  ERROR: rewritten questions absent from a run: {missing}",
              file=sys.stderr)
        return 1

    # n_evidence belongs to the gold set. If any run disagrees, it was scored
    # against a different gold file and the comparison is void.
    n_ev = runs["orig_dense"].loc[rewritten, "n_evidence"].astype(int).to_numpy()
    for k, r in runs.items():
        other = r.loc[rewritten, "n_evidence"].astype(int).to_numpy()
        if not np.array_equal(other, n_ev):
            print(f"  ERROR: {tags[k]} disagrees on n_evidence. Runs were "
                  f"scored against different gold sets.", file=sys.stderr)
            return 1

    # --- apparatus check: untouched questions must score identically -------
    untouched = sorted(set.intersection(*(set(r.index) for r in runs.values()))
                       - set(rewritten))
    drift = []
    for _, col, label in METRICS:
        for arm in ("dense", "bm25"):
            a = [count_spans(c) for c in runs[f"orig_{arm}"].loc[untouched, col]]
            b = [count_spans(c) for c in runs[f"para_{arm}"].loc[untouched, col]]
            diff = [q for q, x, y in zip(untouched, a, b) if x != y]
            if diff:
                drift.append(f"{arm} at {label}: {len(diff)} untouched question(s) "
                             f"scored differently ({', '.join(diff[:6])}"
                             f"{', ...' if len(diff) > 6 else ''})")

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(rewritten), size=(BOOT, len(rewritten)))

    print(f"\n{'=' * 84}")
    print("PARAPHRASE VALIDITY TEST   pre-registered in docs/paraphrase-protocol.md")
    print(f"{'=' * 84}")
    mix = ov["question_type"].value_counts().sort_index()
    print(f"  rewritten questions {len(rewritten)} "
          f"({', '.join(f'{t} {n}' for t, n in mix.items())})")
    print(f"  spans               {int(n_ev.sum())}")
    print(f"  bootstrap           {BOOT:,} resamples of questions, seed {SEED}")
    print(f"  runs                original {args.orig_dense} / {args.orig_bm25}, "
          f"rewritten {args.para_dense} / {args.para_bm25}")

    print(f"\n{'-' * 84}\nMANIPULATION CHECK  (mean overlap drop >= {MIN_OVERLAP_DROP:.2f})\n"
          f"{'-' * 84}")
    print(f"  overlap before {ov['overlap_before'].mean():.3f}   "
          f"after {ov['overlap_after'].mean():.3f}   drop {overlap_drop:.3f}")
    if manipulation_ok:
        print("  PASSED.")
    else:
        print("  FAILED. The rewrites did not remove enough shared vocabulary, so the")
        print("  test is invalid and no verdict is issued below. Rewrite further; do")
        print("  not reinterpret the threshold.")

    if gold_problems or drift:
        print(f"\n{'-' * 84}\nAPPARATUS WARNINGS  (not in the protocol; decide nothing)\n"
              f"{'-' * 84}")
        for p in gold_problems:
            print(f"  gold: {p}")
        for d in drift:
            print(f"  runs: {d}")
        if drift:
            print("  Untouched questions should score identically in both pairs of")
            print("  runs. They do not, so the runs differ in more than wording.")

    rows, qrows = [], {q: {"id": q} for q in rewritten}
    for suffix, col, label in METRICS:
        hits = {k: np.array([count_spans(c) for c in runs[k].loc[rewritten, col]])
                for k in runs}
        for k, v in hits.items():
            for q, h in zip(rewritten, v):
                qrows[q][f"{k}@{suffix}"] = int(h)
        r = analyse(hits, n_ev, draws)
        outcome = classify(r["effect_lo"], r["effect_hi"]) if manipulation_ok else "invalid"
        guard = r["dense_change"] < -GUARD_DENSE_DROP - EPS

        print(f"\n{'-' * 84}\nPRIMARY ANALYSIS   span recall, {label}\n{'-' * 84}")
        print(f"  {'phrasing':<12}{'dense':>9}{'bm25':>9}{'bm25 - dense':>15}{'95% CI':>21}")
        for ph, d, b, key in (("original", "orig_dense", "orig_bm25", "delta_original"),
                              ("rewritten", "para_dense", "para_bm25", "delta_paraphrase")):
            ci = f"[{r[key + '_lo']:+.3f}, {r[key + '_hi']:+.3f}]"
            print(f"  {ph:<12}{r[d]:>9.3f}{r[b]:>9.3f}{r[key]:>+15.3f}{ci:>21}")
        print(f"\n  effect = delta_original - delta_paraphrase   {r['effect']:+.3f}   "
              f"95% CI [{r['effect_lo']:+.3f}, {r['effect_hi']:+.3f}]")
        print(f"  guard: dense change on rewritten questions   {r['dense_change']:+.3f}   "
              f"(warn below -{GUARD_DENSE_DROP:.2f})")
        if guard:
            print("  BROKEN-QUESTION WARNING. Dense recall fell by more than the guard:")
            print("  re-read every rewrite against rule 1 and report this warning")
            print("  whatever the primary result says.")
        if outcome == "invalid":
            print("\n  No verdict: the manipulation check failed.")
        else:
            print(f"\n  {READING[outcome]}")

        rows.append({
            "metric": label,
            "n_questions": len(rewritten),
            "n_spans": int(n_ev.sum()),
            "orig_dense": r["orig_dense"], "orig_bm25": r["orig_bm25"],
            "para_dense": r["para_dense"], "para_bm25": r["para_bm25"],
            "delta_original": r["delta_original"],
            "delta_original_lo": r["delta_original_lo"],
            "delta_original_hi": r["delta_original_hi"],
            "delta_paraphrase": r["delta_paraphrase"],
            "delta_paraphrase_lo": r["delta_paraphrase_lo"],
            "delta_paraphrase_hi": r["delta_paraphrase_hi"],
            "effect": r["effect"], "effect_lo": r["effect_lo"], "effect_hi": r["effect_hi"],
            "outcome": outcome,
            "dense_change": r["dense_change"], "guard_triggered": guard,
            "overlap_drop": overlap_drop, "manipulation_passed": manipulation_ok,
            "apparatus_warnings": len(gold_problems) + len(drift),
            "boot": BOOT, "seed": SEED,
        })

    outcomes = {r["outcome"] for r in rows}
    print(f"\n{'-' * 84}\nREADING\n{'-' * 84}")
    if len(outcomes) > 1:
        each = ", ".join(f"{r['metric']}: {r['outcome']}" for r in rows)
        print(f"  The two metrics disagree ({each}).")
        print("  The protocol names both without ranking them, so report both and")
        print("  do not choose the one that reads better.")
    else:
        print(f"  Both metrics: {outcomes.pop()}.")
    if any(r["guard_triggered"] for r in rows):
        print("  The broken-question guard fired; that warning is part of the result.")
    print("  'Inconclusive' was declared admissible in advance: 20 questions and")
    print("  ~40 spans can detect a large change in delta, not a subtle one.")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    qdf = pd.DataFrame(list(qrows.values()))
    qdf.insert(1, "n_evidence", n_ev)
    qdf = qdf.merge(ov[["id", "question_type", "overlap_before", "overlap_after"]]
                    .assign(id=ov["id"].astype(str)), on="id", how="left")
    qdf.to_csv(q_path, index=False)
    print(f"\n  -> {out_path}\n  -> {q_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
