"""Apply sweep 3's pre-registered criteria to the reranking runs.

Implements docs/reranking-protocol.md and nothing else. Written and tested
before any reranking run exists, as 19 was for the paraphrase test, so
that the analysis cannot be shaped by the result. The constants that
decide outcomes are module constants, not flags.

WHAT IT READS

    retrieval_bm25.csv, retrieval_rerank.csv            original wording
    retrieval_para_bm25.csv, retrieval_para_rerank.csv  paraphrased wording
    paraphrase_overlap.csv                              which 20 were rewritten
    paired_baseline_bm25_rerank_B20000.csv              from 16
    stability_bm25_rerank_B20000.csv                    from 15

Run 16 and 15 first (see the protocol's procedure). This script does not
recompute their numbers. It reads them, so the decision rests on exactly
the instruments the protocol names.

WHAT IT DECIDES

  Primary, rerank vs bm25 at the 20k budget:
    Replaces BM25  CI excludes zero, positive, AND win rate >= 90%
    Leaning        CI excludes zero, positive, win rate < 90%
    Null           CI includes zero (not separable, which is not "equal").
                   Written as "not_separable": pandas reads "null" in a
                   CSV back as a missing value.
    Worse          CI excludes zero, negative

  Secondary, on the 20 rewritten questions (cannot overturn the primary):
    gain_original   = rerank - bm25,           original wording
    gain_paraphrase = para_rerank - para_bm25, rewritten wording
    CI on gain_paraphrase: positive -> survives the confound;
    includes zero -> inconclusive, not negative; negative -> hurts.

WHAT IT REPORTS AND DOES NOT DECIDE

  * A paired randomization test over questions for rerank vs bm25, beside
    McNemar. McNemar treats clustered spans as independent.
  * gain_original - gain_paraphrase with its CI.
  * Determinism: the questions the paraphrase test did not rewrite are
    identical in both gold files, so rerank and para_rerank must score them
    identically. If not, the reranker is nondeterministic, and that is
    printed before any number is read.
  * Latency and the candidate ceiling, from the rerank table.

Writes results/tables/rerank_result.csv and rerank_result_questions.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rageval.config import TABLES_DIR

# --- pre-registered in docs/reranking-protocol.md; do not change -----------
WIN_THRESHOLD = 0.90
BOOT = 10_000
PERMS = 10_000
SEED = 20260827
COL = "hit_spans@B20000"
LABEL = "budget 20,000"


def _count_spans(cell) -> int:
    """Count span indices in a hit_spans cell, surviving pandas dtype guessing.

    A column holding only "0" and blanks is inferred as float64, so "0" comes
    back as 0.0 and a bare isdigit() counts nothing. Sixth copy of this
    guard; the case for hoisting it into rageval is now overdue.
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


def load_run(tag: str, tables_dir: Path) -> pd.DataFrame | None:
    path = tables_dir / f"retrieval_{tag}.csv"
    if not path.exists():
        print(f"  missing: {path}", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    if COL not in df.columns:
        print(f"  {tag}: no column {COL}", file=sys.stderr)
        return None
    scored = df[df["scorable"] == True].copy()  # noqa: E712 - pandas mask
    scored.index = scored["id"].astype(str)
    scored["hits"] = [_count_spans(c) for c in scored[COL]]
    return scored


def classify_primary(lo: float, hi: float, win_rate: float) -> str:
    """The protocol's four-way decision table."""
    if hi < 0:
        return "worse"
    if lo > 0:
        return "replaces" if win_rate >= WIN_THRESHOLD else "leaning"
    return "not_separable"


def classify_secondary(lo: float, hi: float) -> str:
    if lo > 0:
        return "survives"
    if hi < 0:
        return "hurts"
    return "inconclusive"


PRIMARY_READING = {
    "replaces": "RERANKING REPLACES BM25. The CI excludes zero and the win rate clears\n"
                "  90%. Sweep 4 inherits BM25 + reranker.",
    "leaning": "LEANING, NOT DECISIVE. The CI excludes zero but the win rate is below\n"
               "  90%. The baseline stays BM25; reported as a direction, not a result.",
    "not_separable": "NULL - NOT SEPARABLE. The CI includes zero. The baseline stays BM25.\n"
            "  'Not separable' is not 'equal'.",
    "worse": "WORSE. The CI excludes zero on the negative side: reranking measurably\n"
             "  hurt. The baseline stays BM25.",
}
SECONDARY_READING = {
    "survives": "The gain survives on questions without shared vocabulary: it is not\n"
                "  just word overlap.",
    "inconclusive": "Inconclusive, not negative. 20 questions detect a large gain, not a\n"
                    "  modest one.",
    "hurts": "Reranking hurts on the paraphrased questions. Reported whatever the\n"
             "  primary result.",
}


def randomization_p(ha: np.ndarray, hb: np.ndarray, n_ev: np.ndarray,
                    rng: np.random.Generator) -> tuple[float, float]:
    """Two-sided paired randomization test over questions.

    Under the null the two arms are exchangeable within a question, so each
    question's difference in span hits can have its sign flipped. The
    statistic is the difference in span recall.
    """
    d = (hb - ha).astype(float)
    total = n_ev.sum()
    observed = d.sum() / total
    signs = rng.choice([-1.0, 1.0], size=(PERMS, len(d)))
    null = (signs * d).sum(axis=1) / total
    p = (np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (PERMS + 1)
    return float(observed), float(p)


def recall(h: np.ndarray, n_ev: np.ndarray, idx) -> np.ndarray:
    return h[idx].sum(axis=-1) / n_ev[idx].sum(axis=-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", default=None, help="Override results/tables (for testing).")
    parser.add_argument("--paired", default="paired_baseline_bm25_rerank_B20000.csv")
    parser.add_argument("--stability", default="stability_bm25_rerank_B20000.csv")
    args = parser.parse_args()

    tables = Path(args.tables_dir) if args.tables_dir else TABLES_DIR
    out_path = tables / "rerank_result.csv"
    q_path = tables / "rerank_result_questions.csv"
    out_path.unlink(missing_ok=True)
    q_path.unlink(missing_ok=True)

    runs = {t: load_run(t, tables) for t in ("bm25", "rerank", "para_bm25", "para_rerank")}
    if any(r is None for r in runs.values()):
        return 1
    for name in (args.paired, args.stability, "paraphrase_overlap.csv"):
        if not (tables / name).exists():
            print(f"  missing: {tables / name}. Run 16, 15 and 18 first.", file=sys.stderr)
            return 1

    # --- the primary decision, from 16 and 15 exactly as they wrote it ------
    paired = pd.read_csv(tables / args.paired)
    row = paired[(paired["a"] == "bm25") & (paired["b"] == "rerank")]
    stab = pd.read_csv(tables / args.stability)
    srow = stab[stab["run"] == "rerank"]
    if len(row) != 1 or len(srow) != 1:
        print("  paired or stability table has no single bm25-vs-rerank row.", file=sys.stderr)
        return 1
    row, srow = row.iloc[0], srow.iloc[0]
    if row["metric"] != LABEL or srow["metric"] != LABEL:
        print(f"  tables were computed at {row['metric']!r} / {srow['metric']!r}, "
              f"not the pre-registered {LABEL!r}.", file=sys.stderr)
        return 1
    primary = classify_primary(row["ci_lo"], row["ci_hi"], srow["win_rate"])

    # --- alignment ------------------------------------------------------------
    ids = sorted(set(runs["bm25"].index) & set(runs["rerank"].index))
    n_ev = runs["bm25"].loc[ids, "n_evidence"].astype(int).to_numpy()
    if not np.array_equal(runs["rerank"].loc[ids, "n_evidence"].astype(int).to_numpy(), n_ev):
        print("  bm25 and rerank disagree on n_evidence: different gold sets.", file=sys.stderr)
        return 1
    ha = runs["bm25"].loc[ids, "hits"].to_numpy()
    hb = runs["rerank"].loc[ids, "hits"].to_numpy()

    rng = np.random.default_rng(SEED)
    observed, p_rand = randomization_p(ha, hb, n_ev, rng)

    # --- determinism on untouched questions ----------------------------------
    rewritten = sorted(pd.read_csv(tables / "paraphrase_overlap.csv")["id"].astype(str))
    untouched = sorted(set(ids) & set(runs["para_rerank"].index) - set(rewritten))
    drift = [q for q in untouched
             if runs["rerank"].loc[q, "hits"] != runs["para_rerank"].loc[q, "hits"]]

    # --- secondary: the 20 rewritten questions ------------------------------
    missing = [q for q in rewritten if q not in runs["para_rerank"].index
               or q not in runs["para_bm25"].index or q not in runs["rerank"].index]
    if missing:
        print(f"  rewritten questions missing from a run: {missing}", file=sys.stderr)
        return 1
    ev = runs["bm25"].loc[rewritten, "n_evidence"].astype(int).to_numpy()
    arms = {k: runs[k].loc[rewritten, "hits"].to_numpy() for k in runs}
    draws = rng.integers(0, len(rewritten), size=(BOOT, len(rewritten)))
    every = np.arange(len(rewritten))
    g_orig = recall(arms["rerank"], ev, every) - recall(arms["bm25"], ev, every)
    g_para = recall(arms["para_rerank"], ev, every) - recall(arms["para_bm25"], ev, every)
    b_orig = recall(arms["rerank"], ev, draws) - recall(arms["bm25"], ev, draws)
    b_para = recall(arms["para_rerank"], ev, draws) - recall(arms["para_bm25"], ev, draws)
    ci = {name: tuple(float(x) for x in np.percentile(arr, [2.5, 97.5]))
          for name, arr in (("orig", b_orig), ("para", b_para), ("diff", b_orig - b_para))}
    secondary = classify_secondary(*ci["para"])

    # --- report -------------------------------------------------------------
    print(f"\n{'=' * 84}\nSWEEP 3 - RERANKING   pre-registered in docs/reranking-protocol.md\n"
          f"{'=' * 84}")
    if drift:
        print(f"  DETERMINISM WARNING: {len(drift)} untouched question(s) scored differently "
              f"in rerank and para_rerank ({', '.join(drift[:8])}).")
        print("  The reranker is not deterministic; read every number below with that in mind.\n")
    else:
        print(f"  determinism   {len(untouched)} untouched questions identical in rerank "
              f"and para_rerank\n")

    print(f"{'-' * 84}\nPRIMARY   rerank vs bm25, span recall, {LABEL}\n{'-' * 84}")
    print(f"  bm25 {row['recall_a']:.3f}   rerank {row['recall_b']:.3f}   "
          f"delta {row['delta']:+.3f}   95% CI [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]")
    print(f"  Holm p (McNemar, 3 pairs) {row['p_holm']:.4f}   "
          f"randomization p {p_rand:.4f}   win rate {srow['win_rate']:.1%}")
    print(f"\n  {PRIMARY_READING[primary]}")

    print(f"\n{'-' * 84}\nSECONDARY   the {len(rewritten)} rewritten questions, {LABEL}\n"
          f"{'-' * 84}")
    print(f"  gain, original wording    {g_orig:+.3f}   95% CI "
          f"[{ci['orig'][0]:+.3f}, {ci['orig'][1]:+.3f}]")
    print(f"  gain, rewritten wording   {g_para:+.3f}   95% CI "
          f"[{ci['para'][0]:+.3f}, {ci['para'][1]:+.3f}]")
    print(f"  difference (decides nothing) {g_orig - g_para:+.3f}   95% CI "
          f"[{ci['diff'][0]:+.3f}, {ci['diff'][1]:+.3f}]")
    print(f"\n  {SECONDARY_READING[secondary]}")
    print("  The secondary analysis cannot overturn the primary decision.")

    rr = runs["rerank"]
    lat = {}
    if "rerank_s" in rr.columns:
        lat = {"first_stage_s_median": float(rr["first_stage_s"].median()),
               "rerank_s_median": float(rr["rerank_s"].median())}
        print(f"\n  latency per question, median: first stage "
              f"{lat['first_stage_s_median']:.3f}s, rerank {lat['rerank_s_median']:.3f}s")

    pd.DataFrame([{
        "metric": LABEL, "n_questions": len(ids), "n_spans": int(n_ev.sum()),
        "recall_bm25": row["recall_a"], "recall_rerank": row["recall_b"],
        "delta": row["delta"], "ci_lo": row["ci_lo"], "ci_hi": row["ci_hi"],
        "p_holm": row["p_holm"], "p_randomization": p_rand,
        "win_rate": srow["win_rate"], "primary": primary,
        "gain_original": g_orig, "gain_original_lo": ci["orig"][0],
        "gain_original_hi": ci["orig"][1],
        "gain_paraphrase": g_para, "gain_paraphrase_lo": ci["para"][0],
        "gain_paraphrase_hi": ci["para"][1],
        "gain_diff": g_orig - g_para, "gain_diff_lo": ci["diff"][0],
        "gain_diff_hi": ci["diff"][1], "secondary": secondary,
        "untouched_drift": len(drift), **lat,
        "boot": BOOT, "perms": PERMS, "seed": SEED,
    }]).to_csv(out_path, index=False)
    pd.DataFrame({"id": rewritten, "n_evidence": ev,
                  **{f"{k}_hits": v for k, v in arms.items()}}).to_csv(q_path, index=False)
    print(f"\n  -> {out_path}\n  -> {q_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
