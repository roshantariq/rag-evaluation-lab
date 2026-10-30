"""Apply the end-of-phase replication's pre-registered rule.

Implements docs/replication-protocol.md and nothing else. Written and
tested before the replication runs exist, like 19-24. The constants that
decide the outcome are module constants, not flags.

Sweep 1 found chunk size null under dense retrieval (fixed_256 vs
fixed_512). The pipeline has since moved to BM25, whose length
normalisation behaves nothing like an encoder that truncates at 256 word
pieces, so that null does not transfer automatically. This repeats the
comparison under BM25.

WHAT IT READS

    paired_bm25_bm25_chunk256_B20000.csv       from 16 (the decision)
    stability_bm25_bm25_chunk256_B20000.csv    from 15 (the decision)
    retrieval_baseline.csv, retrieval_chunk256.csv          dense, sweep 1
    retrieval_bm25.csv, retrieval_bm25_chunk256.csv         BM25
    retrieval_rerank.csv, retrieval_rerank_chunk256.csv     exploratory

WHAT IT DECIDES

  bm25_chunk256 against bm25 (fixed_512) at the 20k budget, by the shared
  decision table, win-rate threshold 90%. If it replaces, fixed_256
  becomes the baseline chunk size; otherwise fixed_512 stays.

WHAT IT REPORTS AND DOES NOT DECIDE

  * The size effect under dense retrieval (sweep 1's tables, recomputed).
  * The interaction: (256 - 512 under BM25) - (256 - 512 under dense),
    with a cluster-bootstrap CI over questions.
  * Exploratory: reranked BM25 at 256 against reranked BM25 at 512.

Writes results/tables/replication_result.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rageval.config import TABLES_DIR
from rageval.evaluation import decision
from rageval.evaluation.spans import count_spans
from rageval.provenance import record

# --- pre-registered in docs/replication-protocol.md; do not change ---------
CONTROL, ARM = "bm25", "bm25_chunk256"
DENSE = ("baseline", "chunk256")                     # 512, 256 under dense
EXPLORATORY = ("rerank", "rerank_chunk256")          # 512, 256 reranked
WIN_THRESHOLD = 0.90
BOOT = 10_000
SEED = 20260827
COL = "hit_spans@B20000"
LABEL = "budget 20,000"
PAIRED = "paired_bm25_bm25_chunk256_B20000.csv"
STABILITY = "stability_bm25_bm25_chunk256_B20000.csv"


def load_hits(tag: str, tables: Path) -> pd.DataFrame:
    df = pd.read_csv(tables / f"retrieval_{tag}.csv")
    scored = df[df["scorable"] == True].copy()  # noqa: E712 - pandas mask
    scored.index = scored["id"].astype(str)
    scored["hits"] = [count_spans(c) for c in scored[COL]]
    return scored[["hits", "n_evidence"]]


def paired_delta(a: np.ndarray, b: np.ndarray, n_ev: np.ndarray,
                 draws: np.ndarray) -> tuple[float, float, float]:
    """(b - a) span recall, with a 95% cluster-bootstrap CI over questions."""
    d = (b - a).astype(float)
    boot = d[draws].sum(axis=1) / n_ev[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.sum() / n_ev.sum()), float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", default=None, help="Override results/tables (for testing).")
    args = parser.parse_args()

    tables = Path(args.tables_dir) if args.tables_dir else TABLES_DIR
    out_path = tables / "replication_result.csv"
    out_path.unlink(missing_ok=True)

    tags = (CONTROL, ARM) + DENSE + EXPLORATORY
    needed = [PAIRED, STABILITY] + [f"retrieval_{t}.csv" for t in tags]
    missing = [n for n in needed if not (tables / n).exists()]
    if missing:
        print(f"  missing: {missing}. Run 10, 16 and 15 first.", file=sys.stderr)
        return 1

    paired = pd.read_csv(tables / PAIRED)
    stab = pd.read_csv(tables / STABILITY)
    if set(paired["metric"]) != {LABEL} or set(stab["metric"]) != {LABEL}:
        print(f"  tables were not computed at the pre-registered {LABEL!r}.", file=sys.stderr)
        return 1
    p = paired[(paired["a"] == CONTROL) & (paired["b"] == ARM)]
    s = stab[stab["run"] == ARM]
    if len(p) != 1 or len(s) != 1:
        print(f"  no single {CONTROL}-vs-{ARM} row in the paired or stability table.",
              file=sys.stderr)
        return 1
    p, s = p.iloc[0], s.iloc[0]
    outcome = decision.classify(p["ci_lo"], p["ci_hi"], s["win_rate"], WIN_THRESHOLD)
    chosen = decision.decide({ARM: outcome}, CONTROL)

    # --- the reported-only estimates, all on the same questions and draws ----
    runs = {t: load_hits(t, tables) for t in tags}
    ids = sorted(set.intersection(*(set(r.index) for r in runs.values())))
    n_ev = runs[CONTROL].loc[ids, "n_evidence"].astype(int).to_numpy()
    for t, r in runs.items():
        if not np.array_equal(r.loc[ids, "n_evidence"].astype(int).to_numpy(), n_ev):
            print(f"  {t} disagrees on n_evidence: different gold sets.", file=sys.stderr)
            return 1
    h = {t: runs[t].loc[ids, "hits"].to_numpy() for t in tags}
    draws = np.random.default_rng(SEED).integers(0, len(ids), size=(BOOT, len(ids)))

    bm25_eff = paired_delta(h[CONTROL], h[ARM], n_ev, draws)
    dense_eff = paired_delta(h[DENSE[0]], h[DENSE[1]], n_ev, draws)
    inter_d = (h[ARM] - h[CONTROL]) - (h[DENSE[1]] - h[DENSE[0]])
    inter = paired_delta(np.zeros_like(inter_d), inter_d, n_ev, draws)
    expl = paired_delta(h[EXPLORATORY[0]], h[EXPLORATORY[1]], n_ev, draws)

    print(f"\n{'=' * 84}\nEND-OF-PHASE REPLICATION   pre-registered in docs/replication-protocol.md\n"
          f"{'=' * 84}")
    print(f"  DECIDES   fixed_256 vs fixed_512 under BM25, span recall, {LABEL}")
    print(f"    fixed_512 {p['recall_a']:.3f}   fixed_256 {p['recall_b']:.3f}   "
          f"delta {p['delta']:+.3f}   95% CI [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]")
    print(f"    Holm p {p['p_holm']:.4f}   win rate {s['win_rate']:.1%}   -> {outcome}")
    if chosen == CONTROL:
        print("    fixed_512 STAYS the baseline chunk size.")
    else:
        print("    fixed_256 REPLACES fixed_512 as the baseline chunk size.")

    print(f"\n  REPORTED ONLY   ({len(ids)} questions, {int(n_ev.sum())} spans, same draws)")
    for name, (est, lo, hi) in (("size effect, BM25 (256 - 512)", bm25_eff),
                                ("size effect, dense (256 - 512), sweep 1", dense_eff),
                                ("interaction, BM25 minus dense", inter),
                                ("exploratory: reranked BM25 (256 - 512)", expl)):
        print(f"    {name:<42} {est:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")

    pd.DataFrame([{
        "metric": LABEL, "n_questions": len(ids), "n_spans": int(n_ev.sum()),
        "recall_512": p["recall_a"], "recall_256": p["recall_b"], "delta": p["delta"],
        "ci_lo": p["ci_lo"], "ci_hi": p["ci_hi"], "p_holm": p["p_holm"],
        "win_rate": s["win_rate"], "outcome": outcome, "chosen": chosen,
        "bm25_effect": bm25_eff[0], "dense_effect": dense_eff[0],
        "dense_effect_lo": dense_eff[1], "dense_effect_hi": dense_eff[2],
        "interaction": inter[0], "interaction_lo": inter[1], "interaction_hi": inter[2],
        "rerank_effect": expl[0], "rerank_effect_lo": expl[1], "rerank_effect_hi": expl[2],
        "boot": BOOT, "seed": SEED,
    }]).to_csv(out_path, index=False)
    record([out_path])
    print(f"\n  -> {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
