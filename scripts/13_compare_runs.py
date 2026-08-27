"""Put ablation runs side by side, against the ceiling rather than in the air.

Usage:
    python scripts/13_compare_runs.py baseline chunk256 chunk1024

Reads results/tables/retrieval_<tag>.csv for each tag, and
results/tables/oracle_<tag>_k10.csv where it exists.

Absolute retrieval numbers are hard to interpret: is 0.61 good? Against a
measured ceiling it becomes answerable. Each run is therefore reported with
its own oracle ceiling and the fraction of it achieved, because the ceiling
itself moves when chunking changes - a run that raises the ceiling and the
score by the same amount has not improved the retriever.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from rageval.config import TABLES_DIR

# Must match retrieval_metrics.DEFAULT_BUDGETS.
BUDGETS = (5_000, 10_000, 20_000, 40_000)


def _count_spans(cell) -> int:
    """Count span indices in a hit_spans cell, surviving pandas dtype guessing.

    A column holding only "0" and blanks is inferred as float64, so "0"
    comes back as 0.0. Counting with a bare isdigit() then silently returns
    zero for every row - which is exactly the bug this guards against.
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


def span_recall(df: pd.DataFrame, k: int) -> float | None:
    """Span-level recall: fraction of all evidence spans retrieved.

    Question-level Recall@k asks whether ANY span was found, which is
    mechanically easier for multi-evidence questions. Span-level is the
    unit that compares cleanly against the oracle.
    """
    col = f"hit_spans@{k}"
    if col not in df.columns:
        return None
    scored = df[df["scorable"] == True]  # noqa: E712 - pandas mask
    hit = sum(_count_spans(c) for c in scored[col])
    total = int(scored["n_evidence"].sum())
    return hit / total if total else None


def span_recall_at_budget(df: pd.DataFrame, budget: int) -> float | None:
    col = f"hit_spans@B{budget}"
    if col not in df.columns:
        return None
    scored = df[df["scorable"] == True]  # noqa: E712
    hit = sum(_count_spans(c) for c in scored[col])
    total = int(scored["n_evidence"].sum())
    return hit / total if total else None


def ceiling_at_budget(odf: pd.DataFrame | None, budget: int) -> float | None:
    """Share of spans reachable by their own text within `budget` characters.

    Uses the same packing rule as take_within_budget: the hit counts if it
    is ranked first, or if everything above it plus itself fits.
    """
    if odf is None or "chars_before_hit" not in odf.columns:
        return None
    before = pd.to_numeric(odf["chars_before_hit"], errors="coerce")
    length = pd.to_numeric(odf["hit_chunk_chars"], errors="coerce")
    ok = before.notna() & ((before == 0) | (before + length <= budget))
    return float(ok.mean())


def load(tag: str, k: int):
    rpath = TABLES_DIR / f"retrieval_{tag}.csv"
    if not rpath.exists():
        return None, None
    rdf = pd.read_csv(rpath)
    # Take the deepest oracle available: the budget view needs far more
    # ranks than the k view, so oracle runs may be deeper than k.
    cands = sorted(TABLES_DIR.glob(f"oracle_{tag}_k*.csv"),
                   key=lambda p: int(p.stem.rsplit("k", 1)[-1]))
    odf = pd.read_csv(cands[-1]) if cands else None
    return rdf, odf


def ceiling_at_k(odf: pd.DataFrame | None, k: int) -> float | None:
    """Reachable-by-own-text within rank k, computed from oracle_rank.

    Reading the boolean oracle@N column would break whenever the oracle was
    run at a different depth than the k being reported.
    """
    if odf is None or "oracle_rank" not in odf.columns:
        return None
    rank = pd.to_numeric(odf["oracle_rank"], errors="coerce")
    return float((rank.notna() & (rank <= k)).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+", help="Run tags to compare.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--baseline", default=None,
                        help="Tag to measure deltas against (default: first tag).")
    args = parser.parse_args()
    k = args.k
    base_tag = args.baseline or args.tags[0]

    runs = {}
    for tag in args.tags:
        rdf, odf = load(tag, k)
        if rdf is None:
            print(f"  missing: results/tables/retrieval_{tag}.csv", file=sys.stderr)
            continue
        scored = rdf[rdf["scorable"] == True]  # noqa: E712
        runs[tag] = {
            "n": len(scored),
            "recall@5": scored["recall@5"].mean(),
            f"recall@{k}": scored[f"recall@{k}"].mean(),
            "coverage@5": scored["coverage@5"].mean(),
            f"coverage@{k}": scored[f"coverage@{k}"].mean(),
            f"ndcg@{k}": scored[f"ndcg@{k}"].mean(),
            "mrr": scored["mrr"].mean(),
            "span_recall": span_recall(rdf, k),
            "ceiling": ceiling_at_k(odf, k),
            "df": rdf,
            "odf": odf,
        }
    if not runs:
        print("Nothing to compare.")
        return 1

    print(f"\n{'=' * 86}\nQUESTION-LEVEL  (k={k})\n{'=' * 86}")
    print(f"  {'run':<14}{'n':>4}{'R@5':>8}{f'R@{k}':>8}{'Cov@5':>8}"
          f"{f'Cov@{k}':>9}{f'nDCG@{k}':>10}{'MRR':>8}")
    for tag, r in runs.items():
        print(f"  {tag:<14}{r['n']:>4}{r['recall@5']:>8.3f}{r[f'recall@{k}']:>8.3f}"
              f"{r['coverage@5']:>8.3f}{r[f'coverage@{k}']:>9.3f}"
              f"{r[f'ndcg@{k}']:>10.3f}{r['mrr']:>8.3f}")

    print(f"\n{'-' * 86}\nAGAINST THE CEILING  (span level, k={k})\n{'-' * 86}")
    print(f"  {'run':<14}{'spans found':>13}{'ceiling':>10}{'% of ceiling':>15}"
          f"{'headroom':>11}")
    for tag, r in runs.items():
        sr, ceil = r["span_recall"], r["ceiling"]
        if sr is None:
            print(f"  {tag:<14}{'(no hit_spans column)':>13}")
            continue
        if ceil:
            print(f"  {tag:<14}{sr:>13.3f}{ceil:>10.3f}{100 * sr / ceil:>14.1f}%"
                  f"{ceil - sr:>11.3f}")
        else:
            print(f"  {tag:<14}{sr:>13.3f}{'-':>10}{'-':>15}{'-':>11}")

    # --- deltas, and what share of the available gap they closed -----------
    base = runs.get(base_tag)
    if base and base["span_recall"] is not None and len(runs) > 1:
        gap = (base["ceiling"] - base["span_recall"]) if base["ceiling"] else None
        print(f"\n{'-' * 86}\nDELTA vs {base_tag}\n{'-' * 86}")
        if gap:
            print(f"  baseline headroom to its own ceiling: {gap:.3f}\n")
        print(f"  {'run':<14}{'d span recall':>15}{'d Cov@' + str(k):>12}"
              f"{'gap closed':>13}")
        for tag, r in runs.items():
            if tag == base_tag or r["span_recall"] is None:
                continue
            d = r["span_recall"] - base["span_recall"]
            dcov = r[f"coverage@{k}"] - base[f"coverage@{k}"]
            closed = f"{100 * d / gap:>12.1f}%" if gap else f"{'-':>13}"
            print(f"  {tag:<14}{d:>+15.3f}{dcov:>+12.3f}{closed}")
        print("\n  'gap closed' is the share of the BASELINE's headroom recovered.")
        print("  A run that also raises its own ceiling has changed the problem,")
        print("  not only the answer - check the ceiling column above before")
        print("  reading a large number here as an improvement.")

    # --- the comparison that is actually fair across chunkings -------------
    have_budget = all(f"recall@B{BUDGETS[0]}" in r["df"].columns for r in runs.values())
    if have_budget:
        print(f"\n{'=' * 86}\nAT EQUAL CHARACTER BUDGET\n{'=' * 86}")
        print("  Fixed k hands more text to runs with larger chunks. This holds the")
        print("  context budget constant instead - the constraint a generator has.\n")
        for b in BUDGETS:
            print(f"  budget {b:,} characters")
            print(f"    {'run':<14}{'chunks':>8}{'recall':>9}{'coverage':>10}"
                  f"{'spans':>8}{'ceiling':>9}{'% ceil':>9}")
            for tag, r in runs.items():
                df = r["df"]
                scored = df[df["scorable"] == True]  # noqa: E712
                sr = span_recall_at_budget(df, b)
                ceil = ceiling_at_budget(r["odf"], b)
                pct = f"{100 * sr / ceil:>8.1f}%" if (sr and ceil) else f"{'-':>9}"
                print(f"    {tag:<14}{scored[f'k@B{b}'].mean():>8.1f}"
                      f"{scored[f'recall@B{b}'].mean():>9.3f}"
                      f"{scored[f'coverage@B{b}'].mean():>10.3f}"
                      f"{(sr if sr is not None else float('nan')):>8.3f}"
                      f"{(ceil if ceil is not None else float('nan')):>9.3f}{pct}")
            print()

    # --- per-type coverage, the fair cross-type comparison ------------------
    print(f"\n{'-' * 86}\nCOVERAGE@{k} BY QUESTION TYPE\n{'-' * 86}")
    types = ("factual", "comparative", "multi_hop", "ambiguous")
    print(f"  {'run':<14}" + "".join(f"{t:>14}" for t in types))
    for tag, r in runs.items():
        df = r["df"]
        cells = []
        for t in types:
            sub = df[(df["question_type"] == t) & (df["scorable"] == True)]  # noqa: E712
            cells.append(f"{sub[f'coverage@{k}'].mean():>14.3f}" if len(sub) else f"{'-':>14}")
        print(f"  {tag:<14}" + "".join(cells))

    # --- how many questions found nothing at all ---------------------------
    print(f"\n{'-' * 86}\nCOMPLETE MISSES AT k={k}\n{'-' * 86}")
    for tag, r in runs.items():
        df = r["df"]
        scored = df[df["scorable"] == True]  # noqa: E712
        miss = scored[scored[f"recall@{k}"] == 0.0]
        ids = ", ".join(sorted(miss["id"]))
        print(f"  {tag:<14}{len(miss):>3}/{len(scored)}   {ids[:120]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())