"""Is BM25's advantage about retrieval, or about how the gold set was written?

Every gold question was authored while reading its evidence passage. That
is the right way to build span-anchored ground truth, and it hands a
lexical retriever the exact tokens the writer had in front of them. Dense
retrieval gets no comparable gift. So the BM25 arm carries a confound the
baseline does not, and neither the paired test nor the stability analysis
can see it: both resample the same 58 questions, and a bias baked into how
all of them were authored survives any amount of resampling.

This measures the confound directly. For each question, how much of its
vocabulary - weighted the way BM25 weights it - already appears in its own
evidence? Then: does BM25's advantage over dense survive among the
questions that share the least vocabulary with their evidence?

PRE-REGISTERED CRITERION, recorded before the numbers were seen:

    Split the 58 scorable questions into terciles by IDF-weighted overlap.
    If BM25's advantage in the LOW tercile is at least half its advantage
    in the HIGH tercile, the effect is about retrieval - adopt BM25 as the
    baseline and report the full effect size.
    If the low-tercile advantage collapses toward zero, the effect is about
    authoring - still adopt BM25, but report the bounded effect and say
    plainly what the number measures.

Overlap is IDF-weighted with BM25's own IDF, over the corpus in the index,
using the retriever's own tokenizer. An unweighted count would be dominated
by "the", "of" and "model", which every question shares with every passage
and which BM25 scores at almost nothing.

Terciles of ~19 questions are underpowered on purpose - the criterion was
declared on terciles, so terciles decide it. A median split (29 vs 29) is
reported underneath as a power check, and the two should agree in
direction. If they disagree, believe neither and say so.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from rageval.config import CHROMA_DIR, EVAL_DIR, TABLES_DIR
from rageval.evaluation.gold import load_gold
from rageval.evaluation.retrieval_metrics import is_scorable
from rageval.retrieve.bm25 import tokenize
from rageval.store.chroma_store import ChromaStore


def _count_spans(cell) -> set[int]:
    """Span indices from a hit_spans cell, surviving pandas dtype guessing."""
    if cell is None:
        return set()
    try:
        if pd.isna(cell):
            return set()
    except (TypeError, ValueError):
        pass
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return set()
    out = set()
    for part in s.split(";"):
        part = part.strip()
        if part.endswith(".0"):
            part = part[:-2]
        if part.isdigit():
            out.add(int(part))
    return out


def build_idf(texts) -> tuple[dict[str, float], int]:
    """BM25 Okapi IDF over the indexed corpus.

    Deliberately the same formula rank_bm25 uses, so "overlap" means
    overlap in the quantity the retriever actually scores rather than a
    similarly-named one.
    """
    df = Counter()
    n = 0
    for t in texts:
        n += 1
        for term in set(tokenize(t)):
            df[term] += 1
    idf = {term: log((n - d + 0.5) / (d + 0.5) + 1.0) for term, d in df.items()}
    return idf, n


def idf_of(term: str, idf: dict[str, float], n: int, ceiling: float) -> float:
    """IDF for a term, including one absent from the corpus.

    A question term the corpus never uses cannot help retrieval, but it
    should not silently vanish from the denominator either - that would
    make a question look more overlapping than it is.

    It is capped at the highest IDF any real term achieves rather than
    given the formula's unbounded value. Otherwise a single inflection
    difference would outweigh every genuine rare term in the question and
    drive the overlap score toward zero for reasons that have nothing to
    do with the confound being measured.
    """
    return idf.get(term, ceiling)


def overlap_score(question: str, quotes: list[str],
                  idf: dict[str, float], n: int) -> float:
    """Share of the question's IDF mass already present in its evidence."""
    q_terms = set(tokenize(question))
    if not q_terms:
        return 0.0
    e_terms: set[str] = set()
    for quote in quotes:
        e_terms |= set(tokenize(quote))
    ceiling = max(idf.values()) if idf else 1.0
    total = sum(idf_of(t, idf, n, ceiling) for t in q_terms)
    if total <= 0:
        return 0.0
    shared = sum(idf_of(t, idf, n, ceiling) for t in q_terms & e_terms)
    return shared / total


def load_hits(tag: str, col: str, tables_dir: Path):
    path = tables_dir / f"retrieval_{tag}.csv"
    if not path.exists():
        print(f"  missing: {path}", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    if col not in df.columns:
        print(f"  {tag}: no column {col}", file=sys.stderr)
        return None
    scored = df[df["scorable"] == True]  # noqa: E712
    return {str(r["id"]): (int(r["n_evidence"]), _count_spans(r[col]))
            for _, r in scored.iterrows()}


def group_stats(qids, hits_a, hits_b, tag_a, tag_b, boot, rng):
    """Span recall for both runs over a set of questions, plus a cluster
    bootstrap CI on the difference. Questions are the resampling unit."""
    n_ev = np.array([hits_a[q][0] for q in qids])
    ha = np.array([len(hits_a[q][1]) for q in qids])
    hb = np.array([len(hits_b[q][1]) for q in qids])

    def rec(h, idx):
        tot = n_ev[idx].sum()
        return h[idx].sum() / tot if tot else float("nan")

    idx_all = np.arange(len(qids))
    ra, rb = rec(ha, idx_all), rec(hb, idx_all)
    draws = rng.integers(0, len(qids), size=(boot, len(qids)))
    diffs = np.array([rec(hb, d) - rec(ha, d) for d in draws])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n": len(qids), "spans": int(n_ev.sum()),
            tag_a: ra, tag_b: rb, "delta": rb - ra, "lo": lo, "hi": hi}


def report(title, groups, tag_a, tag_b):
    print(f"\n{'-' * 84}\n{title}\n{'-' * 84}")
    print(f"  {'group':<20}{'n':>4}{'spans':>7}{tag_a:>12}{tag_b:>12}"
          f"{'delta':>9}{'95% CI':>20}")
    for name, g in groups.items():
        ci = f"[{g['lo']:+.3f}, {g['hi']:+.3f}]"
        print(f"  {name:<20}{g['n']:>4}{g['spans']:>7}{g[tag_a]:>12.3f}"
              f"{g[tag_b]:>12.3f}{g['delta']:>+9.3f}{ci:>20}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dense", default="baseline", help="Tag of the dense run.")
    parser.add_argument("--lexical", default="bm25", help="Tag of the BM25 run.")
    parser.add_argument("--collection", default="baseline_fixed512_minilm",
                        help="Collection to take corpus IDF from.")
    parser.add_argument("--budget", type=int, default=20_000)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--out", default=None, help="Per-question CSV path.")
    args = parser.parse_args()

    col = f"hit_spans@{args.k}" if args.k else f"hit_spans@B{args.budget}"
    label = f"k={args.k}" if args.k else f"budget {args.budget:,}"

    hits_a = load_hits(args.dense, col, TABLES_DIR)
    hits_b = load_hits(args.lexical, col, TABLES_DIR)
    if hits_a is None or hits_b is None:
        return 1

    gold = {q.id: q for q in load_gold(EVAL_DIR / "gold_questions.jsonl")
            if is_scorable(q)}
    qids = sorted(set(hits_a) & set(hits_b) & set(gold))
    if not qids:
        print("No questions in common.", file=sys.stderr)
        return 1

    print("Reading corpus for IDF...")
    store = ChromaStore(CHROMA_DIR, args.collection)
    records = store.all_records()
    if not records:
        print("Collection is empty.", file=sys.stderr)
        return 1
    idf, n_docs = build_idf(r.text for r in records)

    rows = []
    for qid in qids:
        q = gold[qid]
        quotes = [ev.quote for ev in q.evidence]
        rows.append({
            "id": qid,
            "question_type": q.question_type,
            "overlap": overlap_score(q.question, quotes, idf, n_docs),
            "n_evidence": hits_a[qid][0],
            f"hits_{args.dense}": len(hits_a[qid][1]),
            f"hits_{args.lexical}": len(hits_b[qid][1]),
        })
    qdf = pd.DataFrame(rows).sort_values("overlap").reset_index(drop=True)
    out_path = args.out or (TABLES_DIR / "lexical_overlap.csv")
    qdf.to_csv(out_path, index=False)

    print(f"\n{'=' * 84}")
    print(f"LEXICAL OVERLAP CONFOUND   {args.dense} vs {args.lexical}, {label}")
    print(f"{'=' * 84}")
    print(f"  questions {len(qdf)}   corpus {n_docs} chunks   "
          f"vocabulary {len(idf):,} terms")
    o = qdf["overlap"]
    print(f"  IDF-weighted overlap: min {o.min():.3f}  median {o.median():.3f}  "
          f"max {o.max():.3f}")
    print("\n  Read as: the share of a question's IDF mass that already appears")
    print("  verbatim in its own evidence. 1.000 means every distinguishing term")
    print("  in the question is present in the passage it is scored against.")

    rng = np.random.default_rng(args.seed)
    ids = qdf["id"].tolist()

    # --- the declared test: terciles ---------------------------------------
    third = len(ids) // 3
    terciles = {
        "low overlap": ids[:third],
        "mid overlap": ids[third:2 * third],
        "high overlap": ids[2 * third:],
    }
    tg = {name: group_stats(g, hits_a, hits_b, args.dense, args.lexical,
                            args.boot, rng)
          for name, g in terciles.items()}
    report(f"BY OVERLAP TERCILE  (the pre-registered test)", tg,
           args.dense, args.lexical)

    # --- power check: median split -----------------------------------------
    half = len(ids) // 2
    halves = {"below median": ids[:half], "above median": ids[half:]}
    hg = {name: group_stats(g, hits_a, hits_b, args.dense, args.lexical,
                            args.boot, rng)
          for name, g in halves.items()}
    report("BY MEDIAN SPLIT  (power check - should agree in direction)", hg,
           args.dense, args.lexical)

    # --- the interaction itself, tested rather than eyeballed ---------------
    # Comparing the three tercile CIs by inspection is not a test: two
    # intervals can overlap while their difference still excludes zero, and
    # vice versa. This resamples the low and high terciles independently -
    # they are disjoint question sets - and puts an interval on the
    # difference of the two advantages.
    lo_ids, hi_ids = terciles["low overlap"], terciles["high overlap"]

    def delta_for(qids, draw):
        n_ev = np.array([hits_a[q][0] for q in qids])[draw]
        ha = np.array([len(hits_a[q][1]) for q in qids])[draw]
        hb = np.array([len(hits_b[q][1]) for q in qids])[draw]
        tot = n_ev.sum()
        return (hb.sum() - ha.sum()) / tot if tot else float("nan")

    inter = np.array([
        delta_for(hi_ids, rng.integers(0, len(hi_ids), len(hi_ids)))
        - delta_for(lo_ids, rng.integers(0, len(lo_ids), len(lo_ids)))
        for _ in range(args.boot)
    ])
    i_lo, i_hi = np.percentile(inter, [2.5, 97.5])

    # --- verdict -----------------------------------------------------------
    d_low = tg["low overlap"]["delta"]
    d_high = tg["high overlap"]["delta"]
    ratio = d_low / d_high if d_high > 0 else float("nan")

    print(f"\n{'-' * 84}\nINTERACTION  (does the advantage actually differ by tercile?)\n"
          f"{'-' * 84}")
    print(f"  advantage(high) - advantage(low)   {d_high - d_low:+.3f}   "
          f"95% CI [{i_lo:+.3f}, {i_hi:+.3f}]")
    if i_lo > 0 or i_hi < 0:
        print("  The interaction is real: how much a question reuses its evidence's")
        print("  vocabulary changes which retriever wins, not merely by how much.")
    else:
        print("  The interaction is NOT established. The tercile deltas differ in")
        print("  point estimate, but this interval includes zero - so 'BM25 wins")
        print("  only on high-overlap questions' is a story the data does not yet")
        print("  support. Report the terciles; do not claim the interaction.")

    print(f"\n{'-' * 84}\nVERDICT AGAINST THE PRE-REGISTERED CRITERION\n{'-' * 84}")
    print(f"  advantage in LOW overlap tercile   {d_low:+.3f}")
    print(f"  advantage in HIGH overlap tercile  {d_high:+.3f}")
    print(f"  ratio low/high                     {ratio:.2f}   "
          f"(criterion: >= 0.50)\n")

    agree = (hg["below median"]["delta"] > 0) == (hg["above median"]["delta"] > 0)
    if d_high <= 0:
        print("  The high-overlap tercile shows no advantage at all, so the")
        print("  criterion does not apply as written. Read the terciles directly")
        print("  and say so in the log rather than forcing the ratio.")
    elif ratio >= 0.5:
        print("  SURVIVES. The advantage holds where the question shares least")
        print("  vocabulary with its evidence, so it is about retrieval rather")
        print("  than about how the gold set was authored. Adopt BM25 as the")
        print("  baseline and report the full effect size.")
    else:
        print("  BOUNDED. The advantage is concentrated in questions that reuse")
        print("  their evidence's vocabulary. BM25 is still the better retriever")
        print("  here, but the headline number measures the authoring process as")
        print("  much as the retriever. Adopt it; report the low-overlap tercile")
        print("  as the honest effect size and state the confound in the README.")
    if not agree:
        print("\n  WARNING: the median split disagrees in direction with the")
        print("  terciles. At ~19 questions per tercile that is a real")
        print("  possibility. Believe neither; report both and treat the")
        print("  confound as unresolved.")

    print("\n  Two covariates this does NOT control, and which belong in the")
    print("  write-up: overlap correlates with question type (comparative")
    print("  questions name two models and so carry more rare terms), and with")
    print("  evidence span length, which the Phase 6 miss audit already found")
    print("  predicts hit rate on its own.")
    print(f"\n  per-question overlap -> {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())