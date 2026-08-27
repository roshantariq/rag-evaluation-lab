"""Print the questions retrieval never reaches, for reading by hand.

A complete miss is one of two very different things, and no aggregate can
tell them apart:

  * a genuine retrieval failure - worth reporting as a finding
  * a question whose evidence was badly chosen - a gold-set defect that
    silently contaminates every run and every comparison

Questions missed by EVERY configuration are the priority: a span no
chunking can reach is more likely to be badly anchored than uniformly hard.

Where an oracle table exists, each span is annotated with whether its own
text could reach it. That splits the diagnosis further:

    oracle hit,  question miss  -> matching failure, the interesting case
    oracle miss, question miss  -> unreachable; suspect the evidence
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from rageval.config import EVAL_DIR, TABLES_DIR
from rageval.evaluation.gold import load_gold
from rageval.evaluation.retrieval_metrics import is_scorable


def load_run(tag: str):
    path = TABLES_DIR / f"retrieval_{tag}.csv"
    if not path.exists():
        print(f"  missing {path}", file=sys.stderr)
        return None, None
    rdf = pd.read_csv(path)
    cands = sorted(TABLES_DIR.glob(f"oracle_{tag}_k*.csv"),
                   key=lambda p: int(p.stem.rsplit("k", 1)[-1]))
    odf = pd.read_csv(cands[-1]) if cands else None
    return rdf, odf


def missed_ids(rdf: pd.DataFrame, col: str) -> set[str]:
    scored = rdf[rdf["scorable"] == True]  # noqa: E712
    return set(scored[scored[col] == 0.0]["id"])


def oracle_rank(odf, qid: str, span_i: int):
    if odf is None:
        return None
    row = odf[(odf["id"] == qid) & (odf["span_index"] == span_i)]
    if row.empty:
        return None
    v = row.iloc[0]["oracle_rank"]
    return None if pd.isna(v) else int(v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=["baseline"])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--budget", type=int, default=None,
                        help="Use recall@B<budget> instead of recall@k.")
    parser.add_argument("--quote-chars", type=int, default=900)
    args = parser.parse_args()

    col = f"recall@B{args.budget}" if args.budget else f"recall@{args.k}"
    label = f"budget {args.budget:,}" if args.budget else f"k={args.k}"

    runs, oracles = {}, {}
    for tag in args.tags:
        rdf, odf = load_run(tag)
        if rdf is None:
            continue
        if col not in rdf.columns:
            print(f"  {tag}: no column {col}", file=sys.stderr)
            continue
        runs[tag] = missed_ids(rdf, col)
        oracles[tag] = odf
    if not runs:
        print("No runs loaded.")
        return 1

    everywhere = set.intersection(*runs.values())
    anywhere = set.union(*runs.values())

    gold = {q.id: q for q in load_gold(EVAL_DIR / "gold_questions.jsonl") if is_scorable(q)}

    print(f"{'=' * 78}")
    print(f"COMPLETE MISSES at {label}   runs: {', '.join(runs)}")
    print(f"{'=' * 78}")
    for tag, ids in runs.items():
        print(f"  {tag:<12} {len(ids):>3} missed")
    print(f"\n  missed by EVERY run: {len(everywhere)}  ({', '.join(sorted(everywhere))})")
    print(f"  missed by any run:   {len(anywhere)}\n")

    def show(qid: str) -> None:
        q = gold.get(qid)
        if q is None:
            return
        who = [t for t, ids in runs.items() if qid in ids]
        print(f"\n{'-' * 78}")
        print(f"{q.id}  [{q.question_type}/{getattr(q, 'difficulty', '?')}]  "
              f"missed by: {', '.join(who)}")
        print(f"{'-' * 78}")
        print(f"Q: {q.question}")
        ans = getattr(q, "reference_answer", "")
        print(f"A: {ans[:400]}{'...' if len(ans) > 400 else ''}")
        note = getattr(q, "note", "") or ""
        if note:
            print(f"note: {note[:200]}")
        for i, ev in enumerate(q.evidence):
            marks = []
            for tag in runs:
                r = oracle_rank(oracles.get(tag), qid, i)
                marks.append(f"{tag}={'miss' if r is None else f'rank {r}'}")
            print(f"\n  span {i}: {ev.arxiv_id}  {ev.char_start}-{ev.char_end}  "
                  f"({ev.char_end - ev.char_start} chars)")
            print(f"    oracle: {', '.join(marks)}")
            quote = " ".join(ev.quote.split())
            print(f"    \"{quote[:args.quote_chars]}"
                  f"{'...' if len(quote) > args.quote_chars else ''}\"")

    if everywhere:
        print(f"\n{'#' * 78}\n# MISSED BY EVERY RUN - check these for gold-set defects\n{'#' * 78}")
        for qid in sorted(everywhere):
            show(qid)

    rest = sorted(anywhere - everywhere)
    if rest:
        print(f"\n\n{'#' * 78}\n# MISSED BY SOME RUNS\n{'#' * 78}")
        for qid in rest:
            show(qid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())