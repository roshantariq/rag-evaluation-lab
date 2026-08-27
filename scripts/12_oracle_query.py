"""Can a chunk retrieve itself?

The decisive test after three failed hypotheses. For every evidence span we
query the index with the span's OWN quoted text and check whether a chunk
containing that span comes back. This removes the question from the loop
entirely, so it separates two things that the baseline numbers confound:

    index / embedding health   - can the target be reached at all?
    question-passage matching  - can it be reached from the question?

Reading the result:

  * Oracle near 1.0, questions at 0.65  -> the index is sound and the whole
    gap is semantic matching. There is no bug. The fixes are the ones
    already planned: a retrieval-tuned embedding model, hybrid BM25 for
    rare exact terms, larger k.

  * Oracle also poor -> something is wrong in indexing or embedding itself,
    and the oracle failures are the shortlist to debug.

Oracle recall is also the honest ceiling for this index and chunking: no
reformulation of a question can beat querying with the answer text itself.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import yaml

from rageval.config import (
    CHROMA_DIR,
    CONFIG_DIR,
    EMBED_CACHE,
    EVAL_DIR,
    TABLES_DIR,
    ensure_dirs,
)
from rageval.embed.encoder import Encoder
from rageval.evaluation.gold import load_gold
from rageval.evaluation.retrieval_metrics import Retrieved, is_scorable
from rageval.store.chroma_store import ChromaStore


def first_hit_rank(hits, arxiv_id: str, start: int, end: int) -> int | None:
    for i, h in enumerate(hits, 1):
        if Retrieved(h.arxiv_id, h.char_start, h.char_end).overlaps(arxiv_id, start, end):
            return i
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline.yaml")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--compare", default=None,
                        help="Question-query CSV to compare against "
                             "(default results/tables/retrieval_baseline.csv)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ensure_dirs()

    with open(CONFIG_DIR / args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    store = ChromaStore(CHROMA_DIR, cfg["index"]["collection"])
    if store.count() == 0:
        print("Index is empty. Run scripts/03_build_index.py first.")
        return 1
    encoder = Encoder(cfg["embedding"]["model"], cache_path=EMBED_CACHE)

    questions = [q for q in load_gold(EVAL_DIR / "gold_questions.jsonl") if is_scorable(q)]

    rows = []
    for n, q in enumerate(questions, 1):
        for i, ev in enumerate(q.evidence):
            quote = " ".join(ev.quote.split())
            hits = store.query(encoder.encode_query(quote), k=args.k)
            rank = first_hit_rank(hits, ev.arxiv_id, ev.char_start, ev.char_end)
            rows.append({
                "id": q.id,
                "question_type": q.question_type,
                "span_index": i,
                "arxiv_id": ev.arxiv_id,
                "quote_chars": len(quote),
                "oracle_rank": rank,
                "oracle@1": rank == 1,
                "oracle@5": rank is not None and rank <= 5,
                f"oracle@{args.k}": rank is not None,
            })
        if n % 10 == 0:
            print(f"  {n}/{len(questions)} questions")

    out = pd.DataFrame(rows)
    out_path = TABLES_DIR / f"oracle_query_k{args.k}.csv"
    out.to_csv(out_path, index=False)

    k = args.k
    print(f"\n{'=' * 74}\nORACLE: query = the evidence text itself  "
          f"(n={len(out)} spans)\n{'=' * 74}")
    print(f"  oracle@1   {out['oracle@1'].mean():.3f}")
    print(f"  oracle@5   {out['oracle@5'].mean():.3f}")
    print(f"  oracle@{k:<3} {out[f'oracle@{k}'].mean():.3f}")
    found = out[out["oracle_rank"].notna()]
    if len(found):
        print(f"  median rank when found: {found['oracle_rank'].median():.0f}")

    print(f"\n{'-' * 74}\nBY QUESTION TYPE\n{'-' * 74}")
    print(f"  {'type':<14}{'spans':>7}{'oracle@1':>10}{'oracle@5':>10}{f'oracle@{k}':>11}")
    for qtype in ("factual", "comparative", "multi_hop", "ambiguous"):
        sub = out[out["question_type"] == qtype]
        if not len(sub):
            continue
        print(f"  {qtype:<14}{len(sub):>7}{sub['oracle@1'].mean():>10.3f}"
              f"{sub['oracle@5'].mean():>10.3f}{sub[f'oracle@{k}'].mean():>11.3f}")

    # --- the gap: oracle vs asking the actual question ---------------------
    cmp_path = args.compare or (TABLES_DIR / "retrieval_baseline.csv")
    try:
        qdf = pd.read_csv(cmp_path)
    except FileNotFoundError:
        qdf = None
    if qdf is not None and f"hit_spans@{k}" in qdf.columns:
        qhit = {}
        for _, r in qdf.iterrows():
            cell = r[f"hit_spans@{k}"]
            got = {int(x) for x in str(cell).split(";") if x.isdigit()} if isinstance(cell, str) else set()
            qhit[r["id"]] = got
        out["question_hit"] = [
            r["span_index"] in qhit.get(r["id"], set()) for _, r in out.iterrows()
        ]
        print(f"\n{'-' * 74}\nORACLE vs QUESTION at k={k}\n{'-' * 74}")
        print(f"  reachable by its own text   {out[f'oracle@{k}'].mean():.3f}")
        print(f"  reachable from the question {out['question_hit'].mean():.3f}")
        gap = out[f'oracle@{k}'].mean() - out['question_hit'].mean()
        print(f"  gap attributable to question-passage matching  {gap:.3f}")

        both = out[out[f"oracle@{k}"] & ~out["question_hit"]]
        neither = out[~out[f"oracle@{k}"] & ~out["question_hit"]]
        print(f"\n  findable by text, missed from question: {len(both):>3}"
              "   <- semantic matching cost")
        print(f"  missed by both:                         {len(neither):>3}"
              "   <- unreachable in this index")
        if len(neither):
            print("\n  Unreachable spans (debug these if the count is large):")
            for _, r in neither.head(20).iterrows():
                print(f"    {r['id']:<8} span {r['span_index']}  {r['arxiv_id']:<16}"
                      f"  quote {r['quote_chars']} chars")

    print(f"\n  per-span detail -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())