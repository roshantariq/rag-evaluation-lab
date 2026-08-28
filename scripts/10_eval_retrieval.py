"""Score the retriever against the gold set by character-span overlap.

Retrieval only - no LLM, no API key. This is the first script that turns
the gold set into numbers.

Two things are worth knowing before reading the output:

  * Unanswerable questions carry no evidence and are excluded, not zeroed.
    They are scored in the generation phase instead.
  * Recall@k asks whether ANY evidence span was found; Coverage@k asks what
    fraction of them were. They agree on single-evidence questions and
    diverge on the rest, which is where multi-hop behaviour shows up.

Three retrieval modes share this one scoring path deliberately. A separate
script per arm would let the arms drift apart in how they score, which is
the failure this project keeps finding in other guises:

    dense   the baseline - vector similarity alone
    bm25    lexical only, over the identical chunk set
    hybrid  fused, either by reciprocal rank fusion or by interleaving

The bm25 arm exists because a hybrid result is uninterpretable without it.
If hybrid beats dense but bm25 alone beats both, the dense half is dead
weight; if bm25 alone is near zero and hybrid still improves, the fusion is
doing real work.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict

import pandas as pd
import yaml

from rageval.chunking.base import chunk_document
from rageval.config import (
    CHROMA_DIR,
    CONFIG_DIR,
    EMBED_CACHE,
    EVAL_DIR,
    INTERIM_DIR,
    TABLES_DIR,
    ensure_dirs,
)
from rageval.embed.encoder import Encoder
from rageval.evaluation.gold import load_gold
from rageval.evaluation.retrieval_metrics import (
    DEFAULT_BUDGETS,
    DEFAULT_KS,
    Retrieved,
    aggregate,
    aggregate_by_type,
    count_relevant_chunks,
    evaluate_question,
    is_scorable,
    spans_from_question,
)
from rageval.retrieve.bm25 import BM25Retriever
from rageval.retrieve.fusion import (
    DEFAULT_RRF_K,
    interleave,
    provenance,
    reciprocal_rank_fusion,
)
from rageval.store.chroma_store import ChromaStore

MODES = ("dense", "bm25", "hybrid")
FUSIONS = ("rrf", "interleave")


def interim_path(arxiv_id: str):
    return INTERIM_DIR / f"{arxiv_id.replace('/', '_')}.json"


def rebuild_chunks(arxiv_ids, cfg) -> dict[str, list]:
    """Rebuild chunks for the papers the gold set cites.

    Only chunks from the same paper can overlap an evidence span, so
    counting over these papers is exactly equivalent to counting over the
    whole index - just far cheaper.

    Chunking is deterministic given the same settings, so these are the
    same chunks the index holds. If the index was built with different
    settings the sanity check below will say so rather than silently
    producing wrong IDCG denominators.
    """
    ch = cfg.get("chunking", {})
    strategy = ch.get("strategy", "fixed_512")
    target = ch.get("target_tokens", 512)
    overlap = ch.get("overlap_tokens", 0)

    out: dict[str, list] = {}
    for arxiv_id in sorted(arxiv_ids):
        path = interim_path(arxiv_id)
        if not path.exists():
            logging.warning("No extracted text for %s at %s", arxiv_id, path)
            continue
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        out[arxiv_id] = chunk_document(
            doc, strategy=strategy, target_tokens=target, overlap_tokens=overlap
        )
    return out


def chain_tally(rows, k: int, qtype: str) -> dict[str, list[str]]:
    """Which half of a two-span question was found.

    For multi_hop, evidence[0] is the source paper carrying the referring
    sentence ("FuXi follows the approach of another model...") and
    evidence[1] is the terminal paper holding the answer. So "second only"
    means the retriever found the answer passage without ever finding the
    sentence that identifies it - the failure mode the referring-sentence
    hypothesis predicts.

    For comparative the ordering is just paper A and paper B, with no
    semantic direction, so read that tally as a balance check rather than
    as evidence about chains.
    """
    tally: dict[str, list[str]] = {"neither": [], "first only": [],
                                   "second only": [], "both": []}
    for r in rows:
        if r.get("question_type") != qtype or not r.get("scorable"):
            continue
        if r.get("n_evidence") != 2:
            continue
        hit = {x for x in str(r.get(f"hit_spans@{k}", "")).split(";") if x}
        if not hit:
            key = "neither"
        elif hit == {"0"}:
            key = "first only"
        elif hit == {"1"}:
            key = "second only"
        else:
            key = "both"
        tally[key].append(r["id"])
    return tally


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline.yaml")
    parser.add_argument("--k-max", type=int, default=max(DEFAULT_KS))
    parser.add_argument("--gold", default=None, help="Defaults to data/eval/gold_questions.jsonl")
    parser.add_argument("--out", default=None, help="Per-question CSV path.")
    parser.add_argument("--tag", default="baseline", help="Names the output files.")
    parser.add_argument("--no-idcg-count", action="store_true",
                        help="Skip the relevant-chunk scan; nDCG then uses a weaker ideal.")
    parser.add_argument("--retriever", default=None, choices=MODES,
                        help="Overrides retrieval.mode in the config.")
    parser.add_argument("--pool", type=int, default=100,
                        help="Candidates fetched from EACH retriever before "
                             "fusion (hybrid only). Held constant across runs: "
                             "a deeper pool is more candidates to rerank, not "
                             "more results, but it must still be declared.")
    parser.add_argument("--fusion", default="rrf", choices=FUSIONS,
                        help="How the two ranked lists are combined. "
                             "'interleave' has no constant to choose and is "
                             "the control for any tuned rrf result.")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K,
                        help="Reciprocal rank fusion constant. See fusion.py "
                             "for what it does to unique recall.")
    parser.add_argument("--interleave-first", default="bm25", choices=("bm25", "dense"),
                        help="Which retriever gets position 1 when "
                             "interleaving. Affects exactly one rank; stated "
                             "here rather than buried in the fusion code.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3",
                  "chromadb", "rageval.embed.encoder"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ensure_dirs()

    with open(CONFIG_DIR / args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    mode = args.retriever or cfg.get("retrieval", {}).get("mode", "dense")
    if mode not in MODES:
        print(f"Unknown retrieval mode {mode!r}; expected one of {MODES}.")
        return 1

    gold_path = args.gold or (EVAL_DIR / "gold_questions.jsonl")
    questions = load_gold(gold_path)
    scorable = [q for q in questions if is_scorable(q)]
    print(f"Gold set: {len(questions)} questions, {len(scorable)} with evidence "
          f"({len(questions) - len(scorable)} unanswerable, excluded)\n")

    store = ChromaStore(CHROMA_DIR, cfg["index"]["collection"])
    if store.count() == 0:
        print("Index is empty. Run scripts/03_build_index.py first.")
        return 1

    # --- what this run actually is, printed so the output is self-describing
    print(f"Retrieval mode : {mode}")
    print(f"  collection   : {cfg['index']['collection']}  ({store.count()} chunks)")
    if mode in ("dense", "hybrid"):
        print(f"  embedding    : {cfg['embedding']['model']}")
    if mode == "hybrid":
        if args.fusion == "rrf":
            print(f"  fusion       : RRF k={args.rrf_k}, pool={args.pool} per retriever")
        else:
            print(f"  fusion       : interleave, {args.interleave_first} first, "
                  f"pool={args.pool} per retriever")
    print(f"  depth        : k_max={args.k_max}\n")

    # --- relevant-chunk counts, for an honest IDCG -------------------------
    n_relevant: dict[str, int] = {}
    known_spans: set[tuple[str, int, int]] = set()
    if not args.no_idcg_count:
        papers = {s.arxiv_id for q in scorable for s in spans_from_question(q)}
        chunks_by_paper = rebuild_chunks(papers, cfg)
        for chunks in chunks_by_paper.values():
            for c in chunks:
                known_spans.add((c.arxiv_id, c.char_start, c.char_end))
        for q in scorable:
            spans = spans_from_question(q)
            pool = [c for pid in {s.arxiv_id for s in spans}
                    for c in chunks_by_paper.get(pid, [])]
            n_relevant[q.id] = count_relevant_chunks(pool, spans)
        counts = sorted(n_relevant.values())
        print(f"Relevant chunks per question: min {counts[0]}, "
              f"median {counts[len(counts) // 2]}, max {counts[-1]}")
        if counts[0] == 0:
            zero = [qid for qid, n in n_relevant.items() if n == 0]
            print(f"  WARNING: {len(zero)} question(s) have no overlapping chunk at all: "
                  f"{', '.join(zero[:8])}")
            print("  Those are unreachable under this chunking and will score 0 by "
                  "construction.")
        print()

    # --- retrievers --------------------------------------------------------
    encoder = None
    if mode in ("dense", "hybrid"):
        encoder = Encoder(cfg["embedding"]["model"], cache_path=EMBED_CACHE)

    bm25 = None
    if mode in ("bm25", "hybrid"):
        print("Building BM25 index from the stored chunks...")
        bm25 = BM25Retriever(store.all_records())
        print(f"  {len(bm25)} chunks indexed lexically\n")

    prov_total: dict[str, int] = defaultdict(int)
    PROV_K = 10

    def get_hits(question: str) -> list:
        if mode == "dense":
            return store.query(encoder.encode_query(question), k=args.k_max)
        if mode == "bm25":
            return bm25.query(question, k=args.k_max)

        dense_hits = store.query(encoder.encode_query(question), k=args.pool)
        sparse_hits = bm25.query(question, k=args.pool)
        if args.fusion == "rrf":
            fused = reciprocal_rank_fusion([dense_hits, sparse_hits],
                                           k=args.rrf_k, top_k=args.k_max)
        else:
            order = ([sparse_hits, dense_hits] if args.interleave_first == "bm25"
                     else [dense_hits, sparse_hits])
            fused = interleave(order, top_k=args.k_max)
        for key, n in provenance(fused[:PROV_K],
                                 {"dense": dense_hits, "bm25": sparse_hits}).items():
            prov_total[key] += n
        return fused

    rows = []
    drift = 0
    for i, q in enumerate(questions, 1):
        if not is_scorable(q):
            rows.append(evaluate_question(q, []))
            continue
        hits = get_hits(q.question)
        retrieved = [Retrieved(h.arxiv_id, h.char_start, h.char_end, h.score) for h in hits]
        if known_spans:
            for h in retrieved:
                key = (h.arxiv_id, h.char_start, h.char_end)
                if h.arxiv_id in {s.arxiv_id for s in spans_from_question(q)} \
                        and key not in known_spans:
                    drift += 1
        rows.append(evaluate_question(q, retrieved, n_relevant_total=n_relevant.get(q.id)))
        if i % 10 == 0:
            print(f"  scored {i}/{len(questions)}")

    if drift:
        print(f"\n  WARNING: {drift} retrieved chunk(s) from evidence papers do not match "
              f"rebuilt chunks.\n  The index and this config disagree about chunking; "
              f"nDCG denominators are unreliable.\n")

    df = pd.DataFrame(rows)
    out_path = args.out or (TABLES_DIR / f"retrieval_{args.tag}.csv")
    df.to_csv(out_path, index=False)

    # --- report ------------------------------------------------------------
    ks = DEFAULT_KS
    overall = aggregate(rows, ks)
    print(f"\n{'=' * 74}\nRETRIEVAL  ({overall['n_scored']} scored questions, "
          f"{args.tag}, mode={mode})\n{'=' * 74}")
    header = f"  {'':<14}" + "".join(f"{'@' + str(k):>9}" for k in ks)
    for name in ("recall", "coverage", "ndcg"):
        if name == "recall":
            print(header)
        vals = "".join(f"{overall[f'{name}@{k}']:>9.3f}" for k in ks)
        print(f"  {name:<14}{vals}")
    print(f"  {'MRR':<14}{overall['mrr']:>9.3f}")

    print(f"\n{'-' * 74}\nAT EQUAL CHARACTER BUDGET\n{'-' * 74}")
    print("  Fixed k is not comparable across chunkings: smaller chunks mean")
    print("  more of them, so top-k delivers less text. This view fixes the")
    print("  context budget instead, which is what a generator is limited by.")
    print(f"\n  {'budget':>9}{'chunks':>9}{'recall':>9}{'coverage':>10}")
    for b in DEFAULT_BUDGETS:
        print(f"  {b:>9,}{overall[f'k@B{b}']:>9.1f}"
              f"{overall[f'recall@B{b}']:>9.3f}{overall[f'coverage@B{b}']:>10.3f}")

    print(f"\n{'-' * 74}\nBY QUESTION TYPE\n{'-' * 74}")
    print(f"  {'type':<14}{'n':>4}{'R@5':>8}{'R@10':>8}{'Cov@5':>8}{'Cov@10':>8}"
          f"{'nDCG@10':>9}{'MRR':>8}")
    for qtype, agg in aggregate_by_type(rows, ks).items():
        if not agg.get("n_scored"):
            print(f"  {qtype:<14}{agg['n_questions']:>4}    (no evidence - not scored)")
            continue
        print(f"  {qtype:<14}{agg['n_scored']:>4}{agg['recall@5']:>8.3f}"
              f"{agg['recall@10']:>8.3f}{agg['coverage@5']:>8.3f}"
              f"{agg['coverage@10']:>8.3f}{agg['ndcg@10']:>9.3f}{agg['mrr']:>8.3f}")

    # --- is the fusion doing anything, or just reordering dense? -----------
    if mode == "hybrid" and prov_total:
        total = sum(prov_total.values())
        print(f"\n{'-' * 74}\nWHERE THE TOP {PROV_K} CAME FROM\n{'-' * 74}")
        for key in ("dense only", "bm25 only", "both"):
            n = prov_total.get(key, 0)
            print(f"  {key:<14}{n:>6}{100 * n / total:>8.1f}%")
        print("\n  'bm25 only' near zero means the fusion is reordering documents")
        print("  dense already had, and any gain is reranking rather than recall.")
        print("  RRF at k=60 with a deep pool drives this to exactly zero: see")
        print("  the arithmetic in rageval/retrieve/fusion.py.")

    # --- which half of the chain was found ---------------------------------
    for qtype in ("multi_hop", "comparative"):
        if not any(r.get("question_type") == qtype and r.get("n_evidence") == 2
                   for r in rows):
            continue
        print(f"\n{'-' * 74}\nSPAN COMPLETION - {qtype}\n{'-' * 74}")
        if qtype == "multi_hop":
            print("  span 0 = source paper (the referring sentence)")
            print("  span 1 = terminal paper (the answer)")
        else:
            print("  span order is paper A / paper B and carries no direction")
        for k in (5, 10, 20):
            tally = chain_tally(rows, k, qtype)
            n = sum(len(v) for v in tally.values())
            parts = "  ".join(f"{key}: {len(ids):>2}" for key, ids in tally.items())
            print(f"  k={k:<3} {parts}   (n={n})")
        tally = chain_tally(rows, 20, qtype)
        for key in ("first only", "second only"):
            if tally[key]:
                print(f"    {key:<12} {', '.join(tally[key])}")

    # --- the questions nothing was found for -------------------------------
    misses = [r for r in rows if r.get("scorable") and r.get(f"recall@{max(ks)}") == 0.0]
    if misses:
        print(f"\n{'-' * 74}\nCOMPLETE MISSES AT k={max(ks)}  ({len(misses)})\n{'-' * 74}")
        by_type = defaultdict(list)
        for r in misses:
            by_type[r["question_type"]].append(r["id"])
        for qtype, ids in by_type.items():
            print(f"  {qtype:<14} {', '.join(ids)}")
        print("\n  Read these by hand. A miss is either a retrieval failure worth "
              "reporting\n  or a question whose evidence was badly chosen - and the "
              "aggregate cannot\n  tell you which.")

        # The pre-registered check for this sweep, evaluated automatically so
        # it cannot be quietly forgotten once the aggregate looks good.
        predicted = {"f013", "f015", "m010", "c002"}
        still = predicted & {r["id"] for r in misses}
        if mode != "dense":
            print(f"\n  PRE-REGISTERED (sweep 2): lexical retrieval should recover")
            print(f"  f013, f015, m010, c002. Still missing: "
                  f"{', '.join(sorted(still)) if still else 'none'}"
                  f"  ({len(predicted) - len(still)}/4 recovered)")

    print(f"\n  per-question results -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())