"""Is the baseline's retrieval failure caused by encoder truncation?

all-MiniLM-L6-v2 accepts 256 word-piece tokens. The baseline chunks target
512 tokens (median 2,113 characters). So the back half of every chunk is
dropped before the vector is computed - the text is in the chunk, so
span-overlap scoring calls that chunk relevant, but those words never
influenced the embedding. The retriever is asked to find text it was never
shown.

This script tests that directly. For every evidence span it computes, using
the encoder's own tokenizer, how much of the span survives into the 256
tokens the model actually reads, then cross-tabulates that against whether
the span was retrieved.

The prediction is sharp:

    visible spans    -> mostly hit
    invisible spans  -> almost never hit

If the two rows look alike, truncation is not the cause and the hypothesis
dies here. Nothing is re-indexed and nothing is fixed until this returns:
rebuilding the index now would destroy the evidence.

Reads the hit flags from the existing retrieval CSV rather than re-querying,
so the diagnosis is scored against exactly the run being explained.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter

import pandas as pd
import yaml

from rageval.chunking.base import chunk_document
from rageval.config import (
    CONFIG_DIR,
    EVAL_DIR,
    INTERIM_DIR,
    TABLES_DIR,
    ensure_dirs,
)
from rageval.evaluation.gold import load_gold
from rageval.evaluation.retrieval_metrics import (
    Retrieved,
    is_scorable,
    spans_from_question,
)

MAX_TOKENS = 256  # all-MiniLM-L6-v2; CLS and SEP come out of this budget.


def visible_chars(text: str, tok, max_tokens: int = MAX_TOKENS) -> int:
    """How many characters of `text` survive into the encoder's window."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    budget = max_tokens - 2  # [CLS] and [SEP]
    if len(offsets) <= budget:
        return len(text)
    return offsets[budget - 1][1]


def interim_path(arxiv_id: str):
    return INTERIM_DIR / f"{arxiv_id.replace('/', '_')}.json"


def rebuild_chunks(arxiv_ids, cfg) -> dict[str, list]:
    ch = cfg.get("chunking", {})
    out: dict[str, list] = {}
    for arxiv_id in sorted(arxiv_ids):
        path = interim_path(arxiv_id)
        if not path.exists():
            logging.warning("No extracted text for %s", arxiv_id)
            continue
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        out[arxiv_id] = chunk_document(
            doc,
            strategy=ch.get("strategy", "fixed_512"),
            target_tokens=ch.get("target_tokens", 512),
            overlap_tokens=ch.get("overlap_tokens", 0),
        )
    return out


def parse_hits(cell) -> set[int]:
    if not isinstance(cell, str) or not cell:
        return set()
    return {int(x) for x in cell.split(";") if x}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline.yaml")
    parser.add_argument("--results", default=None,
                        help="Defaults to results/tables/retrieval_baseline.csv")
    parser.add_argument("--k", type=int, default=10, help="Which hit column to explain.")
    parser.add_argument("--model", default=None, help="Override tokenizer name.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
    ensure_dirs()

    with open(CONFIG_DIR / args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    model_name = args.model or cfg["embedding"]["model"]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)

    results_path = args.results or (TABLES_DIR / "retrieval_baseline.csv")
    df = pd.read_csv(results_path)
    hit_col = f"hit_spans@{args.k}"
    if hit_col not in df.columns:
        print(f"{results_path} has no {hit_col} column. Re-run "
              f"scripts/10_eval_retrieval.py with the updated metrics module.")
        return 1
    hits = {r["id"]: parse_hits(r[hit_col]) for _, r in df.iterrows()}

    questions = [q for q in load_gold(EVAL_DIR / "gold_questions.jsonl") if is_scorable(q)]
    papers = {s.arxiv_id for q in questions for s in spans_from_question(q)}
    chunks_by_paper = rebuild_chunks(papers, cfg)

    # How much of the index is being read at all.
    seen, total = 0, 0
    for chunks in chunks_by_paper.values():
        for c in chunks:
            seen += visible_chars(c.text, tok)
            total += len(c.text)
    print(f"Encoder window: {MAX_TOKENS} tokens ({model_name})")
    print(f"Chunk text reaching the encoder: {seen:,} / {total:,} chars "
          f"= {100 * seen / max(total, 1):.1f}%\n")

    rows = []
    for q in questions:
        got = hits.get(q.id, set())
        for i, span in enumerate(spans_from_question(q)):
            best_frac, best_chunk = 0.0, None
            n_containing = 0
            for c in chunks_by_paper.get(span.arxiv_id, []):
                r = Retrieved.from_chunk(c)
                if not r.overlaps(span.arxiv_id, span.char_start, span.char_end):
                    continue
                n_containing += 1
                vis = visible_chars(c.text, tok)
                s0 = max(span.char_start - c.char_start, 0)
                s1 = min(span.char_end - c.char_start, len(c.text))
                in_chunk = max(s1 - s0, 0)
                if in_chunk <= 0:
                    continue
                frac = max(0, min(s1, vis) - s0) / in_chunk
                if frac > best_frac:
                    best_frac, best_chunk = frac, c.chunk_id
            rows.append({
                "id": q.id,
                "question_type": q.question_type,
                "span_index": i,
                "arxiv_id": span.arxiv_id,
                "span_len": span.char_end - span.char_start,
                "n_containing_chunks": n_containing,
                "best_visible_fraction": round(best_frac, 3),
                "best_chunk": best_chunk or "",
                "visible": best_frac > 0.0,
                "mostly_visible": best_frac >= 0.5,
                "hit": i in got,
            })

    out = pd.DataFrame(rows)
    out_path = TABLES_DIR / f"truncation_diagnosis_k{args.k}.csv"
    out.to_csv(out_path, index=False)

    def crosstab(flag: str, label: str):
        print(f"{'=' * 74}\n{label}  (n={len(out)} evidence spans, hits at k={args.k})\n{'=' * 74}")
        print(f"  {'':<22}{'hit':>8}{'miss':>8}{'total':>8}{'hit rate':>11}")
        for value, name in ((True, "reaches encoder"), (False, "cut off")):
            sub = out[out[flag] == value]
            h = int(sub["hit"].sum())
            n = len(sub)
            rate = f"{100 * h / n:.1f}%" if n else "-"
            print(f"  {name:<22}{h:>8}{n - h:>8}{n:>8}{rate:>11}")
        print()

    crosstab("visible", "ANY part of the span inside the window")
    crosstab("mostly_visible", "AT LEAST HALF the span inside the window")

    print(f"{'-' * 74}\nVISIBLE FRACTION, hits vs misses\n{'-' * 74}")
    for name, sub in (("hit", out[out["hit"]]), ("miss", out[~out["hit"]])):
        if not len(sub):
            continue
        v = sub["best_visible_fraction"]
        print(f"  {name:<6} n={len(sub):<4} mean {v.mean():.3f}   median {v.median():.3f}   "
              f"fully visible {int((v >= 0.999).sum())}   fully cut {int((v <= 0.001).sum())}")

    print(f"\n{'-' * 74}\nBY QUESTION TYPE (span-level hit rate)\n{'-' * 74}")
    print(f"  {'type':<14}{'spans':>7}{'visible':>9}{'cut':>7}{'hit|vis':>10}{'hit|cut':>10}")
    for qtype in ("factual", "comparative", "multi_hop", "ambiguous"):
        sub = out[out["question_type"] == qtype]
        if not len(sub):
            continue
        vis, cut = sub[sub["visible"]], sub[~sub["visible"]]
        hv = f"{100 * vis['hit'].mean():.0f}%" if len(vis) else "-"
        hc = f"{100 * cut['hit'].mean():.0f}%" if len(cut) else "-"
        print(f"  {qtype:<14}{len(sub):>7}{len(vis):>9}{len(cut):>7}{hv:>10}{hc:>10}")

    misses = out[~out["hit"]].sort_values("best_visible_fraction")
    print(f"\n{'-' * 74}\nMISSED SPANS, least visible first\n{'-' * 74}")
    print(f"  {'question':<8}{'span':>5}{'paper':<16}{'vis frac':>10}{'chunks':>8}")
    for _, r in misses.head(25).iterrows():
        print(f"  {r['id']:<8}{r['span_index']:>5}  {r['arxiv_id']:<14}"
              f"{r['best_visible_fraction']:>10.3f}{r['n_containing_chunks']:>8}")

    counts = Counter(out["visible"])
    print(f"\n  {counts[False]} of {len(out)} evidence spans never reach the encoder "
          f"at all.")
    print(f"  per-span detail -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())