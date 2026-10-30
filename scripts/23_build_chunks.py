"""Build a chunk set from the frozen extraction and write its spans.

Usage:
    python scripts/23_build_chunks.py --strategy section --target 560
    python scripts/23_build_chunks.py --strategy semantic --target 512

Writes results/tables/chunks_<strategy>.csv: one row per chunk with its
character span. That table, not this script's re-run, is what the harness
scores (`10_eval_retrieval.py --chunks <strategy>`). Section and recursive
chunking are deterministic, but semantic boundaries come from GPU
embeddings, which are not bit-identical across runs or machines; a near-
tie in distance could move a cut. Committing the spans makes the chunk
set itself the reproducible artifact, as the gold set is.

Reads data/interim/ only. Never writes to it.

Checks, before anything is written:
  * span invariant: every chunk is an exact slice, trimmed to word edges
  * tiling: every word of every paper lands in exactly one chunk, in order
Prints the size distribution against fixed_512's, because the protocol
matches medians (docs/chunker-protocol.md).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys

import numpy as np
import pandas as pd

from rageval.chunking.base import chunk_document, get_token_counter, word_spans
from rageval.chunking.strategies import semantic_cuts, sentence_segments
from rageval.config import INTERIM_DIR, TABLES_DIR
from rageval.provenance import record

STRATEGIES = ("fixed", "section", "recursive", "semantic")

# --- semantic chunker, fixed in docs/chunker-protocol.md -------------------
EMBEDDER = "BAAI/bge-large-en-v1.5"
SEGMENT_MAX_TOKENS = 256      # under the embedder's 512 word-piece window
WINDOW = (0.5, 1.5)           # chunk size window, as fractions of the target


def load_docs() -> list[dict]:
    docs = []
    for path in sorted(INTERIM_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            docs.append(json.load(fh))
    return docs


def embed_segments(docs, count_tokens, model) -> dict[str, tuple]:
    """Per paper: the word index, its segments, their costs and adjacent distances.

    Embedding is the slow part and does not depend on the target, so it is
    done once and reused for every target tried.
    """
    out, too_long, n_segs = {}, 0, 0
    tok = model.tokenizer
    for d in docs:
        idx, segs = sentence_segments(d["text"], SEGMENT_MAX_TOKENS, count_tokens)
        if not segs:
            out[d["arxiv_id"]] = (idx, [], [], [])
            continue
        texts = [d["text"][slice(*idx.char_span(a, b))] for a, b in segs]
        too_long += sum(len(tok(t)["input_ids"]) > model.max_seq_length for t in texts)
        n_segs += len(texts)
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
        dist = [float(1.0 - np.dot(emb[k], emb[k + 1])) for k in range(len(emb) - 1)]
        out[d["arxiv_id"]] = (idx, segs, [idx.cost(a, b) for a, b in segs], dist)
    print(f"  segments embedded: {n_segs}; longer than the embedder's window: {too_long}")
    return out


def semantic_spans(embedded: dict, target: int) -> dict[str, list[tuple[int, int]]]:
    lo, hi = int(target * WINDOW[0]), int(target * WINDOW[1])
    out = {}
    for arxiv_id, (idx, segs, costs, dist) in embedded.items():
        groups = semantic_cuts(costs, dist, lo, hi) if segs else []
        out[arxiv_id] = [idx.char_span(segs[g0][0], segs[g1 - 1][1]) for g0, g1 in groups]
    return out


def check(doc: dict, spans: list[tuple[int, int]]) -> str | None:
    """None if the spans tile the document exactly, else what is wrong."""
    text = doc["text"]
    words = word_spans(text)
    wi = 0
    for a, b in spans:
        if wi >= len(words) or words[wi][0] != a:
            return f"chunk {a}-{b} does not start at the next uncovered word"
        while wi < len(words) and words[wi][1] <= b:
            wi += 1
        if words[wi - 1][1] != b:
            return f"chunk {a}-{b} does not end at a word edge"
    if wi != len(words):
        return f"{len(words) - wi} of {len(words)} words not covered"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--target", type=int, nargs="+", required=True,
                        help="Target size in tokens. Several only with --dry-run.")
    parser.add_argument("--out", default=None, help="Default results/tables/chunks_<strategy>.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the size distribution without writing (for calibration).")
    args = parser.parse_args()
    if len(args.target) > 1 and not args.dry_run:
        print("Several --target values are for calibration; add --dry-run.")
        return 1

    docs = load_docs()
    count_tokens = get_token_counter()
    print(f"{len(docs)} papers from {INTERIM_DIR}")

    embedded = None
    if args.strategy == "semantic":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDER)
        print(f"  embedder {EMBEDDER} on {model.device}, max_seq_length {model.max_seq_length}")
        embedded = embed_segments(docs, count_tokens, model)

    ref = [c.char_end - c.char_start for d in docs
           for c in chunk_document(d, "fixed_512", 512, 0, count_tokens)]
    for target in args.target:
        code = build(args, docs, target, count_tokens, embedded, ref)
        if code:
            return code
    return 0


def build(args, docs, target, count_tokens, embedded, ref) -> int:
    if embedded is not None:
        by_paper = semantic_spans(embedded, target)
    else:
        by_paper = {d["arxiv_id"]: [(c.char_start, c.char_end) for c in chunk_document(
            d, f"{args.strategy}_{target}", target, 0, count_tokens)] for d in docs}

    rows = []
    for d in docs:
        spans = by_paper[d["arxiv_id"]]
        problem = check(d, spans)
        if problem:
            print(f"  TILING FAILED for {d['arxiv_id']}: {problem}. Nothing written.")
            return 1
        for a, b in spans:
            rows.append({"chunk_id": f"{d['arxiv_id']}:{a}-{b}", "arxiv_id": d["arxiv_id"],
                         "char_start": a, "char_end": b, "chars": b - a})
    df = pd.DataFrame(rows)

    q = df["chars"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"\n  {args.strategy} target {target}: {len(df)} chunks, tiling and spans verified")
    print(f"  chars  p5 {q[0.05]:.0f}  p25 {q[0.25]:.0f}  median {q[0.5]:.0f}  "
          f"p75 {q[0.75]:.0f}  p95 {q[0.95]:.0f}  max {df['chars'].max()}")
    print(f"  fixed_512 reference: {len(ref)} chunks, median {st.median(ref):.0f} chars "
          f"-> median ratio {q[0.5] / st.median(ref):.3f}")

    if args.dry_run:
        return 0
    out = args.out or (TABLES_DIR / f"chunks_{args.strategy}.csv")
    df.drop(columns="chars").to_csv(out, index=False)
    record([out])
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
