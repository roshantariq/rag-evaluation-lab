"""Chunk the extracted corpus, embed it, and build the vector index."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter

import yaml

from rageval.chunking.base import chunk_document, get_token_counter
from rageval.config import CHROMA_DIR, CONFIG_DIR, EMBED_CACHE, INTERIM_DIR, ensure_dirs
from rageval.embed.encoder import Encoder
from rageval.store.chroma_store import ChromaStore


def load_documents(skip_degraded: bool = False) -> list[dict]:
    docs = []
    for path in sorted(INTERIM_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        if skip_degraded and doc.get("status") != "clean":
            continue
        docs.append(doc)
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-degraded", action="store_true",
                        help="Index only papers that extracted cleanly.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    ensure_dirs()

    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    with open(CONFIG_DIR / args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    strategy = cfg["chunking"]["strategy"]
    target = cfg["chunking"]["target_tokens"]
    overlap = cfg["chunking"].get("overlap_tokens", 0)
    model = cfg["embedding"]["model"]
    collection = cfg["index"]["collection"]

    docs = load_documents(args.skip_degraded)
    if args.limit:
        docs = docs[: args.limit]
    if not docs:
        print("No extracted documents found. Run scripts/02_extract_text.py first.")
        return 1

    print(f"Documents      {len(docs)}")
    print(f"Chunking       {strategy}  target={target} overlap={overlap}")
    print(f"Embedding      {model}")
    print(f"Collection     {collection}\n")

    count_tokens = get_token_counter()
    t0 = time.perf_counter()
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, strategy, target, overlap, count_tokens))
    print(f"Chunked into {len(chunks)} pieces in {time.perf_counter() - t0:.1f}s")

    sizes = sorted(c.char_end - c.char_start for c in chunks)
    print(f"  chars per chunk: median {sizes[len(sizes)//2]}, "
          f"p10 {sizes[len(sizes)//10]}, p90 {sizes[9*len(sizes)//10]}")
    per_doc = Counter(c.arxiv_id for c in chunks)
    print(f"  chunks per paper: median {sorted(per_doc.values())[len(per_doc)//2]}, "
          f"max {max(per_doc.values())}\n")

    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        print(f"FATAL: {len(ids) - len(set(ids))} duplicate chunk IDs. "
              "Retrieval metrics would be silently wrong.")
        return 1

    encoder = Encoder(model, cache_path=EMBED_CACHE)
    t0 = time.perf_counter()
    vectors = encoder.encode([c.text for c in chunks], show_progress=True)
    print(f"Embedded in {time.perf_counter() - t0:.1f}s  shape={vectors.shape}\n")

    store = ChromaStore(CHROMA_DIR, collection)
    store.reset()
    store.add_chunks(chunks, vectors)
    print(f"\nIndex built: {store.count()} chunks in collection '{collection}'")
    print(f"  location: {CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())