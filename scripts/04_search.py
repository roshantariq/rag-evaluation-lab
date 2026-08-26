"""Search the index from the command line. Retrieval only - no LLM, no API key."""

from __future__ import annotations

import argparse
import logging
import sys

import yaml

from rageval.config import CHROMA_DIR, CONFIG_DIR, EMBED_CACHE
from rageval.embed.encoder import Encoder
from rageval.store.chroma_store import ChromaStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+", help="Your question, unquoted is fine.")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--config", default="baseline.yaml")
    parser.add_argument("--full", action="store_true", help="Print whole chunks.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)

    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    with open(CONFIG_DIR / args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    store = ChromaStore(CHROMA_DIR, cfg["index"]["collection"])
    if store.count() == 0:
        print("Index is empty. Run scripts/03_build_index.py first.")
        return 1

    query = " ".join(args.query)
    encoder = Encoder(cfg["embedding"]["model"], cache_path=EMBED_CACHE)
    hits = store.query(encoder.encode_query(query), k=args.k)

    print(f"\nQuery: {query}\n" + "=" * 78)
    for h in hits:
        print(f"\n[{h.rank}] {h.score:.3f}  {h.arxiv_id}  chars {h.char_start}-{h.char_end}")
        print(f"     {h.title[:70]}")
        if h.section and h.section != "(unknown)":
            print(f"     section: {h.section[:70]}")
        body = h.text if args.full else " ".join(h.text.split())[:300] + "..."
        print(f"     {body}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())