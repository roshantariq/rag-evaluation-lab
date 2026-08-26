"""Ask a question end to end: retrieve, generate, show citations.

Requires OPENAI_API_KEY in .env, unless the answer is already cached.
"""

from __future__ import annotations

import argparse
import logging
import sys

import yaml

from rageval.config import CHROMA_DIR, CONFIG_DIR, EMBED_CACHE, RESPONSE_CACHE
from rageval.embed.encoder import Encoder
from rageval.generate.llm import LLMClient
from rageval.generate.prompts import PROMPT_STRATEGIES
from rageval.pipeline import RAGPipeline
from rageval.store.chroma_store import ChromaStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+")
    parser.add_argument("-k", type=int, default=None)
    parser.add_argument("--strategy", default=None, choices=sorted(PROMPT_STRATEGIES))
    parser.add_argument("--config", default="baseline.yaml")
    parser.add_argument("--show-context", action="store_true",
                        help="Print the passages sent to the model.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    with open(CONFIG_DIR / args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    gen = cfg.get("generation", {})
    store = ChromaStore(CHROMA_DIR, cfg["index"]["collection"])
    if store.count() == 0:
        print("Index is empty. Run scripts/03_build_index.py first.")
        return 1

    pipe = RAGPipeline(
        store=store,
        encoder=Encoder(cfg["embedding"]["model"], cache_path=EMBED_CACHE),
        llm=LLMClient(
            provider=gen.get("provider", "openai"),
            model=gen.get("model", "gpt-4o-mini"),
            temperature=gen.get("temperature", 0.0),
            max_tokens=gen.get("max_tokens", 700),
            cache_path=RESPONSE_CACHE,
        ),
        k=args.k or cfg["retrieval"]["k"],
        strategy=args.strategy or gen.get("prompt_strategy", "abstain"),
    )

    question = " ".join(args.question)
    result = pipe.answer(question)

    print(f"\nQ: {question}")
    print(f"   strategy={result.strategy}  k={result.k}"
          f"{'  [cached]' if result.cached else ''}")
    print("=" * 78)

    if args.show_context:
        print("\nRetrieved passages:")
        for h in result.hits:
            marker = " *" if h.rank in result.citations else "  "
            print(f"{marker}[{h.rank}] {h.score:.3f} {h.arxiv_id}  {h.title[:56]}")
            print(f"      {' '.join(h.text.split())[:180]}...")
        print()

    if result.abstained:
        print("\nABSTAINED - model reported insufficient context.\n")
    else:
        print(f"\n{result.answer}\n")

    if result.citations:
        print("Sources:")
        for i in result.citations:
            h = result.hits[i - 1]
            print(f"  [{i}] {h.arxiv_id}  chars {h.char_start}-{h.char_end}")
            print(f"      {h.title[:68]}")
    elif not result.abstained:
        print("No citations given - a faithfulness problem worth noting.")

    uncited = [h.rank for h in result.hits if h.rank not in result.citations]
    print(f"\nretrieved {len(result.hits)}, cited {len(result.citations)}"
          f"{f', unused {uncited}' if uncited else ''}")
    print(f"tokens: {result.prompt_tokens} in, {result.completion_tokens} out"
          f"   retrieval {result.retrieval_s:.2f}s, generation {result.generation_s:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())