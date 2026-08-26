"""Find the character span of a quoted passage. The authoring workhorse.

Copy a sentence out of search results, run it through here, and get the
(arxiv_id, char_start, char_end, quote) block ready to paste into the gold
set - rather than counting characters by hand.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from rageval.config import INTERIM_DIR


def load_text(arxiv_id: str) -> tuple[str, str]:
    path = INTERIM_DIR / f"{arxiv_id.replace('/', '_')}.json"
    if not path.exists():
        matches = list(INTERIM_DIR.glob(f"{arxiv_id.split('v')[0]}*.json"))
        if not matches:
            raise SystemExit(f"No extracted text for {arxiv_id}. Check the ID.")
        path = matches[0]
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc["text"], doc.get("title", "")


def find_spans(text: str, quote: str) -> list[tuple[int, int]]:
    """Locate a quote tolerantly: whitespace runs match any whitespace."""
    words = quote.split()
    if not words:
        return []
    pattern = r"\s+".join(re.escape(w) for w in words)
    return [m.span() for m in re.finditer(pattern, text, re.IGNORECASE)]


def snap_to_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen a span outward to whole-word boundaries."""
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return start, end


def snap_to_sentences(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen a span outward to complete sentences.

    Gold-set evidence is read by a human months later to verify an answer.
    A span that begins mid-word is technically valid and practically
    useless for that.
    """
    i = text.rfind(". ", 0, start)
    start = i + 2 if i != -1 else 0
    j = text.find(". ", end)
    end = j + 1 if j != -1 else len(text)
    return start, end


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper", required=True, help="arXiv ID, e.g. 1506.04214v2")
    p.add_argument("--quote", required=True, help="A phrase from the passage.")
    p.add_argument("--context", type=int, default=200, help="Chars of surrounding text to show.")
    p.add_argument("--extend", type=int, default=0,
                   help="Extend the span by N chars each side before snapping.")
    p.add_argument("--snap", default="sentence", choices=["sentence", "word", "none"],
                   help="Widen the span to clean boundaries. Default: sentence.")
    args = p.parse_args()

    text, title = load_text(args.paper)
    spans = find_spans(text, args.quote)

    if not spans:
        print(f"Quote not found in {args.paper}.")
        print("The extracted text collapses line breaks - try a shorter phrase,")
        print("and check you are quoting the extraction rather than the PDF.")
        return 1

    print(f"\n{args.paper}  {title[:64]}")
    print(f"document length: {len(text)} chars")
    print(f"{len(spans)} match(es)\n")

    for i, (start, end) in enumerate(spans, 1):
        if args.extend:
            start = max(0, start - args.extend)
            end = min(len(text), end + args.extend)
        if args.snap == "sentence":
            start, end = snap_to_sentences(text, start, end)
        elif args.snap == "word":
            start, end = snap_to_words(text, start, end)
        before = " ".join(text[max(0, start - args.context):start].split())
        span_text = text[start:end]
        after = " ".join(text[end:end + args.context].split())

        print(f"--- match {i}: chars {start}-{end} " + "-" * 40)
        if before:
            print(f"...{before[-args.context:]}")
        print(f">>> {' '.join(span_text.split())}")
        if after:
            print(f"{after[:args.context]}...")

        block = {
            "arxiv_id": args.paper,
            "char_start": start,
            "char_end": end,
            "quote": " ".join(span_text.split()),
        }
        print(f"\nevidence block:\n{json.dumps(block, ensure_ascii=False)}\n")

    if len(spans) > 1:
        print("Multiple matches - pick the one whose surrounding text actually")
        print("supports your answer, not just the first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())