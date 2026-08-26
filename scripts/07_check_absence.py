"""Verify that an unanswerable question really is unanswerable.

The headline finding depends on questions whose answers are genuinely
absent. Assuming absence is how a question silently becomes answerable and
the hallucination rate comes out wrong in the other direction.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from rageval.config import INTERIM_DIR


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("terms", nargs="+", help="Terms that would appear if the fact were present.")
    p.add_argument("--window", type=int, default=400,
                   help="Chars within which terms must co-occur to count as a hit.")
    p.add_argument("--context", type=int, default=160)
    p.add_argument("--max-hits", type=int, default=25)
    args = p.parse_args()

    patterns = [re.compile(re.escape(t).replace(r"\ ", r"\s+"), re.IGNORECASE) for t in args.terms]
    hits, docs_scanned = [], 0

    for path in sorted(INTERIM_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        docs_scanned += 1
        text = doc["text"]

        anchors = [m.start() for m in patterns[0].finditer(text)]
        for pos in anchors:
            lo, hi = max(0, pos - args.window), min(len(text), pos + args.window)
            window = text[lo:hi]
            if all(pat.search(window) for pat in patterns[1:]):
                hits.append((doc["arxiv_id"], doc.get("title", "")[:44], pos,
                             " ".join(text[max(0, pos - args.context):pos + args.context].split())))

    print(f"\nTerms: {args.terms}")
    print(f"Scanned {docs_scanned} documents, co-occurrence window {args.window} chars\n")

    if not hits:
        print("NO CO-OCCURRENCES FOUND.")
        print("Consistent with the question being unanswerable, but not proof -")
        print("try synonyms and alternative phrasings before accepting it.")
        return 0

    print(f"{len(hits)} co-occurrence(s) across {len({h[0] for h in hits})} papers.")
    print("READ THESE. If any actually answers the question, it is not unanswerable.\n")
    for aid, title, pos, snippet in hits[:args.max_hits]:
        print(f"  {aid:<16} char {pos:<7} {title}")
        print(f"    ...{snippet}...\n")
    if len(hits) > args.max_hits:
        print(f"  ... and {len(hits) - args.max_hits} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())