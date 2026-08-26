"""Verify the base64 strip: what exactly was removed, and is any left?

Two questions, neither answered by looking at a top-10 list:
  1. Did anything survive that should have been removed?
  2. Was anything removed that should have survived?

The second matters more. Deletions are invisible in every summary
statistic, and the gold set is about to anchor to this output permanently.
"""

from __future__ import annotations

import json
from collections import Counter

from rageval.config import CORPUS_MANIFEST, INTERIM_DIR, RAW_DIR, TABLES_DIR
from rageval.ingest.arxiv_fetch import read_manifest
from rageval.ingest.extract import _ENCODED_MARKER, ENGINES, _looks_encoded

# Things that would be alarming to find in the removed set.
SUSPICIOUS = ("://", "arxiv", "doi", "ERA5", "hPa")

records = [r for r in read_manifest(CORPUS_MANIFEST) if r.download_ok]

removed_by_paper: Counter = Counter()
removed_tokens: list[tuple[str, str]] = []

print(f"Re-extracting {len(records)} papers to diff against the stripped output...\n")
for r in records:
    raw, _ = ENGINES["pymupdf"](RAW_DIR / r.pdf_filename)
    for token in raw.split():
        if _looks_encoded(token):
            removed_by_paper[r.arxiv_id] += 1
            removed_tokens.append((r.arxiv_id, token))

print("=" * 74)
print(f"QUESTION 2 - what was deleted?  {len(removed_tokens)} tokens "
      f"from {len(removed_by_paper)} papers")
print("=" * 74)

if removed_by_paper:
    print("\nPer paper:")
    for aid, n in removed_by_paper.most_common():
        print(f"  {aid:<16} {n:5d} tokens")

    lens = sorted(len(t) for _, t in removed_tokens)
    print(f"\nToken lengths: min {lens[0]}, median {lens[len(lens)//2]}, max {lens[-1]}")

    print("\nEVERY DISTINCT REMOVED TOKEN (read these - this is the false-positive check):")
    seen = set()
    shown = 0
    for aid, tok in removed_tokens:
        if tok in seen:
            continue
        seen.add(tok)
        shown += 1
        if shown > 60:
            print(f"  ... and {len(set(t for _, t in removed_tokens)) - 60} more distinct tokens")
            break
        print(f"  {aid:<16} {tok[:100]}")

    flagged = [(a, t) for a, t in removed_tokens
               if any(s.lower() in t.lower() for s in SUSPICIOUS)]
    print(f"\nTokens containing URL/DOI/identifier markers: {len(flagged)}")
    if flagged:
        print("  ^^ INSPECT THESE. Legitimate identifiers should not be here.")
        for aid, tok in flagged[:20]:
            print(f"     {aid:<16} {tok[:100]}")
else:
    print("\nNothing was removed at all - which would mean the strip never fired.")

print("\n" + "=" * 74)
print("QUESTION 1 - did anything survive?")
print("=" * 74)

survivors = []
for path in sorted(INTERIM_DIR.glob("*.json")):
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    text = doc["text"]
    marker_hits = _ENCODED_MARKER.findall(text)
    blob_tokens = [t for t in text.split() if _looks_encoded(t)]
    if marker_hits or blob_tokens:
        survivors.append((doc["arxiv_id"], len(marker_hits), len(blob_tokens),
                          (blob_tokens or marker_hits)[0][:80]))

if survivors:
    print(f"\n{len(survivors)} papers still contain encoded data:")
    for aid, m, b, sample in survivors:
        print(f"  {aid:<16} markers={m:3d} blobs={b:3d}  {sample}")
else:
    print("\nClean: no encoded markers or blob tokens remain in data/interim/.")

with open(TABLES_DIR / "blob_strip_audit.csv", "w", encoding="utf-8") as fh:
    fh.write("arxiv_id,token_length,token\n")
    for aid, tok in removed_tokens:
        fh.write(f'{aid},{len(tok)},"{tok[:200]}"\n')
print(f"\nFull removal log -> {TABLES_DIR / 'blob_strip_audit.csv'}")