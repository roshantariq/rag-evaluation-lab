"""How much of the index is figure and axis-label noise?

Phase 2 scored quality per document. Retrieval operates on chunks, and a
clean paper can contain chunks that are pure plot annotation - which then
compete for retrieval slots against prose that actually answers questions.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import yaml

from rageval.chunking.base import chunk_document, get_token_counter
from rageval.config import CONFIG_DIR, INTERIM_DIR, TABLES_DIR

WORDLIKE = re.compile(r"^[A-Za-z][A-Za-z'-]{2,}$")


def prose_ratio(text: str) -> float:
    """Share of tokens that are ordinary words, not numbers or symbols."""
    tokens = text.split()
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if WORDLIKE.match(t.strip(".,;:()[]%")))
    return hits / len(tokens)


with open(CONFIG_DIR / "baseline.yaml", "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

count_tokens = get_token_counter()
rows = []
for path in sorted(INTERIM_DIR.glob("*.json")):
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    for c in chunk_document(doc, cfg["chunking"]["strategy"],
                            cfg["chunking"]["target_tokens"],
                            cfg["chunking"].get("overlap_tokens", 0), count_tokens):
        rows.append({
            "arxiv_id": c.arxiv_id,
            "chunk_id": c.chunk_id,
            "title": c.title[:50],
            "prose_ratio": round(prose_ratio(c.text), 3),
            "chars": c.char_end - c.char_start,
            "text": " ".join(c.text.split())[:160],
        })

df = pd.DataFrame(rows)
df.to_csv(TABLES_DIR / "chunk_quality.csv", index=False)

print(f"{len(df)} chunks from {df.arxiv_id.nunique()} papers\n")
print("prose_ratio distribution:")
for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.95):
    print(f"  p{int(q*100):<3} {df.prose_ratio.quantile(q):.3f}")

for threshold in (0.30, 0.40, 0.50, 0.60):
    n = (df.prose_ratio < threshold).sum()
    print(f"\nbelow {threshold:.2f}: {n:5d} chunks ({100*n/len(df):.1f}%)"
          f"  from {df[df.prose_ratio < threshold].arxiv_id.nunique()} papers")

print("\nWorst 10 chunks (these are competing for your top-5 slots):")
for _, r in df.nsmallest(10, "prose_ratio").iterrows():
    print(f"  {r.prose_ratio:.2f}  {r.arxiv_id:<16} {r.text[:88]}")

print("\nPapers with the most low-prose chunks:")
worst = df[df.prose_ratio < 0.40].groupby(["arxiv_id", "title"]).size()
for (aid, title), n in worst.sort_values(ascending=False).head(8).items():
    total = (df.arxiv_id == aid).sum()
    print(f"  {aid:<16} {n:3d}/{total:<4d} chunks  {title}")