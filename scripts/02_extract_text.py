"""Extract text from the downloaded corpus and audit extraction quality."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict

import pandas as pd

from rageval.config import (
    CORPUS_MANIFEST,
    INTERIM_DIR,
    RAW_DIR,
    TABLES_DIR,
    ensure_dirs,
)
from rageval.ingest.arxiv_fetch import read_manifest
from rageval.ingest.extract import compare_engines, extract_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="pymupdf", choices=["pymupdf", "pdfplumber"])
    parser.add_argument("--compare", type=int, default=15,
                        help="Run both engines on this many papers for the comparison table.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    ensure_dirs()

    records = read_manifest(CORPUS_MANIFEST)
    records = [r for r in records if r.download_ok]
    if args.limit:
        records = records[: args.limit]
    print(f"Extracting {len(records)} papers with {args.engine}\n")

    audit_rows = []
    for i, rec in enumerate(records, 1):
        pdf_path = RAW_DIR / rec.pdf_filename
        doc = extract_document(pdf_path, rec.arxiv_id, engine=args.engine)

        out = INTERIM_DIR / f"{rec.arxiv_id.replace('/', '_')}.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "arxiv_id": rec.arxiv_id,
                    "title": rec.title,
                    "published": rec.published,
                    "categories": rec.categories,
                    "status": doc.status,
                    "n_pages": doc.n_pages,
                    "references_stripped": doc.references_found,
                    "sections": [asdict(s) for s in doc.sections],
                    "text": doc.text,
                },
                fh,
                ensure_ascii=False,
            )

        row = doc.to_manifest_row()
        row.update(doc.quality)
        row.pop("quality", None)
        row["title"] = rec.title
        audit_rows.append(row)

        flag = {"clean": "  ", "degraded": " ~", "failed": " X"}[doc.status]
        print(f"[{i:3d}/{len(records)}]{flag} {rec.arxiv_id:<16} "
              f"{doc.clean_chars:7d} chars  {doc.n_sections:2d} sections  {doc.status}")

    audit = pd.DataFrame(audit_rows)
    audit_path = TABLES_DIR / "extraction_audit.csv"
    audit.to_csv(audit_path, index=False)

    counts = Counter(audit["status"])
    total = len(audit)
    print(f"\n{'=' * 64}\nEXTRACTION AUDIT  ({total} papers)\n{'=' * 64}")
    for status in ("clean", "degraded", "failed"):
        n = counts.get(status, 0)
        print(f"  {status:<10} {n:4d}   {100 * n / max(total, 1):5.1f}%")
    print(f"\n  median chars/page  {audit['chars_per_page'].median():.0f}")
    print(f"  references stripped {int(audit['references_found'].sum())}/{total}")
    print(f"  audit table -> {audit_path}")

    worst = audit.nsmallest(5, "chars_per_page")[["arxiv_id", "chars_per_page", "status", "title"]]
    print(f"\n  Five sparsest extractions (inspect these by hand):")
    for _, r in worst.iterrows():
        print(f"    {r['arxiv_id']:<16} {r['chars_per_page']:7.0f}/page  {r['title'][:48]}")

    if args.compare:
        print(f"\n{'=' * 64}\nENGINE COMPARISON  (first {args.compare} papers)\n{'=' * 64}")
        comp_rows = [
            compare_engines(RAW_DIR / r.pdf_filename, r.arxiv_id)
            for r in records[: args.compare]
        ]
        comp = pd.DataFrame(comp_rows)
        comp_path = TABLES_DIR / "engine_comparison.csv"
        comp.to_csv(comp_path, index=False)
        print(f"  pymupdf    median {comp['pymupdf_chars'].median():8.0f} chars  "
              f"{comp['pymupdf_seconds'].median():5.2f}s")
        print(f"  pdfplumber median {comp['pdfplumber_chars'].median():8.0f} chars  "
              f"{comp['pdfplumber_seconds'].median():5.2f}s")
        print(f"  median char delta  {comp['char_delta_pct'].median():+.1f}%  (pymupdf vs pdfplumber)")
        print(f"  comparison table -> {comp_path}")

    return 0 if counts.get("failed", 0) < 0.15 * total else 1


if __name__ == "__main__":
    raise SystemExit(main())