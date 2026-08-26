"""Fetch the arXiv corpus. Run with --dry-run first to inspect what you'd get."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter

from rageval.config import CONFIG_DIR, CORPUS_MANIFEST, RAW_DIR, ensure_dirs
from rageval.ingest.arxiv_fetch import (
    download_pdfs,
    load_corpus_config,
    search_corpus,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Search and report only. Downloads nothing.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Override target corpus size, for quick tests.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("arxiv").setLevel(logging.WARNING)

    ensure_dirs()
    config = load_corpus_config(CONFIG_DIR / "corpus.yaml")
    if args.limit:
        config["target_size"] = args.limit

    records = search_corpus(config)
    if not records:
        print("\nNo papers found. Check your queries and your network.")
        return 1

    years = Counter(r.published[:4] for r in records)
    print(f"\n{'=' * 62}")
    print(f"Found {len(records)} unique papers")
    print(f"Years: {min(years)} - {max(years)}")
    print("Per year:", dict(sorted(years.items())))
    print(f"{'=' * 62}\nSample of 10 titles:\n")
    for r in records[:10]:
        print(f"  {r.arxiv_id:<16} {r.published[:7]}  {r.title[:66]}")

    if args.dry_run:
        print("\nDry run. Nothing downloaded. Drop --dry-run to fetch PDFs.")
        return 0

    print(f"\nDownloading {len(records)} PDFs to {RAW_DIR} ...\n")
    records = download_pdfs(records, RAW_DIR)
    write_manifest(records, CORPUS_MANIFEST)

    ok = sum(r.download_ok for r in records)
    print(f"\nDownloaded {ok}/{len(records)}. Manifest: {CORPUS_MANIFEST}")
    return 0 if ok >= 0.85 * len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())