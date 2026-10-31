"""Fetch the arXiv corpus. Run with --dry-run first to inspect what you'd get.

--from-manifest downloads exactly the papers and versions recorded in the
committed data/eval/corpus_manifest.jsonl, without searching arXiv and
without rewriting the manifest. A search today would return a different
corpus. This is how a clone recovers the PDFs behind the frozen extraction.
Never run either mode in the working tree that holds the frozen data/
(CLAUDE.md §14).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter

from rageval.config import CONFIG_DIR, CORPUS_MANIFEST, RAW_DIR, ensure_dirs
from rageval.ingest.arxiv_fetch import (
    download_pdfs,
    load_corpus_config,
    read_manifest,
    search_corpus,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Search and report only. Downloads nothing.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Override target corpus size, for quick tests.")
    parser.add_argument("--from-manifest", action="store_true",
                        help="Download the committed manifest's papers at their recorded "
                             "versions. No search; the manifest is not rewritten.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("arxiv").setLevel(logging.WARNING)

    ensure_dirs()
    if args.from_manifest:
        records = read_manifest(CORPUS_MANIFEST)
        for r in records:
            r.download_ok, r.download_error = False, None
        print(f"Downloading {len(records)} PDFs from {CORPUS_MANIFEST.name} to {RAW_DIR} ...\n")
        records = download_pdfs(records, RAW_DIR)
        failed = [r for r in records if not r.download_ok]
        print(f"\nDownloaded {len(records) - len(failed)}/{len(records)}.")
        for r in failed:
            print(f"  FAILED {r.arxiv_id}: {r.download_error}")
        return 0 if not failed else 1

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