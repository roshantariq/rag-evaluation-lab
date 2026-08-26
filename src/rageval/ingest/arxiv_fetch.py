"""Fetch a topic-scoped corpus of arXiv papers and record a reproducible manifest."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import arxiv
import requests
import yaml

logger = logging.getLogger(__name__)

# arXiv asks that automated clients identify themselves. Put your own
# contact address here - it is the difference between being rate-limited
# and being blocked.
USER_AGENT = "rag-evaluation-lab/0.1 (research corpus builder; roshantariq2001@gmail.com)"


@dataclass
class PaperRecord:
    """One row of the corpus manifest."""

    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    updated: str
    primary_category: str
    categories: list[str]
    abstract: str
    pdf_url: str
    matched_query: str
    pdf_filename: str | None = None
    download_ok: bool = False
    download_error: str | None = None


def load_corpus_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["corpus"]


def _category_filter(categories: list[str]) -> str:
    return "(" + " OR ".join(f"cat:{c}" for c in categories) + ")"


def _to_record(result: arxiv.Result, matched_query: str) -> PaperRecord:
    return PaperRecord(
        arxiv_id=result.get_short_id(),
        title=" ".join(result.title.split()),
        authors=[a.name for a in result.authors],
        published=result.published.isoformat() if result.published else "",
        updated=result.updated.isoformat() if result.updated else "",
        primary_category=result.primary_category,
        categories=list(result.categories),
        abstract=" ".join(result.summary.split()),
        pdf_url=result.pdf_url or "",
        matched_query=matched_query,
    )


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

def _fetch_query(client: arxiv.Client, full_query: str, max_results: int) -> list[arxiv.Result]:
    """Run one query, backing off exponentially when arXiv throttles.

    arXiv returns 429 when rate-limiting and 503 when busy. Both mean
    'wait longer', so retrying immediately extends the block rather than
    clearing it.
    """
    waits = [0, 20, 60, 150]
    last_exc: Exception | None = None

    for attempt, wait in enumerate(waits, 1):
        if wait:
            jittered = wait + random.uniform(0, 5)
            logger.info("  throttled - waiting %.0fs before attempt %d", jittered, attempt)
            time.sleep(jittered)
        try:
            search = arxiv.Search(
                query=full_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            return list(client.results(search))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("  attempt %d failed: %s", attempt, exc)

    raise RuntimeError(f"Query failed after {len(waits)} attempts") from last_exc


def _balanced_selection(by_query: dict[str, list[PaperRecord]], target: int) -> list[PaperRecord]:
    """Round-robin across queries so no single query dominates the corpus.

    Taking the first N in discovery order would let early queries fill the
    quota and silently exclude entire subtopics.
    """
    selected: list[PaperRecord] = []
    rank = 0
    while len(selected) < target:
        added = False
        for recs in by_query.values():
            if rank < len(recs):
                selected.append(recs[rank])
                added = True
                if len(selected) >= target:
                    break
        if not added:
            break
        rank += 1
    return selected


def search_corpus(config: dict, client: arxiv.Client | None = None) -> list[PaperRecord]:
    """Run every configured query, de-duplicate globally, then select round-robin."""
    # num_retries=1: our own backoff owns retry timing, so the library must
    # not retry underneath us and compound the rate limiting.
    client = client or arxiv.Client(page_size=50, delay_seconds=5.0, num_retries=1)
    cat_clause = _category_filter(config["categories"])
    earliest = config.get("earliest_year")
    pause = config.get("inter_query_pause", 8.0)

    seen: set[str] = set()
    by_query: dict[str, list[PaperRecord]] = {}

    for i, query in enumerate(config["queries"]):
        if i:
            time.sleep(pause)

        logger.info("Query: %s", query)
        by_query[query] = []
        try:
            results = _fetch_query(client, f"{query} AND {cat_clause}", config["max_per_query"])
        except Exception as exc:  # noqa: BLE001 - one dead query must not kill the run
            logger.warning("  ABANDONED: %s", exc)
            continue

        for result in results:
            key = result.get_short_id().split("v")[0]
            if key in seen:
                continue
            if earliest and result.published and result.published.year < earliest:
                continue
            seen.add(key)
            by_query[query].append(_to_record(result, query))

        logger.info("  -> %d unique, %d in pool", len(by_query[query]), len(seen))

    empty = [q for q, r in by_query.items() if not r]
    if empty:
        logger.warning("")
        logger.warning("%d QUERIES RETURNED NOTHING:", len(empty))
        for q in empty:
            logger.warning("   - %s", q)
        logger.warning("Re-run before trusting this corpus.")

    return _balanced_selection(by_query, config["target_size"])


# --------------------------------------------------------------------------
# Download
#
# arxiv>=4.0 removed Result.download_pdf; the package is a metadata client
# only. Fetching pdf_url directly is also strictly better than the old API:
# it makes one HTTP request per paper instead of re-querying the API for
# metadata we already hold.
# --------------------------------------------------------------------------

def _download_one(
    session: requests.Session, url: str, target: Path, timeout: int
) -> tuple[bool, str | None]:
    """Fetch one PDF with backoff. Returns (ok, error_reason)."""
    waits = [0, 15, 45, 120]
    last_error: str | None = None

    for attempt, wait in enumerate(waits, 1):
        if wait:
            jittered = wait + random.uniform(0, 5)
            logger.info("    throttled - waiting %.0fs before attempt %d", jittered, attempt)
            time.sleep(jittered)
        try:
            resp = session.get(url, timeout=timeout, allow_redirects=True)

            # Transient: worth retrying.
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                continue

            resp.raise_for_status()
            body = resp.content

            # arXiv serves an HTML slow-down page with status 200 under load.
            # Without this check it lands on disk as a .pdf and silently
            # corrupts the extraction phase.
            if not body.startswith(b"%PDF"):
                return False, f"not a PDF (starts with {body[:12]!r})"

            target.write_bytes(body)
            if target.stat().st_size < 2048:
                target.unlink(missing_ok=True)
                return False, f"suspiciously small ({len(body)} bytes)"
            return True, None

        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"

    return False, last_error


def download_pdfs(
    records: list[PaperRecord],
    out_dir: Path,
    pause: float = 3.0,
    timeout: int = 60,
) -> list[PaperRecord]:
    """Download each PDF over HTTP, recording failures rather than raising."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for i, rec in enumerate(records, 1):
        filename = f"{rec.arxiv_id.replace('/', '_')}.pdf"
        target = out_dir / filename
        rec.pdf_filename = filename

        if target.exists() and target.stat().st_size > 2048:
            rec.download_ok = True
            logger.info("[%d/%d] cached   %s", i, len(records), rec.arxiv_id)
            continue

        url = rec.pdf_url or f"https://arxiv.org/pdf/{rec.arxiv_id}"
        ok, err = _download_one(session, url, target, timeout)
        rec.download_ok = ok
        rec.download_error = err

        if ok:
            kb = target.stat().st_size // 1024
            logger.info("[%d/%d] ok       %s (%d KB)", i, len(records), rec.arxiv_id, kb)
        else:
            logger.warning("[%d/%d] FAILED   %s - %s", i, len(records), rec.arxiv_id, err)

        time.sleep(pause)

    return records


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def write_manifest(records: list[PaperRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    logger.info("Manifest written: %s (%d records)", path, len(records))


def read_manifest(path: Path) -> list[PaperRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(PaperRecord(**json.loads(line)))
    return records