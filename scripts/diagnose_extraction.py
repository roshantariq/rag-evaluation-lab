"""One-off audit: what did reference stripping actually remove?

strip_references splices (body + appendix) rather than truncating, so the
removed span cannot be recovered by slicing. Verify by probing whether
post-appendix content survived into the result.
"""

from rageval.config import CORPUS_MANIFEST, RAW_DIR
from rageval.ingest.arxiv_fetch import read_manifest
from rageval.ingest.extract import ENGINES, _RESUME, normalize_text, strip_references


def flat(s: str) -> str:
    return " ".join(s.split())


records = [r for r in read_manifest(CORPUS_MANIFEST) if r.download_ok]
no_refs, appendix_lost, appendix_kept, no_appendix = [], [], [], 0

for r in records:
    raw, _ = ENGINES["pymupdf"](RAW_DIR / r.pdf_filename)
    norm = normalize_text(raw)
    body, found = strip_references(norm)
    removed = len(norm) - len(body)

    if not found:
        no_refs.append((r.arxiv_id, removed, r.title[:56]))
        continue

    marker = _RESUME.search(norm)
    if not marker:
        no_appendix += 1
        continue

    # Probe: does content from just after the appendix marker survive?
    probe = flat(norm[marker.end(): marker.end() + 400])[:100]
    if probe and probe in flat(body):
        appendix_kept.append(r.arxiv_id)
    else:
        appendix_lost.append((r.arxiv_id, removed, r.title[:42]))

print(f"\n{len(records)} papers")
print(f"  bibliography removed  {len(records) - len(no_refs)}")
print(f"  no bibliography found {len(no_refs)}")
print(f"\n  with an appendix: {len(appendix_kept) + len(appendix_lost)}"
      f"   kept {len(appendix_kept)}   LOST {len(appendix_lost)}")
print(f"  no appendix at all: {no_appendix}")

if no_refs:
    print(f"\nNo bibliography detected (these keep their reference lists as noise):")
    for aid, _, title in no_refs:
        print(f"  {aid:<16} {title}")

if appendix_lost:
    print(f"\nAppendix genuinely lost:")
    for aid, n, title in sorted(appendix_lost, key=lambda x: -x[1]):
        print(f"  {aid:<16} {n:7d} chars removed  {title}")