# PDF extraction

## Results (130 arXiv papers)

| Status | Count | Share |
|---|---|---|
| clean | 123 | 94.6% |
| degraded | 7 | 5.4% |
| failed | 0 | 0% |

Median 2,296 characters per page after bibliography removal.

## Engine choice

PyMuPDF extracted 2.8% more text than pdfplumber and ran 32× faster
(0.04s vs 1.27s per paper, median over 15 papers). PyMuPDF is the default.

## Two bugs worth recording

**Pipeline ordering.** The first working version collapsed single newlines
into spaces before parsing document structure. Headings and the References
marker are only identifiable as whole lines, so every paper reported one
section and no bibliography — and `classify` then marked 100% of the corpus
degraded from a single rule. Structure is now parsed on line-preserved text
and soft wraps are collapsed per section afterwards.

**Truncating at References.** The fix above produced 93% clean, which looked
finished. It was also deleting every appendix: 20 papers lost content,
one 38k characters, including the appendix of Shi et al. 2015 (ConvLSTM).
`strip_references` now excises only the bibliography block and resumes at
the appendix. Verified: 52 papers have appendices, 52 retained.

The second bug matters more than the first. An obviously broken number gets
investigated; a plausible one does not.

## Known limitations

- 4 papers keep their reference lists (ClimateSet, MetMamba, 2603.16976,
  1506.08768). Their bibliography heading shares a line with the first
  citation and no `Appendix` marker follows, so neither the strict nor the
  positional rule fires. Detecting lettered appendix headings (`A Data
  Details`) needs a pattern loose enough to match ordinary two-word lines.
- 7 degraded papers are figure-dense rather than badly extracted. The
  sparsest (2308.04460, 693 chars/page) is 38 pages of which most are plots.
- Equations fragment into punctuation. Formulae are not retrievable.
- MuPDF colour-space warnings on stderr are suppressed; they do not
  correlate with extraction quality.

### Encoded blob removal

16 tokens removed across 3 papers (GenCast SVG glyph data, JWT segments,
one URL query fragment). Verified by re-extracting all 130 PDFs and reading
the complete removal log; see `results/tables/blob_strip_audit.csv` and
`scripts/verify_blob_strip.py`.

One known false positive: `2008.08626v2` loses a URL query fragment
(`ots=...&sig=...`) that is not encoded data. It has no domain, so the URL
exemption cannot see it, and widening the exemption would start letting
real blobs through. Retrieval value is zero; left in place deliberately.

## Reproducibility (verified 31 October 2026, Phase 9)

The extracted text is not committed: arXiv's default licence does not let
third parties redistribute papers or their derived text. A clone
regenerates it and checks it against `data/eval/interim_sha256.csv`
(per document: the JSON file, its `text` field, and the source PDF).

A throwaway clone at `ec3dd94`, with its own venv (PyMuPDF 1.28.2),
fetched the 130 papers with `01_fetch_corpus.py --from-manifest` (130/130,
no failures) and extracted them twice into separate directories:

| comparison | files | file hash | text hash | PDF hash |
|---|---|---|---|---|
| run 1 vs run 2 (determinism) | 130 | 130 identical | 130 identical | n/a |
| run 1 vs frozen manifest | 130 | 130 identical | 130 identical | 130 identical |
| run 2 vs frozen manifest | 130 | 130 identical | 130 identical | 130 identical |

No paper differs, so none is listed here. The three possible causes of a
mismatch (arXiv serving different bytes, a PyMuPDF version change,
non-deterministic extraction) were each testable and none occurred.
Tables: `results/tables/extraction_check_{run1_vs_run2,run1_vs_frozen,run2_vs_frozen}.csv`.
