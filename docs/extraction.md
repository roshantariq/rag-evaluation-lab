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