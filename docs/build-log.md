# Build log

Running record of decisions, defects and measured results. Phase numbering
follows the build plan. Newest phase last.

**Status:** Phases 0–2 complete. Phase 3 (baseline system) next.

---

## Phase 0 — Environment and repository

**Done.** src-layout package, editable install, MIT licence, LF line endings
enforced via `.gitattributes`.

### Defects found

**`ragas` imports a module `langchain-community` deleted.** `ragas` (all
versions through 0.4.3) imports `langchain_community.chat_models.vertexai`
at module load. That module was removed in `langchain-community` **0.4.2**;
it exists in 0.4.1 and earlier. `ragas` declares `langchain-community` with
no upper bound, so pip installs the newest and produces a broken
combination. Upgrading `ragas` does not help — 0.4.3 has the identical
import.

Behind it, a second problem: `ragas` 0.3.1 imports `PIL` without declaring
`pillow`.

**Resolution.** Pin `langchain-community==0.4.1`, add `pillow` explicitly.
Verified working on langchain 1.3.17 / langchain-core 1.6.0 / ragas 0.3.1.
Both pins carry explanatory comments in `pyproject.toml`; a
`requirements.lock.txt` captures the full resolved set.

---

## Phase 1 — Corpus acquisition

**Done.** 130 papers, 130 PDFs downloaded, manifest committed to
`data/eval/corpus_manifest.jsonl`.

### Corpus

Deep learning for weather and climate forecasting, arXiv, 2015–2026.
Nine queries across `physics.ao-ph`, `cs.LG`, `cs.CV`, `stat.ML`.
282 candidates found, 130 selected.

Year distribution: 2015 (2), 2016 (1), 2017 (2), 2018 (1), 2019 (7),
2020 (12), 2021 (5), 2022 (8), 2023 (19), 2024 (29), 2025 (25), 2026 (19).

Includes Shi et al. 2015 (the original ConvLSTM paper), GraphCast, FengWu,
FuXi, GenCast, Pangu-Weather, AIFS, MetNet, MetMamba.

### Decisions

**Topic widened** from precipitation nowcasting to weather *and climate*
forecasting. Pure nowcasting is too narrow to support comparative
questions; the wider framing puts distinct model families in the same
corpus while preserving the ConvLSTM dissertation link.

**One mild off-topic inclusion** kept: 2107.01343 (photovoltaic power
forecasting) drifted in on a spatiotemporal keyword match. Adjacent-but-
off-topic papers are useful raw material for unanswerable questions.

**Recent skew accepted.** 73 of 130 papers are from 2024 onward. Honest to
a field that expanded sharply after 2022. Consequence: comparative
questions run between contemporary approaches rather than across eras.

### Defects found

**arXiv rate limiting produced a silently partial corpus.** The first run
had six of nine queries fail with HTTP 429/503 and reported 119 papers as
if nothing were wrong. The missing queries were all the nowcasting ones —
the corpus had no ConvLSTM or radar papers at all, which was the entire
reason for choosing this topic. Root cause: the client retried immediately
(`num_retries=3`), compounding the block.

*Resolution.* Exponential backoff with jitter (20s / 60s / 150s),
`num_retries=1` so the library does not retry underneath our own loop, an
8-second inter-query pause, and a loud end-of-run report naming any query
that returned nothing.

**Query order silently determined corpus composition.** Selection took the
first N in dict-insertion order and stopped early once the target was
reached. Once backoff made all nine queries work, queries 1–4 would have
filled the 130 and queries 5–9 would never have run — producing a pure
nowcasting corpus with no medium-range papers, invisibly.

*Resolution.* All queries always run; selection is round-robin by relevance
rank across queries. Verified offline: target 30 from pools of 40/40/5
yields 13/12/5, and the exhaustion case terminates rather than looping.

**`arxiv` 4.x removed downloading.** `Result.download_pdf()` no longer
exists; the package is a metadata client now. The original code also made
one API call per paper to re-fetch metadata already held, which is what
triggered the 429 storm during download.

*Resolution.* Fetch `pdf_url` directly over HTTP. Includes a `%PDF`
magic-bytes check — under load arXiv serves an HTML "slow down" page with
status 200, which would otherwise land on disk as a `.pdf` and corrupt
extraction silently.

---

## Phase 2 — Extraction and failure audit

**Done.** 130 papers extracted to `data/interim/`, audit committed to
`results/tables/extraction_audit.csv`.

### Results

| Status | Count | Share |
|---|---|---|
| clean | 123 | 94.6% |
| degraded | 7 | 5.4% |
| failed | 0 | 0% |

- Median 2,296 characters per page after bibliography removal
- Bibliographies removed from 126 / 130
- Appendices preserved: 52 / 52
- PyMuPDF vs pdfplumber: **+2.8% text, 32× faster** (0.04s vs 1.27s per
  paper, median over 15). PyMuPDF is the default.

### Defects found

**Pipeline ordering destroyed document structure.** `clean_text` collapsed
single newlines into spaces before `split_sections` and `strip_references`
ran. Both identify their targets as whole lines (`^heading$`), so after
collapsing, `References` reads as `References Bouallègue Zied Ben, ...` and
never matches. Every paper reported one section, no bibliography, and
`classify`'s `n_sections < 2` rule marked **100% of the corpus degraded**
from that single cause.

*Resolution.* Split into `normalize_text` (line-preserving) and
`collapse_paragraphs` (run per section, after structural parsing). Also
changed heading anchors from `\s*` to `[ \t]*` — in multiline mode `\s`
consumes newlines and lets `^`/`$` drift across line boundaries.

**Truncating at References deleted every appendix.** The fix above produced
94.6% clean, which looked finished. It was also cutting from the References
heading to end-of-document: 20 papers lost content, the worst 38,187
characters, including the appendix of Shi et al. 2015 — the single most
relevant paper in the corpus. Invisible in every summary statistic.

*Resolution.* Excise only the bibliography block and resume at the
appendix marker.

**A percentage guard was the wrong instrument.** The original rule rejected
any References heading in the first 40% of a document as a false positive.
In appendix-heavy papers (MetMamba: 45 pages, 36 of them figures) the real
bibliography sits at ~20%, so six papers were rejected wrongly. What
distinguishes a real bibliography is how much *body text* precedes it — an
absolute quantity, not a ratio.

*Resolution.* When a resumption marker follows, require 3,000 characters of
body before the heading. The 40% rule still applies when the bibliography
runs to end-of-document.

**The diagnostic script measured the wrong thing after the splice change.**
`strip_references` stopped returning a prefix of its input, but the
diagnostic still computed the removed span as `norm[len(body):]`. After a
splice that slice is an arbitrary tail, frequently containing the word
"Appendix" — so it reported 15 papers as losing appendices when the true
number was zero.

*Resolution.* Probe whether post-appendix content survives into the result
rather than reconstructing what was removed.

### Known limitations

- 4 papers keep their reference lists (ClimateSet, MetMamba, 2603.16976,
  1506.08768). Their bibliography heading shares a line with the first
  citation and no `Appendix` marker follows. Detecting lettered appendix
  headings (`A Data Details`) needs a pattern loose enough to match
  ordinary two-word lines — judged not worth the false positives.
- The 7 degraded papers are figure-dense, not badly extracted. Sparsest is
  2308.04460 at 693 chars/page, being 38 pages of mostly plots.
- Equations fragment into punctuation. Formulae are not retrievable.
- MuPDF colour-space warnings on stderr are suppressed; they do not
  correlate with extraction quality.

### Standing lesson

The first extraction run reported 100% degraded and was obviously broken.
The second reported 94.6% clean and looked finished, while silently
deleting a third of some papers. A plausible number is the one that does
not get audited.

---

## Plan amendments

**Gold-set evidence anchors to character spans, not chunk IDs.** Recorded
in the plan as chunk IDs originally. Chunk IDs are not stable across the
six chunking strategies being ablated — a passage that is chunk 47 under
fixed-512 is part of chunk 12 under section-aware — so chunk-ID ground
truth is valid for exactly one arm of the ablation and meaningless for the
rest. Evidence is recorded as `(arxiv_id, char_start, char_end)` against
the extracted text; a chunk is relevant if its span overlaps.

**Consequence: `data/interim/` is now frozen.** Those offsets are only
meaningful against this exact extractor output. Any later change to
extraction invalidates every span in the gold set.

---

## Open items

- 4 papers retain reference lists (accepted, documented)
- Quality thresholds in `classify` were set a priori and validated only
  against the observed distribution; not tuned
- Engine comparison covers 15 papers, not all 130