# Build log

Running record of decisions, defects and measured results. Phase numbering
follows the build plan. Newest phase last.

**Status:** Phases 0–5 complete. Phase 6 in progress: sweeps 1 (chunk
size), 2 (retrieval function) and 2b (fusion) closed. Chunk size and fusion
are null results; BM25 replaces dense retrieval as the baseline, with its
effect size bounded by a measured authoring confound. The embedding-model
sweep is dropped under its pre-registered condition. Next: reranking, a k
sweep, and a paraphrase validity test on the confound.

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

---

## Phase 2b — Encoded blob removal (extraction freeze)

Triggered by Phase 3 retrieval results, before the gold set could anchor to
extraction output.

### What prompted it

The first real search of the index returned five results for "what is the
fraction skill score", all of them chart axis labels:

    "8 10 0.4 2 4 6 8 10 0.4 0.25 g) Skill score (ACC): t500 ..."

Chunk-level quality measurement (`scripts/diagnose_chunks.py`) showed the
index is 4.3% below 0.40 prose ratio, and that low-prose chunks fall into
four categories a single threshold cannot separate:

| Sample | Category | Valuable |
|---|---|---|
| `0.88/0.98 0.42/0.44 SW62701500` | results table | yes |
| `√ 𝑑+ 1 ≤𝑅 𝑘 √︄ 𝜎2𝑅𝑑1/2` | equations in math Unicode | yes, unretrievable |
| `Drought days 200 100 0 16 80 4` | chart axis labels | no |
| `. . . . . . . 18 1.2` | table-of-contents dot leader | no |

Only one category was actionable: base64 image data leaking from the PDF
text layer (SVG glyph definitions in GenCast, JWT segments elsewhere).
That is not content under any interpretation.

### Decision

**Removed base64 only.** The remaining categories become a measured Phase 6
ablation axis (filter off / 0.3 / 0.4 / 0.5) rather than a silent filter,
because the threshold that would catch axis labels (~0.60) also removes
15.4% of the index across 99 papers, results tables included.

### Defects found

**Vowel frequency does not detect base64.** The first detector used vowel
ratio, on the assumption that encoded data would look less word-like.
Random base64 has roughly the vowel rate of English, and PNG headers are
full of `A` runs. It caught 1 of 4 test blobs.

*Resolution.* Case-flip rate — the share of adjacent letter pairs that
change case. Real words and camelCase identifiers do not alternate every
few characters; base64 does. Combined with an alphanumeric-ratio gate
(URLs and snake_case carry separators) and explicit markers.

**Claimed the fix worked without checking it.** Reported success on the
basis that base64 entries had left a top-10 list and the chunk count had
dropped by six — then recommended committing and freezing on that basis.
Neither observation shows whether anything was deleted that should have
survived, and deletions are invisible in every summary statistic. This is
the same failure mode as the appendix bug earlier in Phase 2.

*Resolution.* `scripts/verify_blob_strip.py` re-extracts all 130 PDFs and
answers two questions separately: did anything encoded survive, and what
exactly was removed. The full removal log is written to
`results/tables/blob_strip_audit.csv` for human reading.

**The verification found two false positives.** A Google Forms URL in FuXi
(`google.com/forms/d/e/1FAIpQLSf...`) and a URL query fragment. Both
cleared the alnum gate at 0.915 and the case-flip test on their random
path segments.

*Resolution.* Explicit URL exemption. Removals went 17 → 16, the Forms
link survived, and the extraction audit stayed byte-identical (123/7/0,
references 126/130, median 2296 chars per page) — confirming the guard
changed only what it was meant to.

### Final state

- 16 tokens removed from 3 papers, every one read and accounted for
- No encoded data survives anywhere in `data/interim/`
- One known false positive retained deliberately: a domainless URL query
  fragment in 2008.08626v2, zero retrieval value; widening the exemption
  would let real blobs through

### Standing lesson

Verification belongs *before* the freeze, not after. Freezing is the point
at which mistakes stop being cheap, and "the symptom disappeared" is not
the same claim as "the fix is correct".

---

## Phase 3 — Baseline system

Chunking, embedding, vector store, dense retrieval, generation.

### Design decisions

**Chunks carry character-span provenance.** Every chunk records
`(arxiv_id, char_start, char_end)` against the extracted text, and its ID
is derived from that span. Boundaries come from real word positions rather
than token-decode arithmetic, so `text[char_start:char_end] == chunk.text`
holds by construction. Verified across 300 randomised documents × 6
chunker settings, zero failures.

**tiktoken is optional.** It downloads its encoding on first use, so an
offline fresh clone would fail at chunking. A word-count estimate is the
fallback; chunk sizes shift slightly, spans stay exact.

**Embeddings supplied to Chroma, never computed by it.** Keeps one code
path for embedding and makes the ablation's model axis explicit. Cached in
SQLite on a hash of (model, text) — the rebuild after the blob fix
recomputed 148 of 2938 chunks and took 12s instead of 84s.

**Response caching keyed on the full request.** Model, temperature, system
and user prompt. Any prompt change is a deliberate cache miss. Necessary
because RAGAS multiplies calls roughly fivefold per answer.

**Three prompt strategies differing in exactly one respect.** `naive`,
`abstain`, `abstain_confidence` share context format, citation
requirement and answer style, so the Phase 7 comparison isolates the
abstention instruction.

### Baseline configuration

Fixed 512-token chunks, no overlap, `all-MiniLM-L6-v2`, ChromaDB cosine,
top-5 dense retrieval. 2,938 chunks from 130 papers; median 2,113 chars
per chunk, median 19 chunks per paper, max 120 (GraphCast).

### Known weakness, deliberately unfixed

Retrieval for metric-name queries returns chart axis labels. Left broken:
the baseline exists to be beaten, and this is the motivating example for
the Phase 6 chunk-filter ablation.

### Defect found: abstention measurement counted formatting as behaviour

The first live test of the abstention pair asked "what was the RMSE of the
ECMWF IFS model in 1987" — a fact absent from a 2015–2026 corpus.

    naive:   "The provided context passages do not contain any information
              regarding the RMSE of the ECMWF IFS model in 1987. Therefore,
              I cannot provide an answer."
    abstain: INSUFFICIENT CONTEXT

Both refused. But `detect_abstention` matched only the sentinel string, so
the naive response was recorded as `abstained=False`. Had Phase 7 computed
hallucination rate as "did not abstain", the naive arm would have scored as
hallucinating on a question it correctly refused — turning a pure output
formatting difference into a large, entirely fictitious finding.

*Resolution.* Detection now covers prose refusals as well as the sentinel:
7 of 7 observed refusal phrasings caught, 0 of 5 real answers misflagged,
including an answer that supplies content while noting a gap. The regex is
explicitly a first pass — at Phase 7 scale (16 unanswerable questions x 3
strategies = 48 answers) every label is hand-verified.

### Risk raised: the headline finding may be small

Both prompt strategies refused the test question, so the abstention
instruction changed nothing but format at n=1.

That test is confounded — the question was trivially unanswerable, which is
the exact failure mode listed in the risk register. It is therefore not
evidence the effect is absent. It *is* evidence the effect is harder to
elicit than the plan assumes: instruction-tuned models refuse far better
than they did when "RAG hallucinates on unanswerable questions" became
conventional wisdom.

**Consequence for Phase 4.** Unanswerable questions must be *adjacent*, not
absent: retrieval returns confident on-topic passages from the right paper,
the question sounds exactly like something the corpus would answer, and the
specific fact is simply not stated. "What batch size did FengWu use during
fine-tuning?" rather than "what was the RMSE of IFS in 1987?". Absence must
be verified by keyword search across `data/interim/`, never assumed.

**Fallback headline.** If the abstention effect measures small, the Phase 6
chunk-quality result leads instead: figure axis labels dominate retrieval
for metric-name queries, and no surface feature separates them from results
tables. Concrete, measured, and rarely reported.

---

## Phase 4 — Gold evaluation set

**Done.** 74 questions, all valid, spanning 33 of 130 papers.

| Type | Count | Target |
|---|---|---|
| factual | 20 | 20 |
| comparative | 14 | 14 |
| multi_hop | 18 | 18 |
| unanswerable | 16 | 16 |
| ambiguous | 6 | 6 |

Coverage of 33 papers is by design. The remaining 97 are the distractor
set; concentrating evidence in a subset is what makes retrieval failures
visible.

### Design decisions

**Reference answers must be derivable from the evidence spans alone.** One
early answer (f019) was filled in from an outside web source. Rejected and
rewritten. If a reference answer contains facts absent from its evidence, a
system that retrieves perfectly and answers faithfully still scores wrong,
and the answerable/unanswerable boundary that the headline finding depends
on stops being sharp.

**Authoring runs through tools, never the console.** `04_search.py` finds
candidate passages, `06_find_evidence.py` converts a quoted phrase into an
exact character span, `09_add_question.py` fetches the quote from the
frozen extraction itself, validates, and writes. The only thing typed by
hand is a pair of integers.

**Questions are sorted by id on write.** File content depends only on what
is in the set, not the order it was authored, so re-running the tool
produces identical bytes and diffs show real changes.

### Construction notes by question type

**Unanswerable — adjacency, not absence.** A question is useless if it is
trivially outside the corpus. Each of the sixteen is built so retrieval
returns confident, on-topic passages while the specific fact is missing.
The strongest is u008: WeatherBench 2 lists SEEPS as a headline score *and*
lists FuXi in its model table, but never reports a SEEPS value for FuXi —
both halves present, only the intersection absent.

Absence is judged against `data/interim/`, not the source PDF. The system
can only answer from what is indexed, so a figure lost in extraction is
genuinely absent from the system's world. WeatherBench 2's Table 1 makes
this concrete for u001: it has an explicit "Inference time" column, filled
for Pangu-Weather and GraphCast, empty for FuXi.

Four candidates were killed by checking:

- *FuXi GPU hours* — stated outright ("approximately 30 hours on a cluster
  of 8 Nvidia A100 GPUs"), and again second-hand in two other papers.
- *FuXi-Extreme on cyclone tracks* — a full IBTrACS evaluation section.
- *GraphCast's CRPS* — GraphCast itself reports none, but GenCast reports
  CRPS scorecards for GraphCast-Perturbed across dozens of passages. The
  boundary was not sharp enough to keep.
- *GraphCast's learning rate* — `2501.19374v2` gives peak and terminal
  learning rates for a fine-tuned GraphCast and states its hyperparameters
  were identical to Lam et al. Same ambiguity.

**Multi-hop — chains collapse more often than they hold.** Papers in this
corpus almost always restate the detail they borrow; a citation usually
arrives with a summary attached. A chain survives only when the citing
paper *uses* something without explaining it, and the terminal fact is a
mechanism, a rationale or an admission rather than a value. Values get
restated; reasons do not. Chains abandoned for collapse include Aardvark →
WeatherBench 2 (Aardvark spells out the climatology computation itself),
FuXi-RTM → FuXi (its section 3.2 describes FuXi), and the first draft of
FuXi → GraphCast (FuXi's own span already gives the 2-to-12 step range, so
the question had to be re-aimed at the gradient-update count).

**A correlation rule, corrected mid-phase.** Two multi-hop questions fail
together when they share a *first* hop — the referring sentence is the hard
retrieval. The terminal paper matters much less. Applying the rule to
terminals instead was making usable chains look unusable; four chains
terminate in ConvLSTM (m003, m009, m013, m015) on four distinct facts —
structure, parameter counts, boundary padding, kernel size — from four
unrelated source papers, one of which is outside weather entirely.

**Five heavily-cited models have no paper in the corpus:** FengWu,
FourCastNet, NowcastNet, SwinVRNN, DiffCast. Search results look like
chains right up to the point of anchoring the second hop. Two rounds were
spent probing FengWu before dumping the manifest; every chain is now
checked against it before a lookup is spent.

**Ambiguous — built on real disagreements, not vagueness.** a001 asks how
long FuXi took to train: its own paper says ~30 hours of pre-training plus
~two days per cascaded model, while WeatherBench 2's table says ~8 days,
and those do not obviously reconcile. a004 exploits the FuXi family (FuXi,
FuXi-Extreme, FuXi Weather, FuXi-RTM) having different horizons. a005 is
ambiguous by variable alone: FuXi extends skillful lead time to 14.5 days
for T2M but 10.5 days for Z500.

### Defects found

**Console round-trips corrupted quotes.** Piping tool output on Windows
raised `UnicodeEncodeError` under cp1252 and silently mangled non-ASCII
mathematical characters when quotes were copied through the terminal.
Initially misdiagnosed as a display artifact. It was real; the file had to
be rebuilt.

*Resolution.* `sys.stdout.reconfigure(encoding="utf-8")`, then `--out` for
file delivery, then superseded entirely by `09_add_question.py`.

**`--out` left a stale file when a lookup failed.** `06_find_evidence.py`
returned early on no match without removing its previous output, so the
last run's file stayed in place and read as a successful result. Three
comparative questions (c006–c008) were written with xLSTM evidence attached
to GraphCast/FuXi questions. Mechanical validation passed: spans and quotes
were internally consistent, just from the wrong paper.

*Resolution.* Unlink the output path before the lookup. Caught in review by
`papers covered` dropping from 6 to 5.

**"Valid but wrong" evidence recurred.** f009, f015, f019, c002, c006–c008:
spans that validate mechanically, quotes matching source exactly, that do
not support the answer. The failure is semantic; no mechanical check
detects it.

*Resolution.* Per-question manual review, backed by the `papers covered`
count. There is no automated fix.

**A question duplicated one already in the set.** c009 as first drafted
asked how FuXi and GenCast step the forecast forward; c006 already asked how
GraphCast and FuXi extend to longer lead times, and the FuXi half of both
answers was the same fact. Near-duplicates correlate their errors, so
effective sample size is smaller than the count — and the set is only 74
questions with small ablation deltas expected.

*Resolution.* Replaced via `--replace` with a MoWE contrast. Standing
check: read the existing questions of a type before drafting a new one.

**Extraction artifacts break quote lookups.** Line-break joins drop spaces
and hyphens — the extracted text reads "datadriven", "namethe",
"theprevious". Lookups must quote short fragments that avoid hyphenated
compounds and word boundaries near line breaks.

### Layered defence

Each layer was added after a real escape, not in anticipation of one:

1. `08_validate_gold.py` — span and quote mechanical check
2. `papers covered` count — catches wrong-paper evidence
3. `06_find_evidence.py` stale-file deletion — catches silent reuse
4. `09_add_question.py` — removes hand-copying entirely
5. `07_check_absence.py` — corpus-wide co-occurrence check for
   unanswerable candidates, read by hand rather than trusted
6. Manual review against the existing set — catches valid-but-wrong
   evidence and redundancy

### Measured retrieval failures found during authoring

Authoring doubled as unstructured retrieval testing. Five failures, to be
quantified in Phase 6:

1. **"fraction skill score" returns chart axis labels**, not definitions.
2. **Generic metric vocabulary returns artifacts.** A query for
   "evaluation metrics CSI HSS RMSE CRPS scorecard" returned GraphCast axis
   labels at ranks 2, 4 and 5 and GenCast's table of contents — those
   artifacts are made of exactly that vocabulary.
3. **The balanced-loss query never finds the paper that introduced it.**
   Three independent attempts to retrieve B-MSE/B-MAE returned WMAE/WMSE
   from `2102.08175v1` and WSSIM from `2203.13263v1`, never TrajGRU.
   Mechanism: the corpus holds several near-synonymous weighted-loss
   formulations and the embedding cannot distinguish which paper
   originated which. m016 deliberately terminates in that passage.
4. **A query about ConvLSTM's mechanism returns mostly the paper that
   critiques it.**
5. **Referring sentences are effectively invisible to dense retrieval.**
   Queries phrased as citation language ("we follow the training strategy
   of", "our backbone is based on the architecture introduced in") scored
   0.37–0.38, barely above noise, and returned topically unrelated papers.
   The embedding keys on subject matter, not rhetorical structure.

   This has a direct consequence for the project's thesis: multi-hop
   questions depend on retrieving a referring sentence, so multi-hop
   Recall@k should come out markedly worse than factual Recall@k — not
   because the questions are harder to answer, but because the first hop
   is a sentence type the retriever cannot see.

### A factual error found in the corpus

`2404.06668v1`, a review of large meteorological models, states that
GraphCast forecasts "at a horizontal resolution of 0.25° across 13 vertical
levels". GraphCast's own paper states 37. Thirteen is the count
WeatherBench 2 lists for Pangu-Weather and FuXi. Recorded as m018, which
tests whether a system prefers the primary source over a secondary summary
that contradicts it.

### Known confound

The multi-hop questions are markedly harder to *construct* than any other
type and nearly all are labelled `hard`, while the ambiguous and factual
sets carry more `medium` and `easy` entries. If multi-hop Recall@k comes
out low in Phase 6, this difficulty skew is a confound to acknowledge, not
a finding to claim.

### Open decision, deferred to before Phase 6

With 74 questions, small ablation deltas are likely noise. Decide whether
to report confidence intervals, or to split the gold set and require the
winning configuration to hold on both halves.

### Standing lessons

Every corruption in this phase entered through a human copying text. The
fix was not more careful copying but removing the copy: the authoring tool
reads quotes from the frozen extraction itself, and the only thing typed by
hand is a pair of integers, which cannot be silently mangled.

And: check what is actually in the corpus before designing around it. Two
rounds of chain-hunting were wasted on models the corpus discusses
constantly but does not contain.

---

## Phase 5 — Retrieval metrics harness and baseline

**Done.** Metrics module, 28 tests, harness, two diagnostic scripts. The
baseline is measured and its ceiling is known.

### What was built

`src/rageval/evaluation/retrieval_metrics.py` scores retrieval by
character-span overlap, with no dependency on the retriever, the vector
store or pandas, so every metric is testable against hand-computed cases
without building an index. `scripts/10_eval_retrieval.py` runs it.
`scripts/11_diagnose_truncation.py` and `scripts/12_oracle_query.py` are
diagnostics, described below.

### Design decisions

**Two recall-style measures, not one.**

    Recall@k    was ANY evidence span retrieved
    Coverage@k  what FRACTION of them were

They are identical for single-evidence questions and diverge on the rest.
Coverage is reported as a fraction rather than a flag so a two-span
question that finds one span reads as half-solved rather than failed.

**Unanswerable questions are excluded, not zeroed.** They carry no
evidence, so 16 rows of 0.0 would read as 16 retrieval failures. They are
marked `scorable: False` with no metric keys at all, and are scored in the
generation phase instead.

**nDCG takes the true relevant-chunk count.** Without it, the ideal ranking
is inferred from what was actually retrieved, which flatters a run that
retrieved nothing. The harness counts, per question, how many chunks in the
index overlap its evidence. That number is diagnostic in its own right:
min 1, median 3, max 6 out of 2,938 chunks, which confirms the evidence
spans are tight rather than sprawling. An earlier worry that
chunk-sized spans would make any chunker look good does not apply.

**Which span was hit, not just how many.** For multi_hop, `evidence[0]` is
the source paper carrying the referring sentence and `evidence[1]` is the
terminal paper holding the answer, so the per-k `hit_spans` column
distinguishes "the chain broke" from "the chain broke at the first hop".

### Baseline results

Fixed 512-token chunks, no overlap, `all-MiniLM-L6-v2`, ChromaDB cosine.
58 scorable questions, 101 evidence spans.

| | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| Recall | 0.190 | 0.448 | 0.552 | 0.655 | 0.759 |
| Coverage | 0.121 | 0.322 | 0.382 | 0.494 | 0.598 |
| nDCG | 0.190 | 0.219 | 0.241 | 0.276 | 0.312 |

MRR 0.348.

| type | n | R@5 | R@10 | Cov@5 | Cov@10 | nDCG@10 | MRR |
|---|---|---|---|---|---|---|---|
| factual | 20 | 0.600 | 0.700 | 0.600 | 0.700 | 0.397 | 0.359 |
| comparative | 14 | 0.571 | 0.643 | 0.286 | 0.464 | 0.245 | 0.359 |
| multi_hop | 18 | 0.611 | 0.778 | 0.306 | 0.417 | 0.242 | 0.418 |
| ambiguous | 6 | 0.167 | 0.167 | 0.111 | 0.111 | 0.050 | 0.076 |

24% of answerable questions (14 of 58) retrieve nothing relevant in twenty
results.

### The result that survives scrutiny

**Both spans retrieved, by cutoff:**

| | k=5 | k=10 | k=20 |
|---|---|---|---|
| multi_hop (n=18) | 0 | 1 | 4 |
| comparative (n=14) | 0 | 4 | 4 |

**Zero of 32 two-span questions had both passages retrieved at k=5.** This
does not depend on which half was missed, on evidence ordering, or on
authoring habits.

It lands directly on Phase 7: if generation receives top-5 chunks, every
multi-evidence question in the gold set is unanswerable before the model
sees anything, and a hallucination measured there would be a retrieval
artifact wearing a generation costume. Generation must be fed a larger k
than the retrieval baseline uses, and the k it is fed must be reported.

### The ceiling

`scripts/12_oracle_query.py` queries the index with each evidence span's
own quoted text, removing the question from the loop.

| | span-level @10 |
|---|---|
| reachable by its own text | 0.911 |
| reachable from the question | 0.436 |
| gap | 0.475 |

oracle@1 is 0.693 and the median rank when found is 1. Comparative spans
reach 1.000 at k=10. Of 101 spans, 51 are findable by their own text but
missed from the question; only 6 are unreachable by either.

**The index is sound.** Chunking, embedding, storage and scoring all work.
The entire baseline gap is question-to-passage semantic matching.

Every ablation from here is reported against this ceiling. "Hybrid
retrieval moved Recall@10 from 0.44 to 0.61" is a number; "it closed 36% of
the measured gap to the ceiling" is a result.

### Three hypotheses, three failures

Recorded with their predictions intact, because the pattern matters more
than any one of them.

**1. Referring sentences are invisible to dense retrieval.** Queries
phrased as citation language scored 0.37–0.38, barely above noise, so the
prediction was that multi_hop Recall@k would fall well below factual.

*Falsified.* multi_hop R@10 was 0.778, the **highest** of any type, against
factual 0.700.

*And the comparison was confounded anyway.* Factual questions have one
evidence span; comparative and multi_hop have two. Recall@k asks whether
any span was found, so two spans means two chances. Cross-type Recall
comparison is mechanically biased toward multi-evidence questions.
**Coverage is the only fair cross-type comparison**, and by that measure
multi_hop is worst (0.417 against factual 0.700).

**2. The missing hop is systematically the referring one.** Prediction:
"second only" — answer found, referring sentence never — should dominate.

*Falsified in direction.* At k=20 the split was first only 7, second only
3. And it is uninformative regardless: a multi_hop question must describe
its first hop and withhold its second, or it is not multi_hop, so the query
paraphrases span 0 by construction. The split largely measures how the
questions were written. The comparative control, where span order carries
no meaning, leans the same way (4 versus 2), and 7-versus-3 on ten cases is
a two-sided binomial p of about 0.17.

**3. Encoder truncation is the root cause.** `all-MiniLM-L6-v2` accepts 256
tokens; chunks target 512, median 2,113 characters. Measured: **53.1% of
chunk text never reaches the encoder.** Prediction: visible spans hit,
cut-off spans do not.

*Falsified.* **42 of 57 missed spans are fully visible.** Mean visible
fraction 0.934 for hits against 0.850 for misses, both medians 1.000. Only
6 spans are entirely cut and 11 majority-cut. The mechanism is real; it is
not the cause.

**A fourth, tested and dropped before it became a hypothesis: signal
dilution** — a short fact diluted inside a large chunk. Mean signal ratio
(span length over embedded chunk text) was 0.733 for hits and 0.732 for
misses, and the quartile hit rates were non-monotonic
(0.333 / 0.458 / 0.696 / 0.292). No effect.

The only real correlate found: spans covered by two chunks hit 50% of the
time against 33% for one chunk. Two lottery tickets beat one.

### The premise was wrong

The investigation started from "factual Recall@10 of 0.700 is anomalously
low for questions authored from their own target passage." That premise was
never checked. A 22M-parameter general-purpose sentence model applied to
dense technical prose full of equations and notation is plausibly just this
weak; published numbers for models of this size sit around 0.6–0.8 on far
easier corpora. **A weak baseline is not a broken one**, and three
hypotheses were built on assuming otherwise.

### Truncation, recorded as a defect rather than a cause

53.1% of chunk text is never embedded. It did not explain the misses, but
it caps the ceiling, and half the corpus is effectively unindexed. It
belongs in the Phase 6 chunk-size ablation with that framing.

**It also creates a confound Phase 6 must control**: comparing embedding
models at 512-token chunking would confound model quality with each model's
context window, and any model with a longer window would win for reasons
unrelated to embedding quality. Either hold chunk size below every
candidate model's limit, or report the interaction explicitly.

### Six unreachable spans

Not retrieved even by their own text. Two are the *same* WeatherBench span
(837 characters) failing under both `a003` and `m006`. One, `m007` span 1,
is the longest quote in the set at 1,822 characters — past the 256-token
window, so the oracle query is itself truncated. That is the oracle being
imperfect rather than the index failing. The remaining three are worth a
hand check before Phase 6.

### A prediction for Phase 6, made in advance

If the bottleneck is question-to-passage matching, the query-side axes
(embedding model, hybrid BM25, query rewriting) should move the numbers
substantially and the chunking axes should move them much less. If chunk
size dominates instead, this prediction is wrong and the build log will say
so. It is recorded here, before the ablations run, so it cannot be
reverse-engineered afterwards.

### Standing lesson

Two mechanisms were inferred from aggregate patterns and both came out
backwards. What resolved the question was a test designed to
*discriminate* — query the index with the answer text and see whether the
target is reachable at all — rather than one designed to confirm. An
aggregate can tell you that something is wrong; it almost never tells you
what, and a plausible mechanism that fits the aggregate is not evidence.
Run the test whose two outcomes point in opposite directions.

---


## Phase 6 — Ablations

### Sweep 1 — chunk size

Three configurations, identical in every other respect: fixed 256, 512
(baseline) and 1024 token chunks, no overlap, `all-MiniLM-L6-v2`, ChromaDB
cosine. Each writes to its own collection (`sweep_fixed256_minilm`,
`sweep_fixed1024_minilm`) so a `store.reset()` cannot clobber the baseline.
58 scorable questions, 101 evidence spans.

Span-level recall, whole gold set:

| configuration | k=10 | budget 10k | budget 20k | oracle ceiling @10 |
|---|---|---|---|---|
| fixed_256 | 0.347 | **0.337** | **0.455** | 0.960 |
| fixed_512 (baseline) | **0.436** | 0.307 | 0.416 | 0.911 |
| fixed_1024 | 0.416 | 0.168 | 0.287 | — |

Paired tests at the 20k budget: McNemar exact on discordant span pairs,
cluster bootstrap over questions, Holm-corrected across the three pairs.
Re-run through `16_paired_test.py` after sweep 2, so every paired test in
Phase 6 comes from one instrument.

| comparison | delta | 95% CI | Holm p |
|---|---|---|---|
| 512 vs 256 | +0.040 | [−0.040, +0.114] | 0.5034 |
| 512 vs 1024 | −0.129 | [−0.222, −0.042] | 0.0089 |
| 256 vs 1024 | −0.168 | [−0.265, −0.078] | 0.0045 |

At fixed k=10 the sign reverses and 256 becomes the *worse* configuration
(0.347 against 0.436). That comparison has not been re-run through
`16_paired_test.py`, so no corrected p is quoted for it.

**Result: null on chunk size.** 1024 is reliably worse at equal budget.
Between 256 and 512 the difference is +0.040 in favour of 256 and the
confidence interval straddles zero. The pre-registered rule — switch only
when the paired CI excludes zero *and* selection is stable — says carry the
baseline forward. **Baseline stays fixed-512.**

### The finding: comparing chunk sizes at fixed k measures text volume

The two metrics do not merely disagree about the margin; they invert the
ranking, and the inversion is stable rather than noise. Across 4,000
stratified half-splits (below), `fixed_1024` is selected 30.8% of the time
at k=10 and **once in 4,000** at the 20k budget. `fixed_256` goes the other
way, 1.9% to 84.4%.

The mechanism is arithmetic. At the same k, 1024-token chunks hand the
scorer roughly four times the characters that 256-token chunks do. Fixed-k
therefore ranks chunkers largely by how much text they deliver, which is
not a property anyone wants to select on — a generator is bounded by its
context window, not by document count. This was the stated reason for
building the character-budget metric in Phase 5; it is now measured rather
than argued.

**Scope of the confound.** It applies to comparisons that *change chunk
size*. The remaining sweeps (hybrid BM25, reranking, k, embedding model)
hold chunking fixed, so both arms hand over the same text at the same k and
fixed-k is not confounded there. The character budget stays the primary
reporting metric regardless, for comparability across the whole phase.

### Selection stability

`scripts/15_selection_stability.py`. Repeated stratified half-splits: split
the 58 scorable questions in half keeping the 20/14/18/6 type mix, select
the winner on one half, score it on the other, 2,000 splits × 2 directions.
Ties are broken at random — taking the first configuration would quietly
award every tie to whichever tag was typed first, and ties are 11–19% of
selections at this sample size.

This exists because Phase 6 scores ~17 configurations against one question
set. Whatever comes out top is top partly because it suits *these*
questions, and those questions are in the set every time. Reporting the
winner's score then reports its luck along with its merit.

| selection metric | winner | win rate | optimism | winner also wins held-out |
|---|---|---|---|---|
| budget 20k | fixed_256 | 84.4% | +0.018 | 74.8% |
| budget 10k | fixed_256 | 74.6% | +0.026 | 57.1% |
| k=10 | fixed_512 | 67.3% | +0.031 | 44.6% |

**Win frequency is not a significance test**, and the two results are not in
conflict. "Wins more often than not" is a much weaker claim than "differs by
a detectable amount": a true difference of +0.039 with SE ≈ 0.040 predicts a
half-split win rate near 75%, so 84.4% is what a small, positive,
unresolved difference looks like. Both numbers describe the same
undecidable gap.

Two observations do favour 256 and are recorded for later: its optimism
when selected is only **+0.010**, so its lead is not a selection artifact
(the baseline's is +0.060 — it wins only when the question draw flatters
it), and its ceiling is **0.960 against 0.911**, leaving more headroom for
the query-side work to exploit.

**The power limit, stated plainly.** The CI width implies SE ≈ 0.040 on the
paired difference. Detecting +0.039 at 80% power needs SE ≈ 0.014 — roughly
**eight times the spans, on the order of 450–500 questions**. A gold set of
74 cannot settle a four-point chunking difference, and no amount of
resampling changes that. Better to record the limit than to keep re-running
the comparison.

**Replication instead of re-testing.** Rather than repeat this comparison,
the winning end-of-phase pipeline will be run under both 512 and 256 as a
confirmation. Two independent conditions agreeing is worth more than one
test repeated.

The reselection rate also tracks discriminating power exactly as it should:
74.8% at the 20k budget, 57.1% at 10k, 44.6% at fixed k, with ties rising
from 11% to 19% as it degrades. At k=10 the half-split winner fails to win
the other half more often than not, because 512 and 1024 sit 0.02 apart
there. The metric that holds text constant is the one that separates
configurations.

### Complete-miss audit

`scripts/14_inspect_misses.py` prints every question no configuration
reaches, annotated with each span's oracle rank, so the two very different
kinds of miss can be told apart:

    oracle hit,  question miss  -> matching failure, the finding
    oracle miss, question miss  -> unreachable; suspect the evidence

Six questions are missed by all three runs at k=20: c002, c006, c008, f005,
f013, m010. All six were read in full.

**No gold-set defects.** Every one has well-formed text and evidence the
oracle reaches, mostly at rank 1. Several of the misses across the wider set
are deliberate probes whose authoring notes predicted the failure before it
could be measured — f005 (a bullet list), f003 (a results table), f011
(pseudocode), and f015, whose B-MSE passage was already recorded in Phase 4
as unreachable by query. Those are the questions doing their job.

Two are genuinely surprising and are the sharpest cases in the set:

- **f013** names two rare tokens, `xLSTM` and `ConvLSTM`, sits at oracle
  rank 1, and misses under every configuration.
- **m010** contains "9.4 percent" and "ECMWF-IFS" near-verbatim from its
  target passage and also misses everywhere.

Both look like lexical signal that a dense encoder dilutes across 256 word
pieces of surrounding prose.

**Pre-registered prediction.** If lexical dilution is the bottleneck, adding
BM25 should recover **f013, f015, m010 and c002 specifically**. If hybrid
retrieval raises the aggregate but leaves those four missing, the lexical
explanation is wrong and the gain came from somewhere else. Recorded before
sweep 2 runs.

**One covariate found.** Spans under 200 characters hit 25% of the time
against 58% for spans of 700–1000 characters. That is the price of choosing
tight evidence spans to keep the ablation discriminating (Phase 5 recorded
median 3 relevant chunks per question for the same reason). It is a
property of the gold set, not of any configuration, so it does not bias the
comparison — but any absolute recall number in this project is depressed by
it and should not be read against published benchmarks with looser
annotation.

### Defects found

**pandas dtype inference silently zeroed the span counts.** The `hit_spans`
columns hold strings like `"0;1"`, but a column containing only `"0"` and
blanks is inferred as `float64`, so `"0"` comes back as `0.0` and a bare
`.isdigit()` test counts nothing. The first budget-matched table therefore
reported zero spans found for any run whose hit column happened to contain
only single indices, and the `% of ceiling` column with it.

*Resolution.* `_count_spans` in `13_compare_runs.py`, which normalises the
`.0` suffix before counting, with the bug named in its docstring. Now
duplicated in `15_selection_stability.py`; if a third script needs it, it
moves into `rageval`.

**The retrieval depth capped the budget view.** The first sweep ran with
`--k-max 20`. At the 40k budget `fixed_256` consumed exactly `20.0` chunks —
it had hit the retrieval limit, not the budget, so the budget was not
actually being held constant. `fixed_512` at 19.4 was nearly capped too,
which made the baseline appear to win at 40k. Re-run at `--k-max 50` /
`--k 50` the picture changed.

*Both were caught before they influenced a decision*, but only because the
budget table was read column by column rather than skimmed for the winner.
An exactly-round `20.0` in a column of decimals is the kind of tell that is
invisible unless someone is looking for it.

**Over-read "consistent across four budgets".** Reported the four budget
columns as if they were four agreeing trials. They are nested subsets of one
ranked list — the 5k result is contained in the 40k result — so they cannot
corroborate each other. Corrected when the paired statistics came back and
showed a single undecided difference rather than four confirmations.

### Decisions

1. **Baseline remains `fixed_512`.** The rule was written before this
   number was seen; 84.4% win frequency with a CI containing zero does not
   clear it.
2. **`fixed_128` dropped.** It was in the plan as a fourth chunk size. With
   256 and 512 indistinguishable at n=58, a fourth point on the same axis
   is underpowered by construction and would spend a run for no resolvable
   answer.
3. **Character budget is the primary reporting metric for Phase 6.** Fixed
   k is reported alongside it where chunking is held constant, and not used
   for selection where it is not.
4. **`fixed_256` is the standing challenger**, to be re-run against the
   winning pipeline at the end of the phase as a replication rather than a
   repeat.

### Standing lesson — sweep 1

The stability analysis was built to guard against selection noise on the
chunking axis. It found something else: a metric whose ranking inverts
depending on how much text each arm is allowed to hand over. The guard
against being fooled by the answer caught a problem in the question.

And the narrower version: two numbers can both be right and still point
opposite ways. p = 0.503 and an 84.4% win rate are the same fact — a small
positive difference this gold set cannot resolve — described by two
instruments with different sensitivities. The temptation is to quote
whichever one supports the switch.

---

### Sweep 2 — retrieval function

Three arms on the *same collection*: `dense` (the baseline), `bm25` alone,
and `hybrid` (reciprocal rank fusion of the two, k=60, pool 100 per
retriever). BM25 indexes the chunks read out of the store rather than
re-chunked from `data/interim/`, so "only the retrieval function differs"
is true by construction rather than by assumption. Nothing was re-indexed
and the bm25 arm computes no embeddings at all.

The bm25-only arm exists because a hybrid result is uninterpretable without
it: if hybrid beats dense but bm25 alone beats both, the dense half is dead
weight. That turned out to be exactly the case.

Span recall, 58 scorable questions, 101 spans:

| arm | k=10 | budget 20k | complete misses @10 |
|---|---|---|---|
| dense (baseline) | 0.436 | 0.416 | 20/58 |
| **bm25** | **0.604** | **0.584** | **9/58** |
| hybrid | 0.545 | 0.525 | 12/58 |

Paired tests at the 20k budget (McNemar exact, cluster bootstrap over
questions, Holm-corrected across the three pairs) and selection stability
(2,000 stratified half-splits x 2 directions):

| comparison | delta | 95% CI | Holm p | win rate |
|---|---|---|---|---|
| bm25 vs dense | +0.168 | [+0.082, +0.253] | 0.0069 | 93.2% |
| hybrid vs dense | +0.109 | [+0.037, +0.186] | 0.0255 | 6.8% |
| bm25 vs hybrid | −0.059 | [−0.141, +0.020] | 0.2101 | — |

The dense baseline was selected in **zero of 4,000 half-splits**. BM25's
optimism is +0.003; hybrid's is +0.111, which is what a configuration looks
like when it wins only on question draws that flatter it.

**BM25 clears both halves of the switching rule and becomes the baseline.**
It closes 35.4% of the dense baseline's headroom to the measured ceiling —
the largest movement any single change has produced in this project — and
it uses no neural model, no GPU and no embedding cache. Against hybrid it
is not statistically separable; the tie is broken on parsimony, not on a
number.

### The fusion failed, and the provenance column says why

`dense only 0, bm25 only 0, both 100.0%` of the fused top 10, across every
question.

This is arithmetic, not chance. A document ranked 50th by both retrievers
scores 2/(60+50) = 0.0182; a document ranked *first* by one and absent from
the other scores 1/(60+1) = 0.0164. Under RRF with k=60 and a pool of 100,
presence in both lists beats a first-place finish in either, so the fused
head contains only documents both retrievers already had. **The fusion can
never surface anything only BM25 found** — which is precisely the recall
the sweep existed to capture.

The pre-registered check makes it concrete: bm25 recovered f015 and m010;
hybrid recovered f015 and **lost** m010. Fusion destroyed a win the lexical
arm had.

The k=60 constant comes from fusing many similar-quality TREC runs, where
agreement is the useful signal. Here the arms have different strengths and,
more to the point, near-disjoint failure modes, so the value is
complementary recall and the constant discards it. That argument needs no
sight of the results; it should have been made when the constant was fixed.

### How much fusion could ever be worth here

The discordant split is 6/23 — BM25 found 23 spans dense missed, dense
found 6 BM25 missed. A *perfect* fusion, keeping every span either
retriever found, would score 0.584 + 6/101 = **0.644**, only +0.059 above
bm25 alone. That is the same magnitude as the bm25-vs-hybrid gap that came
back not separable at this sample size.

So the entire remaining prize from fusion sits inside this gold set's noise
floor. That settles the budget for it: one exploratory run, not a sweep.

### The confound check, and what it did to the headline number

Every gold question was authored while reading its evidence passage. That
is the right way to build span-anchored ground truth and it hands a lexical
retriever the exact tokens the author had in front of them, while giving
dense retrieval nothing comparable. Neither the paired test nor the
stability analysis can see this: both resample the same 58 questions, and a
bias baked into how all of them were written survives any amount of
resampling. The switching rule guards against *sampling* error; this is
*construct* error.

`scripts/17_lexical_overlap.py` measures, per question, the share of its
IDF-weighted vocabulary already present verbatim in its own evidence, using
BM25's own IDF formula and the retriever's own tokenizer — so "overlap"
means overlap in the quantity BM25 actually scores on. Median 0.468, range
0.037 to 1.000.

Criterion declared before the numbers were seen: split into terciles by
overlap; if BM25's advantage in the low tercile is at least half its
advantage in the high tercile, the effect is about retrieval.

| tercile | dense | bm25 | delta | 95% CI |
|---|---|---|---|---|
| low overlap (19q) | 0.519 | 0.556 | +0.037 | [−0.115, +0.208] |
| mid overlap (19q) | 0.382 | 0.618 | +0.235 | [+0.103, +0.385] |
| high overlap (20q) | 0.375 | 0.575 | +0.200 | [+0.070, +0.325] |

Ratio low/high = **0.19** at the budget and **0.00** at k=10, against a
threshold of 0.50. **Verdict: BOUNDED.** The aggregate +0.168 is not a
clean estimate of retrieval quality; it partly measures how the gold set
was authored.

**The interpretation that did not survive.** BM25 is flat across the range
(0.556 / 0.618 / 0.575) while the dense arm falls monotonically (0.519 /
0.382 / 0.375), which suggested the two retrievers suit different question
styles — terminology-dense lookups versus paraphrased conceptual questions.
Tested properly, by bootstrapping the difference of the tercile advantages
with the terciles resampled independently, that is **+0.163, CI [−0.049,
+0.367]** at the budget and **+0.200, CI [−0.003, +0.392]** at k=10. Both
include zero. Comparing three confidence intervals by eye is not an
interaction test, and when the difference got its own interval the story
did not hold. Recorded as an observation, not a finding.

**A defect in the pre-registration itself.** The criterion put a hard
threshold on a *ratio of two noisy estimates* with no uncertainty attached.
The denominator's CI alone is [+0.070, +0.325], so the ratio could read
anywhere from roughly 0.11 to 0.53 from denominator variation. It should
have been declared as an interval rule. It is honoured as written — that is
what pre-registration means — and the flaw is recorded rather than
retrofitted. Future criteria in this project name an interval, not a point.

### How the effect size is reported from here

Not "+0.168". The sentence that survives every instrument run against it:

> BM25 outperforms the dense baseline by +0.168 span recall
> [+0.082, +0.253], Holm p = 0.0069, selected in 93% of half-splits. The
> advantage is not uniform: it is smallest among questions that share least
> vocabulary with their evidence (+0.037, CI [−0.115, +0.208]), where the
> two are not separable. Whether that variation is real is unresolved at
> n=58 — the interaction interval includes zero.

Note that "not separable" in the low tercile is **not** "equal". Nineteen
questions and 27 spans give an interval consistent with a modest dense
advantage and with a large BM25 one. It fails to establish a difference; it
does not establish equivalence.

### The ceiling stops being comparable

`13_compare_runs.py` prints `-` for the bm25 and hybrid ceilings, and that
is correct rather than a missing file. The oracle queries the index with
each span's *own text*, which under a lexical retriever matches itself
almost perfectly — BM25's oracle ceiling would return near 1.0 and restate
its recall. The ceiling is a property of (index, retrieval function), not
of the index alone, so "% of ceiling" is not comparable across retrieval
functions. Improvements from here are reported against the dense
baseline's measured headroom, which is what the `gap closed` column already
does.

### Pre-registered prediction: partially confirmed

Recorded before sweep 2 ran: if lexical dilution is the bottleneck, BM25
should recover f013, f015, m010 and c002 specifically.

**bm25 recovered 2 of 4** (f015, m010); f013 and c002 miss under every
configuration tried so far, dense, lexical and fused. **hybrid recovered 1
of 4**, losing m010 to the intersection behaviour above. Half right: the
lexical explanation holds for two of the four sharpest cases and fails for
the other two, which remain unexplained and are now the most interesting
questions in the set.

The check is computed by `10_eval_retrieval.py` itself and printed on every
non-dense run, so it cannot be quietly forgotten once an aggregate looks
good.

### Standing lesson — sweep 2

Two of the three instruments built this phase were built to guard against
being fooled by an answer, and both caught something in the *question*
instead. The stability analysis, built for selection noise, found a metric
whose ranking inverts with how much text each arm hands over. The overlap
check, built to bound a confound, found that the tercile pattern I wanted
to explain does not survive an interaction test.

And the narrower one: a ratio of two noisy estimates is not a criterion. If
a rule is worth declaring in advance, it is worth declaring with an
interval attached.

### Sweep 2b — fusion, exploratory and post-hoc

Labelled post-hoc throughout. RRF at k=60 was the pre-registered
configuration; every constant tried here was chosen after seeing that it
failed, so nothing in this section can replace a baseline on its own.
Capped at one run because the ceiling arithmetic in sweep 2 put the entire
remaining prize at +0.059, inside this gold set's noise floor.

**The mechanism prediction, confirmed.** Sweep 2 diagnosed RRF at k=60 with
a pool of 100 as an intersection ranker: agreement beats a first-place
finish, so unique recall cannot reach the fused head. Lowering k should let
singletons through. Share of the fused top 10 unique to one retriever:

| fusion | dense only | bm25 only | both |
|---|---|---|---|
| RRF k=60 | 0.0% | 0.0% | 100.0% |
| RRF k=10 | 7.9% | 8.8% | 83.3% |
| RRF k=1 | 13.3% | 15.3% | 71.4% |
| interleave | 15.2% | 18.8% | 66.0% |

m010, which k=60 had lost, returns in every variant — 2/4 on the
pre-registered question list, matching BM25 alone.

**The performance result, null.** Span recall at k=20, against BM25 alone:

| arm | span recall | delta | 95% CI | Holm p | win rate |
|---|---|---|---|---|---|
| bm25 | 0.663 | — | — | — | 14.6% |
| hybrid_rrf1 | 0.683 | +0.020 | — | — | 34.3% |
| hybrid_rrf10 | 0.683 | +0.020 | [−0.032, +0.077] | 1.0000 | 40.8% |
| hybrid_interleave | 0.663 | +0.000 | [−0.053, +0.058] | 1.0000 | 10.2% |

Every pairwise comparison returns Holm p = 1.0000 on 6 to 8 discordant
spans. The stability analysis calls it a coin flip at 40.8% for four arms,
with a mean winning margin of 0.009 and **2,187 of 4,000 half-splits ending
in an exact tie** — over half the time the arms are literally identical on
29 questions. The winner reselects on the held-out half 28% of the time,
which for four arms is chance.

**The control did its job.** Interleaving has no constant: each retriever
nominates its next unseen document in turn, so each one's rank-1 is
guaranteed into the fused top 2. It scored 0.663 — BM25's number exactly.
That is what makes the rrf10 result readable: a tuning-free fusion buys
nothing, a tuned fusion buys +0.020 that no instrument can distinguish from
zero, and the difference between them is not separable either. Without the
control, +0.020 from a constant picked after the fact would have been very
easy to write up as a finding.

**Why the union ceiling was never reachable.** Sweep 2 computed that a
fusion keeping every span either retriever found would score 0.644 at the
20k budget. Interleaving comes closest to that ideal — it has the most
unique recall of any arm, 18.8% bm25-only and 15.2% dense-only — and still
scores exactly what BM25 scores. **Fusion trades slots; it does not add
them.** Every document promoted for being unique displaces one that was
there on merit, and at fixed k or fixed budget that exchange is close to
even. The union ceiling is only collectable by increasing depth, which
changes the budget and so is not the same experiment.

That is a better explanation of the null than "the effect was too small to
see", and it was not obvious in advance.

### Decision: fusion closed, BM25 stands

No arm replaces BM25. The pipeline for sweeps 3 onward is BM25 alone.

**The embedding-model sweep is dropped**, executing the condition recorded
after sweep 2: it was to run only if the fusion work showed dense retrieval
earning its place. Dense contributes unique documents — 15.2% of
interleaving's top 10 — and no measurable recall. A sweep over embedding
models is now a sweep over a component that is not in the pipeline. Its
budget goes to reranking and to the paraphrase validity test instead.

### Predictions made in advance, and how they landed

Kept as a scoreboard because a pre-registration that is only consulted when
it succeeds is not a pre-registration.

| prediction | recorded | outcome |
|---|---|---|
| Query-side axes move the numbers; chunking axes move them much less | Phase 5 | **Confirmed.** Chunking +0.040, not separable. Retrieval +0.168, Holm p = 0.0069. |
| RRF's k controls whether unique recall survives; lowering it restores singletons | Sweep 2 | **Confirmed.** 0.0% → 8.8% → 15.3% bm25-only as k fell from 60 to 10 to 1. |
| The whole remaining fusion prize is +0.059, inside the noise floor | Sweep 2 | **Confirmed.** Measured deltas +0.000 to +0.020, every Holm p = 1.0000. |
| BM25 recovers f013, f015, m010, c002 | Sweep 2 | **Half.** f015 and m010 recovered; f013 and c002 miss under every configuration tried. |
| 512 vs 256 is a coin flip on half-splits | Sweep 1 | **Wrong in magnitude.** Predicted ~50%, measured 84.4%. The conclusion (not separable) held; the prediction about the instrument did not. |
| BM25's advantage survives among low-overlap questions | Sweep 2 | **Failed.** Ratio 0.19 against a threshold of 0.50; effect size bounded. |
| Dense and lexical retrieval suit different question styles | Sweep 2 | **Not established.** Interaction CI [−0.049, +0.367] includes zero. |

Two confirmed mechanisms, two confirmed magnitudes, one half, one wrong,
one failed and one unsupported. The falsified entries are the ones that
changed what the project claims.

### Standing lesson — sweep 2b

The control arm was worth more than the treatment arms. Three RRF variants
would have produced a ranking, a best constant and a plausible story about
why k=10 suits two retrievers with disjoint failure modes. Interleaving —
which has nothing to tune and therefore nothing to get right — landed on
BM25's exact score and made the whole ranking legible as noise.

When a knob is turned after seeing that its default failed, the useful
experiment is not which setting wins. It is whether a method with no knob
does just as well.
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

**The experiment matrix was circular.** As written, the chunking sweep held
retrieval at "hybrid" while the retrieval sweep held chunking at "the
winning chunker". Neither can run first. This is not a scheduling
inconvenience: with the axes defined in terms of each other, the reported
best configuration would depend on an ordering the plan never states, and
so could not be reproduced from the plan.

*Resolution.* Name the seed explicitly and fix the order.

    seed:  fixed_512 / all-MiniLM-L6-v2 / dense-only / k=10
    order: chunking -> retrieval -> reranking -> k -> embedding model

Each sweep varies one axis with all others held at the seed, except that a
sweep inherits any change from an earlier sweep that cleared the switching
rule. Sweep 1 cleared nothing, so sweep 2 runs on the untouched seed.

*Limitation, recorded rather than solved.* This is a greedy coordinate
search, so the result is order-dependent: a chunking that only pays off
under hybrid retrieval will be rejected in sweep 1 and never revisited on
its own. That sits alongside the interaction blindness the plan already
admits. The end-of-phase replication run (winning pipeline × {512, 256}) is
a partial check on exactly this and is the only interaction actually
measured.

**Phase 4's open decision is resolved: confidence intervals plus selection
stability, not a held-out split.** The Phase 4 log deferred a choice between
reporting CIs and splitting the gold set so a winner must hold on both
halves. A permanent holdout is the wrong instrument at n=58: a 30% holdout
is 17 questions, whose CI on recall is roughly ±0.12 — too wide to separate
any configuration from any other. That trades a third of the gold set for a
number that says nothing. Repeated stratified half-splits
(`15_selection_stability.py`) use every question in both roles across
thousands of draws instead, and report the two things a holdout was wanted
for: how often each configuration is selected, and how inflated the
selected one's score is.

**Switching rule, pre-registered.** A configuration replaces the baseline
only when its paired CI excludes zero **and** its half-split win rate is
decisively above chance. When either fails, the baseline carries forward and
the sweep is written up as a null result. Applied to sweep 1 this keeps
`fixed_512` despite `fixed_256` leading at every character budget.

**`fixed_128` removed from the chunking sweep.** Underpowered by
construction once 256 and 512 proved indistinguishable at this sample size.

**Sweep 1's p-values are now Holm-corrected.** Re-run through
`16_paired_test.py` so every paired test in Phase 6 comes from one
instrument. The conclusions are unchanged; the numbers are not:

| comparison | delta | 95% CI | Holm p |
|---|---|---|---|
| 512 vs 256 | +0.040 | [−0.040, +0.114] | 0.5034 |
| 512 vs 1024 | −0.129 | [−0.222, −0.042] | 0.0089 |
| 256 vs 1024 | −0.168 | [−0.265, −0.078] | 0.0045 |

The sweep-1 section above has been updated in place. It originally quoted
an uncorrected p = 0.004 for the 1024 comparison; corrected, it is 0.0089.

**The seed configuration changes after sweep 2.** BM25 cleared the
switching rule, so sweeps 3 onward inherit it:

    seed:  fixed_512 / BM25 / k=10
    order: chunking -> retrieval -> reranking -> k -> embedding model

**The order-dependence limitation now has a concrete instance, not just a
caveat.** Sweep 1 compared chunk sizes *under dense retrieval* and found
256 and 512 indistinguishable. Chunk size interacts with retrieval function
in an obvious way — BM25's length normalisation (b = 0.75) behaves
differently from an encoder that truncates at 256 word pieces — so that
result does not transfer automatically to the new baseline. This is exactly
the greedy-coordinate-search weakness the plan already admits, now
observable rather than hypothetical.

*Consequence.* The end-of-phase replication run becomes: winning pipeline x
{fixed_512, fixed_256} **under BM25**. If 256 wins there, the sweep-1 null
was an artifact of the retrieval function it was measured under, and the
build log says so.

**The embedding-model sweep is dropped.** It was planned as the last axis,
then made conditional on the exploratory fusion work showing dense
retrieval earning its place. Sweep 2b settled it: dense contributes unique
documents (15.2% of interleaving's fused top 10) and no measurable recall,
with every fusion arm returning Holm p = 1.0000 against BM25 alone. A sweep
over embedding models is now a sweep over a component that is not in the
pipeline. Its budget moves to reranking and to the paraphrase validity
test. Dropping a planned experiment because the pipeline moved is a result,
not an omission.

**Fusion was capped at one exploratory run, and it is now closed.** The
run covered RRF at k=1 and k=10 plus plain rank interleaving, labelled
post-hoc because choosing a constant after seeing that k=60 failed is a
post-hoc choice however well motivated the mechanism argument is. Result:
null, with the tuning-free control landing on BM25's exact score. The
sweep-2b section records why the union ceiling was unreachable — fusion
trades slots rather than adding them.

**The seed for sweeps 3 onward is BM25 alone.**

    seed:  fixed_512 / BM25 / k=10
    order: reranking -> k -> (embedding model: dropped)

**`_count_spans` moves into `rageval`.** Three scripts now carry private
copies of the same pandas-dtype guard (13, 15, 16, and 17 has a fourth
variant). The bug it defends against — a column of only "0" and blanks
inferred as float64, so `.isdigit()` counts nothing — already produced one
wrong results table in sweep 1. Four copies is how a fix lands in three of
them.

---

## Open items

- 4 papers retain reference lists (accepted, documented)
- Quality thresholds in `classify` were set a priori and validated only
  against the observed distribution; not tuned
- Engine comparison covers 15 papers, not all 130