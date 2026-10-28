# CLAUDE.md — RAG Evaluation Lab

Operating manual for this repository. Read this fully before acting. The
narrative detail lives in `docs/build-log.md` (1,442 lines); this file is
the orientation, the conventions, the current state and the rules that
govern what counts as a result.

**Repo:** `rag-evaluation-lab` · **Owner:** Roshan Tariq
**Platform:** Windows, PowerShell, VSCode, Python 3.11+, RTX 5070
**Path:** `C:\RAG Project\rag-evaluation-lab`
**Status date:** 28 October 2026

---

## 1. How to work with the owner

These are not preferences to be optimised away. They were established over
the whole project and they hold.

- **Guide step by step. He runs every command himself.** Explain what a
  command does and why before giving it. Do not run the pipeline for him.
- **PowerShell only.** Never bash syntax, never `&&` chaining, never
  forward-slash paths in commands. Quote the repo path — it contains a
  space.
- **Whole files, not patches.** When a file changes, emit the complete
  file. His words: *"why not give me the entire code for this .py to avoid
  confusion."* Targeted edits have twice caused the wrong version to be
  saved; whole files are safer here even when wasteful.
- **He pastes terminal output; you interpret it and draft the next step.**
- **Verify what landed.** Files delivered have silently failed to reach
  `scripts/` more than once. When a command errors with "unrecognized
  arguments", suspect a stale file before suspecting the logic, and give a
  `findstr /C:"<marker>" <path>` check.
- **Keep the build log current.** Every sweep closes with a section in
  `docs/build-log.md`: what was built, what was measured, what was decided,
  what went wrong, and the standing lesson. Falsified hypotheses stay in
  with their predictions intact — they are the most valuable content in the
  document.

---

## 2. What this project is

The deliverable is **not a RAG system.** It is a set of measured findings
about which design decisions change retrieval quality and answer
faithfulness. The system is the apparatus that produces them.

Three things must be true at completion:

1. A reviewer reads a results table and learns something they did not know
   about RAG behaviour.
2. Every number is reproducible from a fresh clone.
3. The limitations section is honest enough that the confirmed findings
   become credible by association.

**Planned headline:** hallucination rate on deliberately unanswerable
questions, and how far explicit abstention instructions move it. Phase 3
already flagged the risk that this effect measures small — instruction-tuned
models refuse well. Fallback headline is the chunk-quality finding (figure
axis labels dominate retrieval for metric-name queries). As of now the
*strongest measured* result is actually the BM25 one in §6.

Full plan, scope boundary, risk register, cost model and definition of done:
`claude/rag-evaluation-lab-build-plan.md` in the Claude project.

---

## 3. Hard invariants — do not break these

- **`data/interim/` is frozen.** Every gold evidence span is
  `(arxiv_id, char_start, char_end)` against this exact extractor output.
  Any change to extraction invalidates all 101 spans. If extraction must
  change, the gold set must be re-anchored, and that is a multi-day job.
- **Gold evidence anchors to character spans, never chunk IDs.** Chunk IDs
  are not stable across chunking strategies, so chunk-ID ground truth would
  be valid for exactly one arm of the ablation.
- **Chunk span provenance holds by construction:**
  `text[char_start:char_end] == chunk.text`. Verified across 300 randomised
  documents × 6 size/overlap settings.
- **The gold set is closed at 74 questions.** Do not add, remove or reword
  questions. Changing it invalidates every prior comparison. (The paraphrase
  test in §8 writes a *separate* file; it does not touch the original.)
- **Overlap semantics are half-open:**
  `chunk.char_start < span_end and span_start < chunk.char_end`.
- **Unanswerable questions are excluded from retrieval scoring, not
  zeroed.** They carry `scorable: False` with no metric keys at all, so 16
  rows of 0.0 cannot read as 16 retrieval failures.
- **One scoring path.** All retrieval arms go through
  `scripts/10_eval_retrieval.py`. A second harness would let arms drift
  apart in how they score, which is the failure class this project keeps
  finding in other guises.

---

## 4. Layout and script map

```
config/            baseline.yaml (bm25 since sweep 2), dense_phase5.yaml, sweep_*.yaml
data/
  raw/             PDFs                      [gitignored]
  interim/         extracted text            [gitignored, FROZEN]
  eval/            manifest + gold set       [COMMITTED — the deliverable]
docs/              build-log.md, dependencies.md, extraction.md,
                   paraphrase-protocol.md
results/tables/    ablation CSVs             [committed]
results/figures/   charts                    [created in Phase 8]
scripts/           numbered pipeline entry points
src/rageval/
  ingest/          arxiv_fetch.py, extract.py
  chunking/        base.py (chunk_document: fixed-size token windows only;
                   the `strategy` string is a label, not a dispatch)
  embed/           encoder.py (+ SQLite cache keyed on hash(model, text))
  store/           chroma_store.py (ChromaStore, Retrieved, all_records)
  retrieve/        bm25.py, fusion.py
  generate/        llm.py, prompts.py
  evaluation/      gold.py, retrieval_metrics.py
  config.py        CHROMA_DIR, CONFIG_DIR, EMBED_CACHE, EVAL_DIR, INTERIM_DIR, TABLES_DIR
  pipeline.py      RAGPipeline, parse_citations, detect_abstention
tests/             test_bm25.py, test_chunking.py, test_extract.py, test_gold.py,
                   test_pipeline.py, test_retrieval_metrics.py
                   (155 tests, all passing, 28 Oct 2026)
```

**`results/tables/retrieval_baseline.csv` holds the Phase 5 DENSE run**,
not the current `baseline.yaml` (BM25). The `baseline` tag must never be
overwritten: comparisons, the oracle ceiling and the truncation diagnostic
all read it. `10_eval_retrieval.py` defaults to `--tag baseline`, so always
pass `--tag`. To re-run dense, use `--config dense_phase5.yaml`.

Scripts in use (run `dir scripts` for the full list):

| script | purpose |
|---|---|
| `01_fetch_corpus.py` | arXiv search and PDF download |
| `02_extract_text.py` | PDF → `data/interim/` (FROZEN output) |
| `03_build_index.py` | chunk + embed + load a Chroma collection |
| `04_search.py` | ad-hoc query against the index |
| `05_ask.py` | end-to-end question → cited answer |
| `06_find_evidence.py` | quoted phrase → exact character span |
| `07_check_absence.py` | corpus-wide co-occurrence check for unanswerable candidates |
| `08_validate_gold.py` | span and quote mechanical validation |
| `09_add_question.py` | authors a gold question, reading quotes from frozen extraction |
| `10_eval_retrieval.py` | **the harness.** `--retriever dense\|bm25\|hybrid`, `--fusion rrf\|interleave` |
| `11_diagnose_truncation.py` | how much chunk text reaches the encoder |
| `12_oracle_query.py` | query the index with each span's own text → the ceiling |
| `13_compare_runs.py` | runs side by side, against the ceiling and at equal budget |
| `14_inspect_misses.py` | questions no run reaches, annotated with oracle rank |
| `15_selection_stability.py` | repeated stratified half-splits → win frequency, optimism |
| `16_paired_test.py` | McNemar exact + cluster bootstrap + Holm correction |
| `17_lexical_overlap.py` | the authoring-confound diagnostic |
| `18_paraphrase_worksheet.py` | emit / validate the paraphrase worksheet |
| `19_paraphrase_result.py` | **not yet written** — criterion fixed in `docs/paraphrase-protocol.md` |

Three unnumbered diagnostics also exist: `diagnose_chunks.py`,
`diagnose_extraction.py`, `verify_blob_strip.py`.

---

## 5. Current configuration

```yaml
chunking:  fixed_512, overlap 0        # 2,938 chunks, 130 papers, median 2,113 chars
embedding: all-MiniLM-L6-v2            # builds the index; max_seq_length 256 word pieces
index:     baseline_fixed512_minilm    # ChromaDB, cosine
retrieval: bm25                        # <- replaced dense in sweep 2
reporting: span recall at k=10 and at the 20,000-character budget
```

Gold set: **74 questions** — 20 factual, 14 comparative, 18 multi_hop,
16 unanswerable, 6 ambiguous. **58 scorable, 101 evidence spans, 33 papers.**
The other 97 papers are the distractor set.

Seed and sweep order for what remains:

```
seed:  fixed_512 / BM25 / k=10
order: reranking -> k -> (embedding model: DROPPED)
```

---

## 6. What has been measured

**Phases 0–5 complete. Phase 6 three sweeps in.**

### Dense baseline and its ceiling (Phase 5)

| | span level, k=10 |
|---|---|
| reachable by its own text (oracle) | 0.911 |
| reachable from the question | 0.436 |
| gap | 0.475 |

The oracle median rank when found is 1 and only 6 of 101 spans are
unreachable by either route. **The index, chunking, embedding and scoring
all work; the entire gap was question-to-passage matching.** Every
improvement since is reported against this headroom.

### Sweep 1 — chunk size: NULL

| | k=10 | budget 20k |
|---|---|---|
| fixed_256 | 0.347 | 0.455 |
| **fixed_512** | 0.436 | 0.416 |
| fixed_1024 | 0.416 | 0.287 |

512 vs 256: Holm p = 0.5034, CI [−0.040, +0.114]. 1024 significantly worse
(Holm p = 0.0089). Baseline unchanged. `fixed_128` dropped as underpowered.

**Finding:** comparing chunk sizes at fixed k measures *text volume*, not
chunking quality. 1024-token chunks hand the scorer ~4× the characters at
the same k. The ranking inverts between fixed-k and equal-budget, and the
inversion is stable across 4,000 half-splits (`fixed_1024` wins 30.8% at
k=10 and 1 of 4,000 at the 20k budget). The confound applies only to
comparisons that change chunk size — later sweeps hold chunking fixed.

### Sweep 2 — retrieval function: BM25 REPLACES DENSE

| arm | k=10 | budget 20k | misses@10 |
|---|---|---|---|
| dense | 0.436 | 0.416 | 20/58 |
| **bm25** | **0.604** | **0.584** | **9/58** |
| hybrid (RRF k=60) | 0.545 | 0.525 | 12/58 |

| comparison | delta | 95% CI | Holm p | win rate |
|---|---|---|---|---|
| bm25 vs dense | +0.168 | [+0.082, +0.253] | 0.0069 | 93.2% |
| hybrid vs dense | +0.109 | [+0.037, +0.186] | 0.0255 | 6.8% |
| bm25 vs hybrid | −0.059 | [−0.141, +0.020] | 0.2101 | — |

Dense was selected in **zero of 4,000 half-splits**. BM25's optimism is
+0.003. It closes 35.4% of the dense baseline's headroom and uses no neural
model, no GPU and no cache. Against hybrid it is not separable; the tie is
broken on parsimony.

Discordant split is **6/23** — BM25 found 23 spans dense missed; dense found
6 BM25 missed. So a perfect fusion caps at 0.644, only +0.059 above BM25
alone, which is inside this gold set's noise floor.

### Sweep 2b — fusion: NULL (exploratory, post-hoc)

RRF at k=60 with a pool of 100 is an **intersection ranker**: a document
ranked 50th by both scores 2/110 = 0.0182, one ranked *first* by one and
absent from the other scores 1/61 = 0.0164. Measured: 100.0% of the fused
top 10 was "both", 0.0% unique to either arm, and the fusion *lost* a
question BM25 alone had recovered.

Unique share of the fused top 10 as k falls: **0.0% → 8.8% → 15.3% →
18.8%** (k=60, k=10, k=1, interleave). Mechanism confirmed. Performance:
every arm within 0.663–0.683, every Holm p = 1.0000, 2,187 of 4,000
half-splits an exact tie. **Interleaving — which has no constant to tune —
scored 0.663, BM25's number exactly.**

**Fusion trades slots; it does not add them.** Every document promoted for
being unique displaces one that was there on merit. The union ceiling is
only collectable by increasing depth, which changes the budget.

### The authoring confound (bounds the headline number)

Every gold question was written while reading its evidence, which hands a
lexical retriever the exact tokens the author had. `17_lexical_overlap.py`
measures the IDF-weighted share of each question's vocabulary already
present in its evidence (median 0.468, range 0.037–1.000).

| tercile | dense | bm25 | delta | 95% CI |
|---|---|---|---|---|
| low overlap | 0.519 | 0.556 | +0.037 | [−0.115, +0.208] |
| mid | 0.382 | 0.618 | +0.235 | [+0.103, +0.385] |
| high | 0.375 | 0.575 | +0.200 | [+0.070, +0.325] |

Pre-registered criterion was low/high ≥ 0.50; measured **0.19**. Verdict
**BOUNDED**. The interpretation that BM25 and dense suit different question
styles was tested and **not established** — interaction CI [−0.049, +0.367].

**The sentence that survives every instrument:**

> BM25 outperforms the dense baseline by +0.168 span recall
> [+0.082, +0.253], Holm p = 0.0069, selected in 93% of half-splits. The
> advantage is not uniform: it is smallest among questions that share least
> vocabulary with their evidence (+0.037, CI [−0.115, +0.208]), where the
> two are not separable. Whether that variation is real is unresolved at
> n=58.

"Not separable" in the low tercile is **not** "equal" — 19 questions and 27
spans give an interval consistent with either direction.

### Pre-registered predictions, and how they landed

| prediction | outcome |
|---|---|
| Query-side axes move the numbers; chunking axes move them much less | **Confirmed** |
| RRF's k controls whether unique recall survives | **Confirmed** |
| The whole remaining fusion prize is +0.059, inside the noise floor | **Confirmed** |
| BM25 recovers f013, f015, m010, c002 | **Half** — f015 and m010 only |
| 512 vs 256 is a coin flip on half-splits | **Wrong in magnitude** — 84.4%, not ~50% |
| BM25's advantage survives among low-overlap questions | **Failed** |
| Dense and lexical retrieval suit different question styles | **Not established** |

f013 and c002 miss under every configuration tried — dense, lexical and
fused. They are the most interesting unexplained questions in the set.

---

## 7. Methodology rules

These are binding. Several were written down before the results that tested
them.

**The switching rule (pre-registered).** A configuration replaces the
baseline only when its **paired CI excludes zero AND its half-split win rate
is decisively above chance**. When either fails, the baseline carries
forward and the sweep is written up as a null result. This kept fixed-512
despite fixed_256 leading at every budget with an 84.4% win rate.

**Win frequency is not a significance test.** "Wins more often than not" is
far weaker than "differs detectably". A true difference of +0.039 with
SE ≈ 0.040 predicts a half-split win rate near 75%.

**The character budget is the primary metric.** Fixed k is reported
alongside it, and never used for selection where chunk size varies.

**The oracle ceiling is retriever-specific.** Under a lexical retriever the
oracle queries with a span's own text and matches itself, so BM25's ceiling
returns ≈1.0 and restates its recall. `13_compare_runs.py` printing `-` for
bm25/hybrid ceilings is correct, not a missing file. Report gains against
the *dense baseline's* measured headroom.

**Holm-correct across every pair tested together.** Seventeen planned runs
at α = 0.05 makes a spurious result near-certain.

**Cluster bootstrap resamples questions, not spans.** Spans within a
question are correlated — usually the same paper, often adjacent chunks.

**Coverage, not Recall, for cross-type comparison.** Recall@k asks whether
*any* span was found, so two-span questions get two chances. This confounded
a Phase 5 hypothesis before it was caught.

**nDCG takes the true relevant-chunk count.** Inferring the ideal from what
was retrieved flatters an empty run.

**Budget columns are nested subsets of one ranked list, not independent
trials.** "Consistent across four budgets" was an over-read and was
corrected.

**Pre-register with intervals, not point ratios.** The overlap criterion put
a hard threshold on a ratio of two noisy estimates; the denominator's own CI
[+0.070, +0.325] means the ratio could read 0.11 to 0.53 from denominator
variation alone. Recorded as a defect in the pre-registration, honoured as
written. Future criteria name an interval.

**The power limit.** Detecting +0.039 at 80% power needs SE ≈ 0.014 against
the actual 0.040 — roughly **8× the spans, on the order of 450–500
questions**. This gold set cannot resolve a four-point difference, and no
amount of resampling changes that. Say so rather than re-running.

**Run the test whose two outcomes point in opposite directions.** The oracle
test settled in one run what three plausible mechanisms could not.

---

## 8. In flight: the paraphrase validity test

The only remaining work that *lifts* the bound on the headline number rather
than refining it. Protocol pre-registered and committed:
`docs/paraphrase-protocol.md`. Read it before touching anything here.

**Design.** Rewrite the 20 highest-overlap questions to ask the same thing
in different words; re-score dense and BM25 on both phrasings. Within
question, so evidence, difficulty and type are held constant by
construction — far more power than the between-group tercile split it
replaces. `18_paraphrase_worksheet.py --build` writes
`data/eval/gold_paraphrased.jsonl`: all 74 questions, with only those 20
having new text, every other field byte-identical.

**State of `data/eval/paraphrase_worksheet.txt`:**

- **7 blocks filled and passing:** a006, m002, a005, f009, m007, m008, m009
- **8 need small edits** to drafts already written: c007, f018, f019, c009,
  c012, c011, m001, m016 — mostly adding a `keep:` or replacing a dangling
  "the model" with a name or description
- **5 need writing fresh:** a002, a004, m003, m010, m015

**Rules the owner is working to.** Banned terms are *computed* (the shared
terms carrying the top 80% of overlapping IDF mass, skipping anything in
>20% of the corpus), compliance is checked mechanically, and the worksheet
deliberately carries **no retrieval results** so a rewrite cannot be aimed.
`--build` writes nothing unless all 20 pass, so it is safe to run as a
linter at any time.

**Do not write the paraphrases.** You know which arm each rewrite favours;
that is exactly the bias the test exists to remove. Reviewing for banned
terms, dangling referents and answer leakage is fine and has been done
before — note in the log that any such review happened *without* access to
per-question retrieval results.

**The three failure modes that matter**, none caught by the script:

1. **Changing what is asked.** "Why did X need a new loss?" is not a
   paraphrase of "which loss did X introduce?".
2. **Dangling referents.** The retriever sees the question text alone. "In
   that model's paper…" points at nothing and the referring sentence — hop 1
   — is no longer described by anything in the query.
3. **Answer leakage.** Dodging the banned word by naming the answer
   instead. "reduce the number of temporal aggregations" *is* m010's answer.

**`keep:` needs every sub-token.** The tokenizer emits a compound plus its
parts: `FuXi` → fuxi, fu, xi · `GraphCast` → graphcast, graph, cast ·
`GenCast` → gencast, gen, cast. A bare `keep: fuxi` still fails on `fu` and
`xi`.

**Manipulation check, declared in advance:** mean IDF-weighted overlap must
drop by **≥ 0.25 absolute**. Below that the rewrites did not remove enough
vocabulary — rewrite further; do not lower the threshold.

**Guard, declared in advance:** if dense span recall on the rewritten
questions falls by more than 0.15 against the same questions' originals,
treat it as a broken-question warning, re-read every rewrite, and report the
warning whatever the primary result says.

**Then write `19_paraphrase_result.py`.** Its criterion is already fixed in
the protocol as a three-way interval rule, including an explicit
"inconclusive" outcome — at 20 questions and ~40 spans that is a likely
result and must be admissible in advance, so a wide interval is not written
up afterwards as evidence of no effect.

---

## 9. What is left

**Phase 6 remainder, in order:**

1. **Paraphrase test** (§8) — finish worksheet, `--build`, two harness runs
   with `--gold data/eval/gold_paraphrased.jsonl`, then
   `19_paraphrase_result.py`.
2. **Sweep 3 — reranking.** Cross-encoder over BM25 candidates. The
   remaining axis where a large effect is most plausible: it scores query
   and passage together instead of comparing two independently-made vectors.
   Latency is a finding, not a problem to hide.
3. **Sweep 4 — value of k** (1, 3, 5, 10, 20). Phase 7 needs this
   regardless: **zero of 32 two-span questions had both passages retrieved
   at k=5.** Generation must be fed a larger k and the number must be
   defensible.
4. **Chunk-quality filter ablation** (off / 0.3 / 0.4 / 0.5). An outstanding
   commitment from Phase 2b, which deferred it rather than applying a silent
   filter. Cheaper now — BM25 needs no embeddings.
5. **WRITE section-aware, recursive and semantic chunkers — they do not
   exist — then sweep them under BM25.** `chunk_document` produces only
   fixed-size token windows; sweep 1 covered sizes of that one strategy.
6. **End-of-phase replication:** winning pipeline × {fixed_512, fixed_256}
   **under BM25**. Sweep 1 compared chunk sizes under *dense* retrieval, and
   BM25's length normalisation (b = 0.75) behaves nothing like an encoder
   that truncates at 256 word pieces — so that null does not transfer
   automatically. This is the only interaction effect actually measured.
7. **Hoist `_count_spans` into `rageval`.** Four private copies now (13, 15,
   16, 17). The pandas-dtype bug it guards against already produced one
   wrong results table; four copies is how a fix lands in three of them.

**Dropped, with reasons recorded:**

- **Embedding-model sweep** — dense contributes unique documents but no
  measurable recall, so this would sweep a component not in the pipeline.
  Executed per the condition written down after sweep 2.
- **`fixed_128`** — underpowered once 256 and 512 proved indistinguishable.

**Scope not met:** the build plan's six-chunker chunking axis. Only one
chunker exists (fixed-size token windows, swept at three sizes). Item 5
above is the work that closes it; until then any claim about "chunking
strategy" is a claim about chunk size.

**Then:** Phase 7 (generation evaluation — the stated headline, still
untested), Phase 8 (analysis and write-up), Phase 9 (fresh-clone
reproducibility, `v1.0` tag). **Project 2** (text-to-SQL) begins the week
after the tag, not before.

**Unresolved:** whether the cu128 CUDA wheel was ever installed. If it was,
`docs/dependencies.md` and `requirements.lock.txt` are stale, and the
embedding cache — keyed only on `hash(model, text)` — may hold a mix of CPU-
and GPU-computed vectors, which are not bit-identical. Clearing the cache
and rebuilding makes the question moot.

---

## 10. The bug museum

Every one of these was found the hard way. Several will recur.

**pandas dtype inference silently zeroes span counts.** The `hit_spans`
columns hold strings like `"0;1"`, but a column containing only `"0"` and
blanks is inferred as `float64`, so `"0"` returns as `0.0` and a bare
`.isdigit()` counts nothing. Produced an entire wrong budget table. Guard is
`_count_spans`, which normalises the `.0` suffix.

**Retrieval depth caps the budget view.** The first sweep ran `--k-max 20`
and `fixed_256` consumed exactly `20.0` chunks at the 40k budget — it had
hit the retrieval limit, not the budget. **Always run `--k-max 50`.** An
exactly-round number in a column of decimals is the tell.

**Windows cp1252 mangles quotes copied through the terminal.** Raised
`UnicodeEncodeError` and silently corrupted non-ASCII mathematical
characters; the gold file had to be rebuilt. Fixed with
`sys.stdout.reconfigure(encoding="utf-8")` and then by removing the copying
entirely. **Never route gold text through a console round-trip.** Save
worksheets and gold files as UTF-8.

**A stale `--out` file read as a successful result.** `06_find_evidence.py`
returned early on no match without removing its previous output, so three
comparative questions (c006–c008) were authored with xLSTM evidence attached
to GraphCast/FuXi questions. Mechanical validation passed — the spans and
quotes were internally consistent, just from the wrong paper. Caught only
because `papers covered` dropped from 6 to 5. **Unlink an output path before
writing to it.**

**"Valid but wrong" evidence recurs and no mechanical check finds it.**
f009, f015, f019, c002, c006–c008 all validated cleanly while not supporting
their answers. Only per-question manual review catches it.

**Extraction artifacts break quote lookups.** Line-break joins drop spaces
and hyphens — the extracted text reads `datadriven`, `namethe`,
`theprevious`. Quote short fragments that avoid hyphenated compounds and
word boundaries near line breaks.

**Wrong CUDA wheel.** The RTX 5070 is Blackwell (sm_120) and needs cu128+.
`cu121` installs cleanly, reports `cuda True`, then fails with "no kernel
image is available" — a failure mode where the obvious check passes.

**`ragas` imports a module `langchain-community` deleted.** Pin
`langchain-community==0.4.1` and add `pillow` explicitly; both pins carry
explanatory comments in `pyproject.toml`. Never bare-upgrade mid-project.

**Encoder truncation is real but was not the cause.** `all-MiniLM-L6-v2`
accepts 256 word pieces against chunks of median 2,113 characters, so
**53.1% of chunk text never reaches the encoder.** 42 of 57 missed spans
were nonetheless fully visible. It caps the ceiling; it did not explain the
misses. It also means comparing embedding models at 512-token chunking would
confound model quality with context window.

**Pipeline ordering destroyed document structure** (Phase 2). `clean_text`
collapsed newlines before `split_sections` and `strip_references` ran, both
of which match whole lines — so 100% of the corpus reported degraded from
one cause. Then the fix cut from References to end-of-document and deleted
every appendix, including the single most relevant paper's, while reporting
94.6% clean.

**Abstention detection counted formatting as behaviour** (Phase 3). The
naive prompt refuses in prose; the sentinel-only matcher recorded that as
`abstained=False`. Had Phase 7 computed hallucination as "did not abstain",
the naive arm would have scored as hallucinating on a question it correctly
refused. Detection now covers prose refusals, and at Phase 7 scale (48
answers) **every label is hand-verified**.

---

## 11. Standing lessons

- **A plausible number is the one that does not get audited.** The first
  extraction run reported 100% degraded and was obviously broken. The second
  reported 94.6% clean and looked finished while deleting a third of some
  papers.
- **Verification belongs before the freeze**, and *"the symptom
  disappeared"* is not the same claim as *"the fix is correct."*
- **A weak baseline is not a broken one.** Three hypotheses were built on
  calling factual Recall@10 = 0.700 anomalous. A 22M-parameter
  general-purpose encoder on dense technical prose is plausibly just that
  weak.
- **An aggregate tells you something is wrong; it almost never tells you
  what.** A plausible mechanism that fits the aggregate is not evidence.
- **The control arm was worth more than the treatment arms.** Three RRF
  variants would have produced a ranking and a story. Interleaving — nothing
  to tune, nothing to get right — landed on BM25's exact score and made the
  ranking legible as noise.
- **A ratio of two noisy estimates is not a criterion.**
- **Every corruption in the gold phase entered through a human copying
  text.** The fix was not more careful copying but removing the copy: the
  authoring tool reads quotes from the frozen extraction and the only thing
  typed by hand is a pair of integers.
- **Check what is actually in the corpus before designing around it.** Two
  rounds of multi-hop chain-hunting were spent on models the corpus
  discusses constantly but does not contain (FengWu, FourCastNet,
  NowcastNet, SwinVRNN, DiffCast).
- **Two numbers can both be right and point opposite ways.** p = 0.503 and
  an 84.4% win rate are the same fact described by instruments of different
  sensitivity. The temptation is to quote whichever supports the switch.

---

## 12. Commands

```powershell
cd "C:\RAG Project\rag-evaluation-lab"
python -m pytest -q

# a retrieval arm (always --k-max 50)
python scripts\10_eval_retrieval.py --config sweep_bm25.yaml --tag bm25 --k-max 50
python scripts\10_eval_retrieval.py --config sweep_hybrid.yaml --tag hybrid_interleave --k-max 50 --fusion interleave

# compare, test, check stability
python scripts\13_compare_runs.py baseline bm25 hybrid --k 20
python scripts\16_paired_test.py baseline bm25 hybrid
python scripts\15_selection_stability.py baseline bm25 hybrid --splits 2000

# diagnostics
python scripts\12_oracle_query.py --tag bm25 --k 50
python scripts\14_inspect_misses.py --tags baseline bm25 --k 20
python scripts\17_lexical_overlap.py

# the paraphrase test
python scripts\18_paraphrase_worksheet.py
python scripts\18_paraphrase_worksheet.py --build
```

**Commit protocol.** Imperative subjects under 60 characters; bodies explain
*why*. Commit at phase boundaries and meaningful sub-steps, spread across
real days — target 40–60 commits. Never commit API keys, PDFs, Chroma
directories or `mlruns/`. Always commit the gold set, results tables,
figures and the lockfile. Commit a pre-registration **before** the run it
governs — a protocol landing in the same commit as its result is not a
pre-registration. Tag `v1.0` only after the Phase 9 fresh-clone check
passes.

---

## 13. Documentation duties

These are not tidy-up for later. A sweep is not closed until all of this is
done, and the next session inherits whatever you leave behind.

**After every sweep, ablation, or significant change, in this order:**

1. **`docs/build-log.md`** — append a section: what was built, what was
   measured (with CI, Holm p and win rate), what was decided, what went
   wrong, and the standing lesson. Falsified predictions stay in with their
   original wording. Never delete a superseded number — supersede it in
   place and say what it was and why it changed.

2. **`CLAUDE.md` itself.** Update the **Status date** line, §5 if the
   baseline configuration moved, §6 with the new result, §9 to strike what
   is done and re-order what remains, §4 if a script was added, and §10 if a
   new bug class was found. A stale CLAUDE.md is worse than none: the next
   session will act on it.

3. **`results/tables/`** — commit the CSVs the claims rest on. A number in
   the build log with no table behind it is not reproducible, and
   reproducibility is one of the three completion criteria.

**Other docs, when triggered:**

- **A new pre-registration gets its own file** — `docs/<name>-protocol.md`,
  following `docs/paraphrase-protocol.md` as the pattern: purpose, design,
  selection rule, manipulation check, primary analysis, and the criterion
  stated as an interval with every outcome (including "inconclusive")
  admissible in advance. **Commit it before the run it governs.** A protocol
  landing in the same commit as its result is not a pre-registration.
- **`docs/dependencies.md`** — on any pin, wheel, driver or environment
  change, including resolving the open CUDA question in §9. Record what
  changed and why, with the symptom it fixes.
- **Plan changes** — recorded in the build log's `## Plan amendments`
  section, never silently. Dropping a planned experiment because the
  pipeline moved is a result, not an omission, and gets written up as one.
- **Claims in this file about the repo must be verified against the repo,
  not trusted.** CLAUDE.md was written from a chat session and drifted
  within days: it referenced three files that had never been committed and
  one with a misspelled name. When you read a path here, check it exists
  before acting on it, and correct this file when it does not. The project's
  own standing lesson applies to its documentation: a plausible claim is the
  one that does not get audited.

**If CLAUDE.md and `docs/build-log.md` disagree, the build log wins.** It is
the primary record; CLAUDE.md is a summary of it. Fix the summary.

---

## 14. Do not do these without asking

All of these are things a capable agent will do helpfully and should not.

- **Rebuild an index or clear the embedding cache.** Both are slow and the
  cache's provenance is already an open question (§9).
- **`git push`.** Commit freely; the owner pushes.
- **Re-run the full pipeline to "check" something.** Ask which single arm
  would settle it.
- **Add, remove or reword a gold question.** See §3.
- **Write the paraphrases.** See §8.
- **Change a pre-registered criterion after seeing a result**, including
  "clarifying" a threshold. Record the defect and honour the rule as
  written; that has already happened once and is in the log as such.
- **Install or upgrade a dependency** beyond what `requirements.lock.txt`
  pins.
- **Never run 01_fetch_corpus.py or 02_extract_text.py.** data/interim/ is
  frozen and gold evidence is anchored to its exact byte offsets.
  Regenerating it silently invalidates all 101 spans. In a cloud session
  with no data/, the correct response to a missing index is to say so and
  stop, not to rebuild it.

## 15. Known gaps in this file

- **MLflow.** The build plan's Phase 6 acceptance criterion is "17 logged
  runs, re-runnable by run ID". What exists instead is tagged CSVs in
  `results/tables/`, which is arguably better for reproducibility but is a
  departure from a stated criterion that no document records. Either adopt
  MLflow or write the deviation into the build log's `## Plan amendments` —
  do not leave it undecided.
- **Phase 7 operational detail**, which matters because it is next after
  Phase 6:
  - **RAGAS is the cost multiplier, not generation** — one answer can cost
    five or more LLM calls. Set a hard spend cap in the OpenAI dashboard
    before the first run and dry-run on 5 questions.
  - Responses cache on the full request (model, temperature, system and user
    prompt), so any prompt change is a deliberate cache miss.
  - The three prompt strategies differ in exactly one respect by design, so
    the comparison isolates the abstention instruction. Keep it that way.
  - **Generation must be fed k > 5.** Zero of 32 two-span questions had both
    passages retrieved at k=5, so a hallucination measured at k=5 would be a
    retrieval artifact wearing a generation costume. The k used must be
    reported.
  - All 48 abstention labels are hand-verified. The regex is a first pass.
- **Environment bootstrap** is not described here; `docs/dependencies.md` is
  the reference, and the two pins in `pyproject.toml` carry their own
  explanatory comments.
