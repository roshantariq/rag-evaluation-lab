# Chunker sweep, section-aware / recursive / semantic: pre-registration

Written before any chunker run exists. Nothing below is revised after
seeing a result; if the design proves wrong, the flaw is recorded next to
the outcome. The literature behind the design is in the build log,
"Literature check, before the chunker sweep" (`14fd7e3`).

## What it tests

Until now the project had one chunker, fixed-size token windows, and
sweep 1 varied only its size. The build plan's chunking axis asks a
different question: **at the same chunk size, does where the cuts fall
change span recall?** Three chunkers, each differing from `fixed_512` in
that one respect:

| chunker | cuts at | module |
|---|---|---|
| `fixed_512` (control) | every ~512 tokens, at a word edge | `chunking/base.py` |
| section-aware | section headings, never crossing one; long sections split into near-equal windows; a section under a quarter of the target joins the next | `chunking/strategies.py` |
| recursive | the strongest boundary near the size limit: paragraph break, line break, sentence end, then word; pieces merged greedily up to the limit | `chunking/strategies.py` |
| semantic | the sentence boundary, within a size window, where adjacent sentences are least similar | `chunking/strategies.py` |

Common to all: cuts fall only between whitespace-delimited words, so
`text[char_start:char_end] == chunk.text` holds by construction; every
chunker tiles the document (no overlap, no word lost); sizes are counted
with the same per-word token costs; overlap is 0.

**Recursive gets a sentence level.** The common splitter's separators are
paragraph, line, space. This extraction keeps few line breaks (median 12
paragraph breaks per paper), so without a sentence level the splitter
would fall straight to word windows and be the fixed chunker under
another name.

**Semantic, specified in full.** Sentences are split at `.`, `!` or `?`
followed by an upper-case word, except after a listed abbreviation
(`et al.`, `Fig.`, `Eq.`, `e.g.` and others in `strategies._ABBREV`) or a
single-letter initial. A sentence over 256 tokens (a table or equation
with no full stop) is split into near-equal windows so no unit exceeds the
embedder's 512 word-piece input; on this corpus none did. Each unit is
embedded with **`BAAI/bge-large-en-v1.5`** (normalised, no instruction
prefix). The distance between neighbours is 1 − cosine. From each chunk
start, every cut leaving a chunk of 0.5× to 1.5× the target is a
candidate; the largest distance wins, ties to the earliest. If one long
unit jumps the window, the first cut reaching 0.5× is taken.

The embedder was chosen from published MTEB scores, not by trying several
on the gold set (sweep 3's rule): it scores 83.11 on STS against ~78.9 for
the index's MiniLM, and Qu et al. (2024) used it. The window keeps size
under control, because a relative threshold lets a semantic chunker's
size drift (Smith & Troynikov, 2024).

## Size matching, done on the corpus alone

Sweep 1 showed that comparing chunkers of different sizes measures size.
Each chunker's target was therefore set, before any run, to the value on
a 10-token grid whose **median chunk length is closest to `fixed_512`'s
2,113 characters**. The gold set played no part.

| chunker | grid searched | target | chunks | median chars | ratio to fixed_512 |
|---|---|---|---|---|---|
| `fixed_512` | — | 512 | 2,938 | 2,113 | 1.000 |
| section | 512–710 | **680** | 2,859 | 2,123 | 1.005 |
| recursive | 512–710 | **650** | 3,040 | 2,118 | 1.002 |
| semantic | 440–530 | **520** | 2,753 | 2,121 | 1.004 |

Medians match within 0.5%. The distributions are not identical:
structure-aware chunks vary more in length (p5–p95 in characters:
section 846–2,967, recursive 464–2,897, semantic 1,055–3,252, against
fixed windows that are nearly uniform). Recursive makes the most short
chunks, because it stops at the last boundary under the limit. That is a
property of the chunkers, not a defect of the matching.

## Built and checked before the runs

- **Chunk sets are committed artifacts.** `23_build_chunks.py` writes each
  set's spans to `results/tables/chunks_<name>.csv`, and the harness
  scores that table (`--chunks <name>`). Semantic boundaries come from GPU
  embeddings; the semantic set was built twice in separate processes and
  was identical (2,753 chunks, no differing ID), but the committed spans
  are the reference.
- **Tiling verified on all 130 papers for every set**: each chunk starts
  at the next uncovered word and ends at a word edge, and every word is
  covered. Randomised tests check the same for every chunker.
- **The new scoring path is equivalent to the old.** `fixed_512` written
  as a chunk set and scored with `--chunks` reproduced the committed
  `retrieval_bm25.csv` cell for cell.
- **`chunk_document` dispatch** keeps every `fixed_*` config unchanged:
  rebuilding `fixed_512` from the extraction still yields exactly the
  index's 2,938 chunk IDs.
- **Every new option was exercised** on one question (`--limit 1`) and
  `--chunks` refuses dense mode.

## A bias the scoring rule carries, known before the run

A span counts as found if any retrieved chunk overlaps it. A span cut by
a chunk boundary overlaps two chunks, so it can be found two ways.

| chunker | gold spans cut by a boundary | chunks overlapping each question's evidence (median) |
|---|---|---|
| `fixed_512` | **62 of 101** | 3 |
| section | 26 | 2 |
| recursive | 38 | 2 |
| semantic | 31 | 2 |

Keeping evidence whole is what structure-aware chunkers are for, and the
scoring rule rewards the opposite. **This bias favours `fixed_512`.** A
null therefore does not show structure is worthless. A structure-aware
chunker that wins anyway has won against the bias.

*Unexplained:* with spans of median 830 characters and chunks of about
2,100, a randomly placed boundary would cut roughly 40% of spans, not 62%.
One possibility is that some evidence was located through retrieval over
the `fixed_512` index during authoring. It is recorded, not explained.

## Primary analysis and rule

Span recall at the **20,000-character budget**, each chunker against
`bm25` (= `fixed_512`), 58 scorable questions. `16_paired_test.py` on
`bm25 chunk_section chunk_recursive chunk_semantic` (cluster bootstrap
over questions, 10,000 resamples, seed 20260827; McNemar with Holm across
the six pairs) and `15_selection_stability.py` on the same four runs
(2,000 stratified half-splits × 2 directions).

The shared decision table (`rageval.evaluation.decision`):

| outcome | condition | what happens |
|---|---|---|
| **Replaces** | CI on (chunker − `fixed_512`) excludes zero, positive, **and** win rate ≥ 90% | That chunker becomes the baseline chunking |
| **Leaning** | CI excludes zero, positive, win rate < 90% | `fixed_512` stays; reported as a direction |
| **Not separable** | CI includes zero | `fixed_512` stays; "not separable" is not "equal" |
| **Worse** | CI excludes zero, negative | `fixed_512` stays; reported as a measured harm |

"Excludes zero" is strict. At most one arm can reach a 90% win rate.
k=10 and k=20 are reported and decide nothing. `24_chunker_result.py`
applies the table; it and its tests are committed with this protocol.

## Predictions, recorded before the run

1. **All three chunkers are not separable from `fixed_512`.** Published
   differences at matched size are 1–2 recall points; this gold set
   resolves about 8.
2. **No chunker's point estimate exceeds `fixed_512` by more than
   +0.04** at the budget.
3. **No arm reaches a 90% win rate.**

## Threats not controlled

- The overlap bias above, which favours the control.
- One target per chunker, set by median length. Matching a different
  statistic (mean, or the whole distribution) is not tested.
- Section boundaries depend on the Phase 2 extractor's heading detection;
  a missed heading is a section-aware chunk that crosses a section.
- The sentence splitter is a rule, not a model, and will misfire on some
  abbreviations and references.
- BM25 only. Under dense retrieval, chunk boundaries also change what the
  encoder sees, a different mechanism not tested here.
- 58 questions; see prediction 1.

## Procedure

```powershell
python scripts\23_build_chunks.py --strategy section --target 680
python scripts\23_build_chunks.py --strategy recursive --target 650
python scripts\23_build_chunks.py --strategy semantic --target 520
python scripts\10_eval_retrieval.py --config sweep_bm25.yaml --tag chunk_section --k-max 50 --chunks section
python scripts\10_eval_retrieval.py --config sweep_bm25.yaml --tag chunk_recursive --k-max 50 --chunks recursive
python scripts\10_eval_retrieval.py --config sweep_bm25.yaml --tag chunk_semantic --k-max 50 --chunks semantic
python scripts\16_paired_test.py bm25 chunk_section chunk_recursive chunk_semantic
python scripts\16_paired_test.py bm25 chunk_section chunk_recursive chunk_semantic --k 10
python scripts\16_paired_test.py bm25 chunk_section chunk_recursive chunk_semantic --k 20
python scripts\15_selection_stability.py bm25 chunk_section chunk_recursive chunk_semantic --splits 2000
python scripts\24_chunker_result.py
```

The three chunk sets are already built and are committed with this
protocol; the first three commands regenerate them.

## References

- Qu, R., Tu, R. & Bao, F. (2024). *Is Semantic Chunking Worth the
  Computational Cost?* arXiv:2410.13070. <https://arxiv.org/abs/2410.13070>
- Smith, B. & Troynikov, A. (2024). *Evaluating Chunking Strategies for
  Retrieval.* Chroma technical report.
  <https://www.trychroma.com/research/evaluating-chunking>
- BAAI. *bge-large-en-v1.5* and the MTEB English table on the
  *bge-base-en-v1.5* model card (vendor-published).
  <https://huggingface.co/BAAI/bge-base-en-v1.5>
- Muennighoff, N., Tazi, N., Magne, L. & Reimers, N. (2023). *MTEB:
  Massive Text Embedding Benchmark.* EACL 2023. arXiv:2210.07316.
