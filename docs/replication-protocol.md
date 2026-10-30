# End-of-phase replication, chunk size under BM25: pre-registration

Written before the replication runs exist. Nothing below is revised
after seeing a result; if the design proves wrong, the flaw is recorded
next to the outcome.

## What it tests

Sweep 1 compared chunk sizes under **dense** retrieval and found
`fixed_256` vs `fixed_512` not separable at the 20k budget (+0.039,
CI [−0.040, +0.114], Holm p 0.5034, `fixed_256` winning 84.4% of
half-splits). Sweep 2 then replaced dense retrieval with BM25. BM25's
length normalisation (b = 0.75) behaves nothing like an encoder that
truncates at 256 word pieces, so sweep 1's null does not transfer
automatically. Every later sweep held `fixed_512` fixed.

**Question.** Under the pipeline Phase 6 selected (BM25, `fixed_512`,
no filter), does `fixed_256` change span recall at the 20k budget?

Secondary, reported only: **is the size effect different under BM25
than under dense?** This is the one interaction between two axes that
the phase can estimate, because both sizes have now been run under both
retrievers on the same questions.

## Design

| tag | retriever | chunks | status |
|---|---|---|---|
| `bm25` | BM25 | `fixed_512` (`baseline_fixed512_minilm`) | existing table, control |
| `bm25_chunk256` | BM25 | `fixed_256` (`sweep_fixed256_minilm`) | **new, decides** |
| `baseline` | dense | `fixed_512` | existing, sweep 1/Phase 5 |
| `chunk256` | dense | `fixed_256` | existing, sweep 1 |
| `rerank` | BM25 + bge-reranker-v2-m3 | `fixed_512` | existing, sweep 3 |
| `rerank_chunk256` | BM25 + bge-reranker-v2-m3 | `fixed_256` | **new, exploratory** |

**The 256 collection is the right one.** Rebuilding `fixed_256` from the
frozen extraction reproduces the collection's 5,816 chunk IDs exactly.
BM25 builds from the collection's stored text, so no index rebuild and
no embeddings are involved.

**Depth.** The new 256-token arms run at `--k-max 100`: the 20k budget
needs about twice as many 256-token chunks, and sweep 1's lesson is that
a budget column must never be capped by retrieval depth. The 512 arms ran
at 50, which their budgets never reach (max 22 chunks at 40k).

*Found while checking:* sweep 1's dense `chunk256` table reaches exactly
50 chunks at the **40k** budget for 1 of 58 questions, so that one cell is
depth-capped. The 20k column used here peaks at 27 and is unaffected. No
decision has used the 40k column.

**Exploratory arm.** The reranker rescores BM25's top 100 chunks. At 256
tokens that is about half as much candidate text as at 512, so the
reranked comparison changes chunk size and candidate volume together. It
decides nothing.

## The scoring bias, which this time favours the challenger

A span counts as found if any retrieved chunk overlaps it, so a span cut
by a boundary can be found two ways (chunker sweep).

| chunking | gold spans cut by a boundary | overlapping chunks per question (median) |
|---|---|---|
| `fixed_512` | 62 of 101 | 3 |
| `fixed_256` | **80 of 101** | 3 |

**This favours `fixed_256`.** A gain for `fixed_256` is therefore partly
expected from the scoring rule alone; a null has held up against it.

## Primary analysis and rule

Span recall at the **20,000-character budget**, `bm25_chunk256` against
`bm25`, 58 scorable questions. `16_paired_test.py bm25 bm25_chunk256`
(cluster bootstrap over questions, 10,000 resamples, seed 20260827) and
`15_selection_stability.py bm25 bm25_chunk256` (2,000 stratified
half-splits × 2 directions).

The shared decision table (`rageval.evaluation.decision`), win-rate
threshold 90%:

| outcome | condition | what happens |
|---|---|---|
| **Replaces** | CI on (256 − 512) excludes zero, positive, **and** win rate ≥ 90% | `fixed_256` becomes the baseline chunk size |
| **Leaning** | CI excludes zero, positive, win rate < 90% | `fixed_512` stays; reported as a direction |
| **Not separable** | CI includes zero | `fixed_512` stays |
| **Worse** | CI excludes zero, negative | `fixed_512` stays; reported as a measured harm |

"Excludes zero" is strict. k=10 is not reported for this comparison:
chunk size changes, and fixed k measures text volume (sweep 1).

## Reported, deciding nothing

Computed by `25_replication_result.py` on the same questions and the same
bootstrap draws:

- the size effect under BM25 (the primary delta, recomputed);
- the size effect under dense, from sweep 1's tables;
- **the interaction**: (256 − 512 under BM25) − (256 − 512 under dense),
  per question, with its CI;
- exploratory: reranked BM25 at 256 against reranked BM25 at 512.

## Predictions, recorded before the run

1. **Not separable.** Sweep 1's dense difference was +0.039 with a CI
   twice that wide, and this gold set resolves about 8 points.
2. **The point estimate is positive.** The scoring bias favours
   `fixed_256`, and so did sweep 1's dense estimate.
3. **The interaction CI includes zero.** Two noisy differences on 58
   questions.
4. **`fixed_256`'s win rate lies between 50% and 90%**, as it did under
   dense (84.4%).

## Threats not controlled

- The scoring bias above, favouring `fixed_256`.
- Two sizes only; `fixed_1024` was significantly worse under dense in
  sweep 1 and is not re-run.
- The interaction combines four runs, two of them (dense) from sweep 1
  and two from now. The gold set, extraction and scoring path are
  unchanged since sweep 1, which the regression checks of sweeps 3 and 4
  support, but the runs are months apart in the project's history.
- The exploratory arm confounds chunk size with candidate volume.

## Procedure

```powershell
python scripts\10_eval_retrieval.py --config sweep_chunk256.yaml --retriever bm25 --tag bm25_chunk256 --k-max 100
python scripts\10_eval_retrieval.py --config sweep_chunk256.yaml --retriever bm25 --tag rerank_chunk256 --k-max 100 --rerank BAAI/bge-reranker-v2-m3 --rerank-depth 100 --rerank-max-length 1024
python scripts\16_paired_test.py bm25 bm25_chunk256
python scripts\15_selection_stability.py bm25 bm25_chunk256 --splits 2000
python scripts\25_replication_result.py
```

Both commands were exercised on one question (`--limit 1`) before this
protocol was committed.
