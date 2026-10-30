# Chunk-quality filter ablation: pre-registration

Written before any filtered run exists. Nothing below is revised after
seeing a result; if the design proves wrong, the flaw is recorded next to
the outcome.

## What it tests

Phase 2b found chunks that are mostly numbers and symbols: chart axis
labels, table-of-contents leaders, results tables and equations. Only the
first two are noise. A single threshold cannot separate them, so Phase 2b
removed base64 only and deferred the rest to this ablation instead of
applying a silent filter. The motivating failure was a dense query for
"fraction skill score" whose top five were all axis labels.

**Question.** Does removing low-prose chunks from the BM25 index raise
span recall, by enough to make the filter part of the baseline?

The score is `prose_ratio`: the share of a chunk's whitespace tokens that
are ordinary words (three or more letters, after stripping punctuation).
It was defined in Phase 2b's `diagnose_chunks.py` and is moved unchanged
to `rageval/chunking/quality.py`. Recomputed from the live index, it
matches the committed `chunk_quality.csv` to three decimals on every
chunk the two share.

## Design

| arm | tag | filter | chunks removed |
|---|---|---|---|
| control | `bm25` | off | 0 — existing table, not re-run |
| 1 | `filter030` | prose ratio < 0.3 | 71 (2.4%), 29 papers |
| 2 | `filter040` | prose ratio < 0.4 | 127 (4.3%), 41 papers |
| 3 | `filter050` | prose ratio < 0.5 | 211 (7.2%), 62 papers |

The thresholds are Phase 2b's, fixed then. 0.6, which would catch most
axis labels, was ruled out in Phase 2b because it removes 15.4% of the
index across 99 papers, results tables included.

**The filter acts on the BM25 index, not on the result list.** Removed
chunks are dropped before BM25 is built (`10_eval_retrieval.py
--min-prose-ratio`), as a filter at ingestion would. IDF and average
length are recomputed over the survivors. Chunking, collection, gold set
and scoring path are unchanged. Removed chunks that held gold evidence
still count as relevant, so a filter that deletes evidence is charged for
it.

**What the filter costs the gold set, known before any run.** Measured by
overlap of the 101 gold spans with the removed chunks:

| threshold | spans made unreachable | spans losing one of several chunks |
|---|---|---|
| 0.3 | 0 | 0 |
| 0.4 | 0 | 1 (m005 span 1) |
| 0.5 | **1 (f011 span 0; its only chunk scores 0.44)** | 2 (m002 span 1, m005 span 1) |

At 0.5, f011 can no longer be retrieved by construction. That is a real
cost of the filter and is scored as one.

**Mechanism check, recorded per question.** With the filter on, the
harness also builds unfiltered BM25 and counts how many of each
question's unfiltered top 10 the filter removed (`low_prose_top10`). Those
are the only slots the filter can hand to other chunks. If they are near
zero, a null is explained before it is interpreted. A one-question smoke
run at 0.4 found one such slot.

## Primary analysis and rule

Span recall at the **20,000-character budget**, each filtered arm against
`bm25`, 58 scorable questions. Instruments as in sweep 3:
`16_paired_test.py` on `bm25 filter030 filter040 filter050` (cluster
bootstrap over questions, 10,000 resamples, seed 20260827; McNemar with
Holm across the six pairs), and `15_selection_stability.py` on the same
four runs (2,000 stratified half-splits × 2 directions).

| outcome | condition | what happens |
|---|---|---|
| **Replaces** | CI on (arm − `bm25`) excludes zero, positive, **and** the arm's win rate ≥ 90% | That filter enters the baseline |
| **Leaning** | CI excludes zero, positive, win rate < 90% | Baseline stays unfiltered; reported as a direction |
| **Not separable** | CI includes zero | Baseline stays unfiltered; "not separable" is not "equal" |
| **Worse** | CI excludes zero, negative | Baseline stays unfiltered; reported as a measured harm |

"Excludes zero" means strictly: a bound of exactly 0.000 does not, as in
sweep 3. The win rate is each arm's share of half-split selections among
all four runs, so at most one arm can reach 90% and the rule cannot
return two winners. k=10 and k=20 are reported and decide nothing.
`22_filter_result.py` applies the table; it and its tests are committed
with this protocol, before any filtered run.

## Predictions, recorded before the run

1. **All three arms are not separable from unfiltered BM25.** The
   motivating failure was dense retrieval, which ranks by overall
   similarity and can be pulled toward number-heavy chunks. BM25 ranks by
   shared terms, and a question shares few terms with an axis label.
   Freed top-10 slots will be few, well under 10% of the 580.
2. **No arm reaches a 90% win rate**, and many half-splits are exact ties,
   as in the fusion sweep.
3. **If any arm is worse, it is `filter050`**, because it is the only one
   that removes gold evidence.

## Threats not controlled

- **The gold set under-represents table evidence.** Every gold span was
  quoted from text an author could read and quote, which favours prose.
  Results tables are exactly what a prose filter removes, so the gold set
  sees little of the filter's cost there. A null here does not show the
  filter is harmless for questions answered by a table.
- Three thresholds of one score. A filter that separates axis labels from
  tables (the Phase 2b problem) is not tested.
- 58 questions: a filter that frees a handful of slots cannot show a
  detectable effect, whatever it does to those questions.

## Procedure

```powershell
python scripts\10_eval_retrieval.py --config sweep_bm25.yaml --tag filter030 --k-max 50 --min-prose-ratio 0.3
python scripts\10_eval_retrieval.py --config sweep_bm25.yaml --tag filter040 --k-max 50 --min-prose-ratio 0.4
python scripts\10_eval_retrieval.py --config sweep_bm25.yaml --tag filter050 --k-max 50 --min-prose-ratio 0.5
python scripts\16_paired_test.py bm25 filter030 filter040 filter050
python scripts\16_paired_test.py bm25 filter030 filter040 filter050 --k 10
python scripts\15_selection_stability.py bm25 filter030 filter040 filter050 --splits 2000
python scripts\22_filter_result.py
```
