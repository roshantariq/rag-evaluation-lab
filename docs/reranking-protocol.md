# Sweep 3, cross-encoder reranking: pre-registration

Written before any reranking code or run exists. Nothing below is revised
after seeing a result. If the design turns out to be wrong, the flaw is
recorded next to the outcome, as with the ratio criterion in sweep 2.

## What it tests

BM25 is the baseline. The paraphrase test then showed that its lead over
dense retrieval on this gold set is mostly shared vocabulary between
question and evidence. A cross-encoder scores the question and the
passage together rather than matching terms or comparing two separately
built vectors. It is the remaining axis where a large effect is
plausible.

Two questions, in order:

1. **Primary.** Does reranking BM25's candidates with a cross-encoder raise
   span recall over BM25 alone, by enough to replace the baseline?
2. **Secondary.** Does any gain survive on questions that do not share
   vocabulary with their evidence? This is the confound that bounded the
   last result. Testing for it is declared now, not discovered afterwards.

## Design

| | |
|---|---|
| first stage | BM25 on `baseline_fixed512_minilm`, unchanged from `sweep_bm25.yaml` |
| reranker | `BAAI/bge-reranker-v2-m3` (568M parameters), sentence-transformers `CrossEncoder`, `max_length` **1024** |
| depth | BM25's top **100** chunks are rescored; the harness then scores the reordered list to `--k-max 50` |
| everything else | identical: chunking, collection, gold set, scoring path |

**Why this model: chosen on published evidence, not on this gold set.**
Choosing among rerankers by running them on the 58 questions and then
reporting the winner on the same 58 would make the test grade its own
selection. The model was therefore fixed from outside evidence, and the
gold set is used once, to test it.

| reranker | params | MTEB(eng, v2) retrieval nDCG@10 | NanoBEIR nDCG@10 |
|---|---|---|---|
| ms-marco-MiniLM-L6-v2 | 23M | 0.508 | 0.631 |
| ms-marco-MiniLM-L12-v2 | 33M | 0.507 | 0.637 |
| bge-reranker-base | 278M | 0.489 | 0.603 |
| mxbai-rerank-base-v1 | 184M | 0.487 | 0.623 |
| mxbai-rerank-large-v1 | 435M | 0.506 | 0.649 |
| bge-reranker-large | 560M | 0.510 | 0.638 |
| **bge-reranker-v2-m3** | **568M** | **0.553** | — |

Source: the Hugging Face Ettin reranker comparison, a third party
benchmarking every model on the same top-100 candidates. Two findings
decided it:

- **Among small and base-size rerankers, size buys nothing measurable.**
  The 23–33M MiniLM models match or beat 184–560M models. A
  vendor table (Mixedbread, over BM25's top 100) shows mxbai-rerank-base
  far ahead of bge-reranker-base, but that gap does not survive
  independent measurement. "Stronger" among these is not supported.
- **bge-reranker-v2-m3 is the one model with a clear independent margin**,
  about +0.045 nDCG@10 over every other model in the table. It is the
  strongest reranker this budget allows, so a null cannot be blamed on a
  weak model.

Its own published BEIR evaluation reranks a dense retriever's top 100,
not BM25's. No source found measures it over BM25. One model only: a
second would add a comparison to a sweep whose first question is whether
reranking helps at all.

**Why `max_length` 1024.** The model's tokenizer allows 8,192 tokens. It
splits this corpus finer than BERT does (median chunk 533 tokens, longest
929, longest question 77). Measured before the run:

| `max_length` | chunks truncated | chunk text unseen |
|---|---|---|
| 512 | 92.9% | **18.4%** |
| 768 | 0.3% | 0.1% |
| **1024** | **0%** | **0%** |

At the default 512 the reranker would miss nearly a fifth of the text it
is asked to judge, the same failure Phase 5 measured in the dense encoder.
At 1024 it reads every chunk whole. Whether its scoring quality holds past
512 tokens is not documented, and is listed as a threat.

**Why depth 100.** Reranking TREC-COVID with SciBERT was limited at depths
of 20 or less by first-stage ranking, best at 50–100, and worse beyond.
100 also matches the pool size the fusion sweep used per retriever. A
reranker can only reorder what BM25 retrieved, so BM25's span recall
within its top 100 is the ceiling of this arm and is reported with it.

**One scoring path.** Reranking is added to `10_eval_retrieval.py` as a
stage after BM25 (`--rerank <model> --rerank-depth 100 --rerank-max-length 1024`), not as a second
harness. Every new option is exercised once on a single question before
the scored runs, because an unexercised `--gold` crashed the paraphrase
runs.

## Runs

| tag | gold set | purpose |
|---|---|---|
| `bm25` | original | existing table, the control. Not re-run |
| `rerank` | original | primary |
| `para_bm25` | paraphrased | existing table. Not re-run |
| `para_rerank` | paraphrased | secondary |

`para_bm25` and `rerank` share every question the paraphrase test did not
touch. The untouched 38 must score identically in `rerank` and
`para_rerank`. If they do not, the reranker is nondeterministic and that
is reported before any number is read.

## Primary analysis and criterion

Span recall at the **20,000-character budget**, `rerank` against `bm25`,
all 58 scorable questions. Instruments: `16_paired_test.py` (cluster
bootstrap over questions, 10,000 resamples, seed 20260827; McNemar exact
and Holm reported for continuity) and `15_selection_stability.py` (2,000
stratified half-splits × 2 directions). Script 16 is run on `baseline`
(dense), `bm25` and `rerank` together, so Holm corrects across all three
pairs. Only `rerank` vs `bm25` decides anything.

**A paired randomization test over questions is reported beside McNemar**
and decides nothing. McNemar is a sign test on discordant spans that
treats clustered spans as independent (literature check, build log).

The switching rule has never named a number for "decisively above
chance". This sweep fixes one before the run: script 15's own **ROBUST**
threshold, a win rate of at least **90%**.

| outcome | condition | what happens |
|---|---|---|
| **Replaces BM25** | CI on (`rerank` − `bm25`) excludes zero, positive, **and** win rate ≥ 90% | Reranking enters the baseline; sweep 4 inherits it |
| **Leaning** | CI excludes zero, positive, win rate < 90% | Baseline stays BM25; reported as a direction, not a result |
| **Null — not separable** | CI includes zero | Baseline stays BM25; written up as a null. "Not separable" is not "equal" |
| **Worse** | CI excludes zero, negative | Baseline stays BM25; reported as a measured harm |

k=10 and k=20 are reported alongside and decide nothing. Chunking is
fixed, so fixed k is not confounded, but the budget is the primary metric
for the whole phase.

## Secondary analysis: does the gain survive the confound?

On the same 20 rewritten questions the paraphrase test used, at the 20k
budget, with the same bootstrap:

    gain_original   = rerank − bm25,           original phrasing
    gain_paraphrase = para_rerank − para_bm25, rewritten phrasing

| outcome | reading |
|---|---|
| CI on `gain_paraphrase` excludes zero, positive | The reranker helps where BM25's lexical advantage is gone. Its gain is not just vocabulary. |
| CI on `gain_paraphrase` includes zero | **Inconclusive, not negative.** 20 questions and 40 spans detect a large gain, not a modest one. |
| CI on `gain_paraphrase` excludes zero, negative | Reranking hurts on paraphrased questions. Reported whatever the primary result. |

`gain_original − gain_paraphrase` is reported with its CI and decides
nothing. The secondary analysis **cannot overturn the primary decision.**
It qualifies what the decision claims, as the paraphrase test did for
sweep 2.

## Reported regardless

- Latency: wall-clock per question for BM25 alone and for BM25 + rerank.
  Latency is a finding, not something to hide.
- The candidate ceiling: BM25 span recall within its top 100.
- Which questions the reranker newly recovers or loses at k=20, read by
  hand, including f013 and c002. Every configuration has missed them.

## Predictions, recorded before the run

1. **The primary gain is positive but small, and probably not separable.**
   The reranker cannot surface anything BM25 missed beyond depth 100.
   BM25 already reaches 0.86 question-level recall at k=20, so the room is
   in ordering, not in recall.
2. **f013 and c002 stay missed.** If BM25 never ranks their evidence
   anywhere near the top, reranking cannot help.
3. **The secondary analysis is inconclusive**, for the power reason above.

## Threats not controlled

- One reranker, chosen on general-domain benchmarks, applied to
  scientific prose. No source measures it over BM25 candidates.
- `max_length` 1024 avoids truncation, but published evaluations use 512.
  Quality past 512 tokens is assumed, not shown.
- One depth. The depth curve is not measured.
- The secondary analysis reuses the 20 questions selected for the
  paraphrase test, chosen for high overlap, not at random.
- A null here says this reranker at this depth did not separate on 58
  questions. It does not say reranking cannot help.

## Procedure

```
# after the harness change is committed and every new option is exercised once
python scripts/10_eval_retrieval.py --config sweep_bm25.yaml --rerank BAAI/bge-reranker-v2-m3 --rerank-depth 100 --rerank-max-length 1024 --tag rerank --k-max 50
python scripts/10_eval_retrieval.py --config sweep_bm25.yaml --rerank BAAI/bge-reranker-v2-m3 --rerank-depth 100 --rerank-max-length 1024 --gold data/eval/gold_paraphrased.jsonl --tag para_rerank --k-max 50

python scripts/16_paired_test.py baseline bm25 rerank
python scripts/15_selection_stability.py bm25 rerank --splits 2000
```

The secondary analysis and the randomization test need a small script
(`20_rerank_result.py`). It is written and tested on synthetic data
before either run, as `19` was.
