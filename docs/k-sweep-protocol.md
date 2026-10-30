# Sweep 4, value of k: pre-registration

Written before the step analysis below has been computed. Nothing in it is
revised after seeing a result; if the design proves wrong, the flaw is
recorded next to the outcome.

## What it decides

How many chunks Phase 7 feeds the generator: `retrieval.k` in
`config/baseline.yaml`, currently 5 as a placeholder. Phase 7 measures
hallucination on unanswerable questions and faithfulness on answerable
ones, and both depend on what the generator is given. If a gold passage is
missing from the context, a wrong answer is a retrieval failure that looks
like a generation failure. The k must therefore be chosen on evidence and
reported, not picked.

Phase 5 found that **zero of 32 two-span questions had both passages
retrieved at k=5** under dense retrieval. This sweep is the same question
asked properly, under BM25.

## How this sweep differs from the others

**No new run.** The harness scores every arm at k = 1, 3, 5, 10, 20 from a
single ranked list, so `retrieval_bm25.csv` already holds every value this
sweep needs. The k values are nested prefixes of one list, not independent
configurations (methodology rules: "budget columns are nested subsets of
one ranked list"). Two consequences:

1. **Recall can only rise with k.** "Which k wins" has a trivial answer,
   so this is not a selection under the switching rule. The question is
   where adding chunks stops adding evidence detectably.
2. **Some numbers are already known.** BM25 span recall at k=10 (0.604)
   and k=20 (0.663) was reported in sweeps 2 and 3, and so was the count of
   two-span questions with both spans at k=10 (11 of 33). The k=1, 3 and 5
   columns are in the committed table but have not been analysed. The
   step-by-step gains and their intervals have not been computed. What is
   fixed in advance is the rule, not ignorance of the data.

## Primary analysis and rule

Span recall under BM25 (`retrieval_bm25.csv`), 58 scorable questions,
101 spans. For each step along the ladder **1 → 3 → 5 → 10 → 20**:

    gain(step) = spans found at the larger k − spans found at the smaller k,
                 summed over questions, / 101

with a 95% percentile CI from the cluster bootstrap over questions, 10,000
resamples, seed 20260827 (the project's instrument).

**Rule: fixed-sequence testing.** Test the steps in increasing order. Stop
at the first step whose CI does not exclude zero. The chosen k is the
**upper end of the last step whose CI excludes zero.**

| outcome | chosen k |
|---|---|
| 1 → 3 not separable | 1 |
| 1 → 3 separable, 3 → 5 not | 3 |
| up to 3 → 5 separable, 5 → 10 not | 5 |
| up to 5 → 10 separable, 10 → 20 not | 10 |
| every step separable | **20, the top of the ladder**. Reported as "gains not exhausted at 20", not as a plateau |

Fixed-sequence testing keeps the family-wise error rate at 0.05 without a
Holm adjustment, because each step is tested only if every earlier step
was rejected.

**What "excludes zero" means here.** Gains are nested, so each question's
gain is zero or positive and the bootstrap lower bound is either exactly
0 or above it. It is above 0 only when fewer than 2.5% of resamples drew
no question that gained a span. Roughly, a step passes when 4 or more of
the 58 questions gain a span. This is a low bar on purpose: the rule asks
whether a step adds evidence at all, not whether the evidence is worth
its cost. Cost is Phase 7's side of the trade, below.

**Why k and not the character budget.** The budget is the primary metric
for the phase because chunk size varied in sweep 1. Here chunking is fixed
at `fixed_512` (median 2,113 characters per chunk), so k and budget are
interchangeable to within chunk-length variation, and the generator is
configured by k.

**Why the ladder stops at 20.** At about 2,100 characters per chunk, k=20
is about 42,000 characters, roughly 10,000 tokens of context per answer.
Beyond that, Phase 7's cost grows with every RAGAS call, and long contexts
bring their own failure mode, where models use the middle of a long
context poorly. If every step is separable, the write-up says retrieval
gains continue past the largest k that Phase 7 can afford. The ladder is
not extended after the fact.

## Reported regardless, deciding nothing

- **Two-span completeness:** the share of two-span questions with both
  spans retrieved, at each k. This is the Phase 5 number, re-measured.
- **Complete coverage:** the share of scorable questions with every gold
  span retrieved, at each k. A generator can fully answer a multi-hop or
  comparative question only from complete evidence.
- **By question type:** span recall at each k for factual, comparative,
  multi_hop and ambiguous.
- **The 20 rewritten questions** (`retrieval_para_bm25.csv`): span recall
  at each k. If the curve on reworded questions is still climbing steeply
  at the chosen k, the chosen k is optimistic for questions that do not
  share vocabulary with their evidence. That is the confound the
  paraphrase test measured.
- **Reranked BM25** (`retrieval_rerank.csv`, exploratory since sweep 3):
  the same curve, and the smallest k at which reranked recall reaches
  BM25's recall at the chosen k. Generation uses the baseline, BM25.

## Predictions, recorded before the analysis

1. **Every step is separable and the rule returns k=20.** The 10 → 20
   step is already known to add 6 spans (0.604 → 0.663). Whether they fall
   in 4 or more questions is not known.
2. **Two-span completeness at k=5 is well above Phase 5's zero**, because
   BM25 completes 11 of 33 two-span questions at k=10 where dense
   completed far fewer.
3. **On the rewritten questions the curve is still rising at k=20**, so
   the chosen k is optimistic for low-overlap questions.

## Threats not controlled

- The rule says where evidence stops arriving. It says nothing about
  whether more context helps or hurts the generator, which is Phase 7's
  question. A larger k raises the chance that the evidence is present and
  also the number of distractors around it.
- 58 questions: a step gaining 3 spans in 3 questions fails the rule,
  even if the same gain would be detectable on a larger gold set.
- The authoring confound inflates BM25's curve on the original wording
  (paraphrase test). The rewritten-question curve is reported for that
  reason, but it covers 20 questions.

## Procedure

```powershell
python scripts\21_k_sweep.py
```

Reads the committed retrieval tables and writes
`results/tables/k_sweep.csv` (the steps and the decision) and
`results/tables/k_sweep_curve.csv` (recall by k for each arm and
question type). The script and its tests are committed with this protocol,
before the script is run on the real tables.
