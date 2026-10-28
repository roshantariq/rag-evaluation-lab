# Paraphrase validity test — pre-registration

Written before any question was rewritten. Nothing below is revised after
seeing a result; if the design turns out to be wrong, the flaw is recorded
alongside the outcome the way the ratio criterion in sweep 2 was.

## What it tests

Sweep 2 measured BM25 beating dense retrieval by +0.168 span recall
(CI [+0.082, +0.253], Holm p = 0.0069, 93.2% of half-splits). It then found
that advantage concentrated in questions sharing vocabulary with their
evidence — low-overlap tercile +0.037, CI [−0.115, +0.208] — and returned a
BOUNDED verdict.

The reason is structural. Every gold question was authored while reading
its evidence passage, which hands a lexical retriever the exact tokens the
author had in front of them and gives dense retrieval nothing comparable.
Neither the paired test nor the stability analysis can detect this: both
resample the same 58 questions, and a bias present in how all of them were
written survives any amount of resampling. The switching rule guards
against sampling error; this is construct error.

This test intervenes rather than describes. **Rewrite the highest-overlap
questions to ask the same thing in different words, and measure whether the
advantage survives.**

## Why this beats the tercile split it replaces

The tercile comparison was between groups: 19 questions against 20
different ones, differing in subject matter, difficulty and type as well as
in overlap. Its interaction interval was [−0.049, +0.367] — it could not
establish anything.

This is within question. The same question appears in two phrasings, scored
by both retrievers, with evidence spans, reference answer, difficulty and
type held constant by construction. The only thing that varies is wording.
Same n, far less noise.

## Selection

The 20 scorable questions with the highest IDF-weighted overlap between
question and evidence, computed by `18_paraphrase_worksheet.py`. Selection
is mechanical — no question is chosen or excluded by judgement.

Type composition is reported but not balanced. Overlap correlates with
question type (a comparative question names two systems and so carries more
rare terms), so forcing a balance would break the "highest overlap" rule
that makes selection mechanical. The composition is recorded as a
limitation instead.

## Rules for rewriting

1. **The answer must not change.** The same evidence spans must still
   answer it and the reference answer must still be correct.
2. **Use none of the terms under `avoid:`.** That list is computed, not
   judged: the shared terms carrying the top 80% of the overlapping IDF
   mass — the quantity BM25 actually scores on. Sub-forms count, because
   the tokenizer emits them: banning `ecmwf-ifs` also bans `ECMWF` and
   `IFS`. Terms appearing in more than 20% of the corpus are never banned;
   they are function words BM25 already scores near zero.
3. **Describe rather than name**, where the description is unambiguous
   within this corpus.
4. **Keep it a question someone would plausibly type.** A rewrite nobody
   would ask tests nothing.
5. **Declare any retained term** with `keep: <term>` when a question
   becomes genuinely ambiguous without it. Declared exceptions are counted
   and reported; undeclared ones are rejected and nothing is written.

### Bias controls

The writer knows the hypothesis. Three defences, in descending strength:

- the banned list is computed, so compliance is mechanical rather than a
  matter of good intentions;
- `--build` rejects the whole set if any block violates, so partial
  compliance cannot silently weaken the manipulation;
- **the worksheet contains no retrieval results.** Which questions BM25
  currently gets right is withheld deliberately, so a rewrite cannot be
  aimed, consciously or otherwise.

The rewrites are done by the project author, not by an LLM — a model that
has seen the analysis would be the single worst choice of rewriter here.

## Manipulation check, declared in advance

The test is valid only if the rewrites actually removed shared vocabulary.

**Mean IDF-weighted overlap must fall by at least 0.25 (absolute).** Below
that, the manipulation failed: rewrite further. Do not reinterpret the
threshold after seeing it.

## Primary analysis

On the rewritten questions only, with span recall at the 20k character
budget and at k=20:

    Δ_original   = bm25 − dense, original phrasing
    Δ_paraphrase = bm25 − dense, rewritten phrasing
    effect       = Δ_original − Δ_paraphrase

95% CI by cluster bootstrap over questions, 10,000 resamples, seed 20260827
— questions are the resampling unit because spans within a question are
correlated.

## Criterion — stated as an interval, not a point

Sweep 2's pre-registration put a hard threshold on a ratio of two noisy
estimates and was recorded as defective for it. This one is an interval
rule.

| outcome | reading | what gets reported |
|---|---|---|
| CI on `effect` excludes zero, positive | The advantage shrinks when vocabulary is removed. Confound explanation supported. | `Δ_paraphrase` becomes the honest effect size; the aggregate +0.168 is reported as inflated by authoring, with the measured amount. |
| CI includes zero | Inconclusive, **not** negative. | Aggregate reported with the sweep 2 tercile caveat retained, and this test reported as underpowered rather than as support for either side. |
| CI excludes zero, negative | The advantage is larger without shared vocabulary. Confound explanation refuted. | The bound lifts; +0.168 stands as an estimate of retrieval quality. |

"Inconclusive" is a real outcome here and is expected to be reasonably
likely: 20 questions carry roughly 40 spans, so this can detect a large
change in Δ and not a subtle one. That is stated now so that a wide
interval is not written up afterwards as evidence of no effect.

## Secondary check: did the rewrites break the questions?

A rewrite that makes a question vaguer lowers both retrievers, which the
paired difference absorbs — but it also means the test is measuring
question quality rather than vocabulary.

**Guard, declared in advance:** if dense span recall on the rewritten
questions falls by more than 0.15 against the same questions' original
scores, treat it as a broken-question warning, re-read every rewrite
against rule 1, and report the warning whatever the primary result says.
Dense retrieval is the right instrument for this because it is the arm the
manipulation is not aimed at.

## Threats not controlled

- **One rewriter, who knows the hypothesis.** Mitigated, not eliminated.
- **n = 20 questions, ~40 spans.** Underpowered for a subtle effect.
- **Rewrites may be less natural than real user queries**, in the opposite
  direction from the original questions being less natural for the same
  reason. Neither phrasing is a sample of real usage.
- **Type composition of the selected 20 is whatever the overlap ranking
  gives**, and is not balanced.

## Procedure

```
python scripts/18_paraphrase_worksheet.py
# fill every paraphrase> block by hand
python scripts/18_paraphrase_worksheet.py --build

python scripts/10_eval_retrieval.py --config dense_phase5.yaml \
    --gold data/eval/gold_paraphrased.jsonl --tag para_dense --k-max 50
python scripts/10_eval_retrieval.py --config sweep_bm25.yaml \
    --gold data/eval/gold_paraphrased.jsonl --tag para_bm25 --k-max 50

python scripts/19_paraphrase_result.py
```

*Procedural correction, 28 October 2026.* The dense arm originally read
`--config baseline.yaml`. Sweep 2 later redefined `baseline.yaml` as BM25,
so that command would have run BM25 twice. The dense arm is now
`dense_phase5.yaml`, the preserved Phase 5 dense configuration. Nothing
else in this protocol changed, and no paraphrase run had happened. See
`docs/build-log.md`, Plan amendments.