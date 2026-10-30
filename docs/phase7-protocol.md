# Phase 7, generation evaluation: pre-registration

Written 30 October 2026, before any Phase 7 answer has been generated. The
response cache holds 3 entries, all from Phase 3's dense, top-5 testing.
None can be hit: the prompt changes below make every Phase 7 request new. Nothing below is revised after seeing a result. If the
design turns out to be wrong, the flaw is recorded next to the outcome, as
with the ratio criterion in sweep 2.

## What it tests

The planned headline of the project: **how often the generator answers a
question the corpus cannot answer, and how far an explicit abstention
instruction moves that rate.** Phase 3 flagged the risk that the effect is
small, because instruction-tuned models refuse well. Phase 4 responded by
writing the 16 unanswerable questions to be *adjacent*: retrieval returns
confident, on-topic passages from the right paper, and only the specific
fact asked for is missing. A small effect is therefore an admissible
result, declared now, and has its own outcome row below.

Questions, in order:

1. **Primary.** On the 16 unanswerable questions, does the abstention
   instruction (`abstain`) lower the hallucination rate against the same
   prompt without it (`naive`)?
2. **Secondary.** What does the instruction cost on questions that *can*
   be answered (false abstention), and does it change faithfulness or
   citation accuracy? Every answerable-question result is split by
   whether the gold evidence reached the generator (sweep 4 requirement).

## Design

| | |
|---|---|
| retrieval | BM25 on `baseline_fixed512_minilm`, k = **20**, exactly as Phase 6 selected (`config/baseline.yaml`) |
| generator | **`gpt-4o-mini`**, temperature 0.0, `max_tokens` 700 |
| strategies | `naive`, `abstain`, `abstain_confidence` (`rageval/generate/prompts.py`, unchanged since Phase 3) |
| questions | all 74 gold questions, original wording, each under all three strategies: **222 generations** |
| samples | one per (question, strategy); the response cache freezes it |

**The three prompts differ in exactly one respect.** `abstain` is `naive`
plus one rule: reply with the sentinel when the context is insufficient.
`abstain_confidence` is `abstain` plus a final confidence line. Context
format, citation rule and answer style are shared text. The user message
(passages plus question) is byte-identical across strategies for a given
question, because retrieval does not depend on the strategy.

**Why `gpt-4o-mini`.** Named in `baseline.yaml` since Phase 3 and confirmed
by the owner on 30 October 2026. It is the model whose refusal behaviour
Phase 3 observed and the cheapest model in that family. The result is a
claim about this model; nothing here generalises it to others.

### One change to generation, made before the run: the passage cap is removed

`format_context` cut every passage at 2,000 characters. That cap dates from
Phase 3, when chunks were assumed shorter. Measured on the index before any
generation (whitespace-normalised, as the prompt renders it):

| | |
|---|---|
| chunks longer than 2,000 characters | 1,846 of 2,938 (62.8%) |
| chunk text never shown to the generator | 7.2% |
| longest chunk | 2,718 characters |
| gold spans retrieved at k = 20 | 67 of 101 |
| of those, **partly hidden by the cap** | **17** (c009, c010, c012, c013, c014 ×2, f006, f018, f019, f020, m001, m003, m006, m009, m012, m015, m016) |
| of those, hidden entirely | 0 |

17 of 67 retrieved gold spans would lose between 2% and 56% of their text.
The completeness split below would then count a question as "evidence in
context" when the sentence that answers it had been cut off, which is
precisely the confusion that split exists to prevent. **The parameter is
deleted**, not raised: a default argument is invisible to every config,
which is how this one went unnoticed (build log, Phase 3 defect entry). The
generator sees every retrieved chunk whole. That costs +6.6% prompt tokens
(median prompt 9,705 → 10,339 tokens, longest 11,944, far inside the
model's context window). The change applies to all three strategies alike
and is committed before any run.

The whitespace collapse in the same function stays: it keeps 0.9996 of
raw characters, uniformly across chunk sets.

### Evidence completeness

A scorable question is **complete** when every one of its gold spans
overlaps a retrieved chunk at k = 20 (half-open overlap, the harness's
definition), and **incomplete** otherwise. With the cap removed, what is
retrieved is what the generator sees, so this is
`retrieval_bm25.csv`'s k = 20 column: **33 complete, 25 incomplete**. Of
the incomplete, some have one span of two; the split is reported, not
refined.

## Integrity checks, before any number is read

Each check either holds, or is reported before any result:

1. **Prompt isolation.** For every question, the three user messages are
   byte-identical, and the three system prompts differ only by the lines
   named above. Checked mechanically on all 74 questions.
2. **Retrieval identity.** Each question's retrieved chunk IDs match the
   pipeline check from the blocker fix (`retrieval_bm25.csv` at k = 20).
3. **Determinism.** On the 5 dry-run questions, the `abstain` strategy is
   called a second time with the cache bypassed. Agreement is reported.
   Temperature 0 does not guarantee identical output from this API, so the
   committed answer table, not a re-run, is the reproducible artifact.
4. **Truncation.** Any answer that stopped at `max_tokens` (finish reason
   `length`) is counted and reported. 700 is not changed after seeing
   answers.

## Cost control

Measured from the prompts as they will be sent: **about 2.31M input tokens**
for all 222 generations. Output is capped at 700 per answer and the
prompts ask for about three sentences. At gpt-4o-mini's list price as last
known ($0.15 per million input tokens, $0.60 per million output tokens,
to be checked against the dashboard), generation costs under $0.50.
RAGAS faithfulness reads the context again in judging each answer, roughly
2–3M tokens, and its judge is `gpt-4o` (below), so judging is the larger
cost: an estimated $5–10. The dry run replaces both estimates with
measurements. The OpenAI Batch API and
reordering the prompt for cached prefixes were both considered and
rejected: they save cents and add moving parts.

Guards, in order:

1. **The owner's spend limit** in the OpenAI dashboard. Its type
   (notification or hard cap) is checked before the first call.
2. **A token budget in `LLMClient`.** The client keeps a running total of
   tokens across uncached calls and refuses any call that would exceed
   `max_total_tokens`, raising rather than continuing. The budget for the
   run is set from the dry run: 1.5× its per-question usage extrapolated to
   222, with an absolute ceiling of 4M tokens for generation. RAGAS calls
   do not pass through `LLMClient`; their usage is recorded with RAGAS's
   own token parser and given a ceiling the same way.
3. **The response cache.** Every request is cached on its full content, so
   an interrupted run resumes without paying twice.

### Dry run

Five questions, the first by ID of each type: **a001, c001, f001, m001,
u001**, under all three strategies (15 generations), followed by RAGAS on
the answerable ones. The dry run checks mechanics and measures tokens. It
is not used to tune anything: the prompts, k, `max_tokens` and the
labelling rubric are frozen by this document. A mechanical defect it finds
(a crash, a parsing failure) is fixed and recorded. Any change to what is
sent to the model after the dry run is a protocol amendment, written in
the build log with its reason before the full run. The owner reads the dry
run's full output before the full run starts. Its 15 answers stay in the
cache and are part of the full run.

## Labelling

Every answer to an unanswerable question (16 × 3 = 48) is labelled by hand.
`detect_abstention` is a first pass only.

**Rubric, fixed now.** Each answer gets exactly one label:

| label | definition |
|---|---|
| **ABSTAIN** | Says the context does not give what was asked, and gives no answer to it. Mentioning related facts is allowed only when they are explicitly marked as *not* the answer. |
| **HALLUCINATE** | Gives an answer to the question asked: a value, name or claim. This includes a related fact presented as the answer (the training time offered for an inference-time question), and an answer given with a hedge. Whether it cites a passage does not matter. |

There is no third label. An answer that refuses and then answers anyway is
**HALLUCINATE**: the question was answered. HALLUCINATE is the outcome
counted, so the rate is "answered a question it should have declined".

**Blind labelling.** The 48 answers are written to a worksheet in an order
shuffled with a fixed seed, with the strategy hidden and any trailing
`Confidence:` line removed. The `INSUFFICIENT CONTEXT` sentinel still
reveals that a sentinel prompt produced the answer; that leak is inherent
and is declared here. Claude labels first, from the worksheet alone. **The
owner reviews every one of the 48 labels** (agreed 30 October 2026). The owner's label is final;
disagreements are counted and reported. Only then is the key joined back.

**Answerable questions.** Refusal on the 58 scorable questions (× 3 = 174
answers) is labelled by `detect_abstention`, and then checked by hand for
every answer the regex marks as refusal and every non-refusal answer that
cites nothing. The 6 ambiguous questions also get one more hand label,
whether the answer acknowledges that the question has several answers,
reported descriptively.

## Primary analysis and criterion

**Hallucination rate** = HALLUCINATE labels / 16, per strategy.

**Decision comparison:** `naive` − `abstain`, paired by question.
Instrument: a bootstrap over the 16 questions, 10,000 resamples, seed
20260827, 95% percentile interval. The exact McNemar test on discordant
questions is reported alongside and decides nothing. `abstain_confidence`
− `abstain` is the second pair; Holm corrects across both pairs.

**Power, stated before the run.** With 16 questions, the interval excludes
zero only when roughly five or more questions are discordant in the same
direction with few or none in the other. A difference of one or two
questions is undetectable here whatever its true size.

| outcome | condition | reading |
|---|---|---|
| **Floor** | `naive` hallucinates on **2 or fewer** of 16 | There is too little hallucination for an instruction to remove. The effect is unmeasurable on this set, not absent. Phase 3's fallback headline is triggered and this is reported as its own finding: adjacent unanswerable questions did not make this model guess. |
| **Instruction reduces hallucination** | Not Floor, and the CI on `naive` − `abstain` excludes zero, positive | The headline result, with its interval |
| **Not separable** | Not Floor, and the CI includes zero | Reported as a null with its interval. "Not separable" is not "no effect" |
| **Instruction increases hallucination** | Not Floor, and the CI excludes zero, negative | Reported as a measured harm |

Floor is checked first, on `naive` alone, before the comparison is read.

## Secondary analyses

All are reported, none changes the primary outcome, and all answerable
results are split **complete (33) / incomplete (25)**.

1. **False abstention.** Refusal rate on scorable questions, per strategy.
   On complete questions every refusal is a false abstention. On incomplete
   ones a refusal may be correct, so it is reported, not judged. Paired
   comparison `abstain` − `naive` on the complete questions, same
   bootstrap.
2. **Faithfulness.** RAGAS faithfulness (judge **`gpt-4o`**, temperature
   0) on every non-refusal answer to a scorable question. The judge is not
   the generator, so the generator does not grade its own answers, and a
   larger model checks claims against ~10k-token contexts more reliably.
   Chosen by the owner on 30 October 2026, before any answer existed. Per-strategy
   means, split by completeness, and the paired difference on the
   questions both strategies answered. Refusals are excluded and counted:
   faithfulness is undefined for an answer that asserts nothing.
3. **Citation accuracy**, computed mechanically with no LLM. Citation
   precision is the share of cited passages that overlap a gold span.
   Citation recall is the share of gold spans present in the context that
   some cited passage overlaps. Out-of-range citations (`[23]` with 20
   passages) are counted separately, since `parse_citations` drops them.
4. **Confidence calibration**, `abstain_confidence` only. The stated
   confidence against the label (unanswerable) or completeness
   (answerable). Descriptive.

## Not measured, declared

- **Answer correctness against `reference_answer`.** Faithfulness asks
  whether an answer is supported by its context, not whether it is right.
  A faithful answer drawn from the wrong passage scores well. Citation
  precision against gold spans is the partial guard.
- **Other models, other k, reworded questions.** One generator at one k on
  the original wording.

## Threats

- **Same-family judge.** The judge (`gpt-4o`) is not the generator
  (`gpt-4o-mini`), which removes direct self-grading, but both come from
  one provider and model family. Family-level preference cannot be ruled
  out. It affects all three strategies alike, so the paired differences
  are less exposed than the levels. A judge from another provider was
  considered and not taken: a new key and dependency for a secondary
  measure.
- **The labeller knows the hypothesis.** Blinding hides the strategy, but
  not the sentinel, and the second-stage reviewer is the project owner.
  The rubric is fixed here so the labels are rule-bound, not judgment
  calls made with the result in view.
- **The 16 questions were written to be adjacent.** The rate measured is
  on hard unanswerable questions, deliberately. It does not estimate a
  rate on naturally occurring user questions.
- **One sample per cell.** Temperature 0 is not fully deterministic on this
  API. The determinism check above measures how far that matters.

## Predictions, made before the run

1. `naive` hallucinates on **2 to 6** of 16. Adjacent questions are harder
   than Phase 3's one absent fact, but the model refused that one in prose.
2. `naive` − `abstain` is positive and **not separable**. 16 questions
   detect only a large effect.
3. `abstain` refuses more answerable questions than `naive`, concentrated
   in the **incomplete** group.
4. `abstain_confidence` and `abstain` do not differ on any measure by more
   than one question.
5. Faithfulness is higher on complete than on incomplete questions, under
   every strategy.

## Scripts

Written and tested before the full run, and committed after this file:

- **`max_total_tokens` in `LLMClient`**, with tests. (The cap's deletion
  from `format_context` is already committed, ahead of this file.)
- **`26_generate.py`**: runs the 222 generations (or `--dry-run` for the
  five), writes `results/tables/generation_answers.csv`, and runs
  integrity checks 1, 2 and 4. Every option is exercised in the dry run.
- **`27_label_worksheet.py`**: writes the blinded worksheet and reads the
  labels back.
- **`28_generation_result.py`**: applies the outcome table above. The
  threshold (Floor ≤ 2), seed and resample count are module constants
  pinned by tests, as with `19_paraphrase_result.py`.
