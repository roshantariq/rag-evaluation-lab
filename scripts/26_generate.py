"""Phase 7: generate every answer, check its integrity, then judge faithfulness.

Implements docs/phase7-protocol.md. Nothing here decides an outcome; that
is 28_generation_result.py. The dry run and the full run are the same code
on different question lists, so the dry run exercises every option the full
run uses.

STAGES

  generate   74 questions x 3 strategies = 222 answers (dry run: a001, c001,
             f001, m001, u001 = 15), BM25 at k = 20, pinned generator.
             Every answer is cached on its full request, so a run stopped by
             the token budget or the provider's cap resumes for free.
  checks     1 prompt isolation, 2 retrieval identity against
             retrieval_bm25.csv at k = 20, 4 answers cut off at max_tokens,
             5 one served model version. Dry run adds 3, determinism: the
             abstain strategy called again with the cache bypassed.
  judge      (--judge) RAGAS faithfulness, pinned judge, on every scorable
             answer the regex does not mark as a refusal. Checkpointed per
             answer; a failed call stops the run; no NaN is written.

BUDGETS

  --budget-tokens is required: the generation token budget for this
  invocation's uncached calls. The protocol's absolute ceiling, 4,000,000,
  is refused above. --judge-budget-tokens is required with --judge.

WRITES

  results/tables/generation_answers[_dryrun].csv
  results/tables/generation_faithfulness[_dryrun].csv   (with --judge)
  .cache/faithfulness.jsonl                             (judge checkpoint)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys

import pandas as pd
import yaml

from rageval.config import CHROMA_DIR, CONFIG_DIR, EVAL_DIR, PROJECT_ROOT, RESPONSE_CACHE, TABLES_DIR
from rageval.evaluation import faithfulness as fa
from rageval.evaluation.gold import load_gold
from rageval.evaluation.spans import span_indices
from rageval.generate.llm import LLMClient, TokenBudgetExceeded
from rageval.generate.prompts import PROMPT_STRATEGIES, build_prompt
from rageval.pipeline import RAGPipeline
from rageval.retrieve.factory import build_retriever
from rageval.store.chroma_store import ChromaStore

# --- fixed by docs/phase7-protocol.md; do not change -----------------------
STRATEGIES = ("naive", "abstain", "abstain_confidence")
DRY_RUN_IDS = ("a001", "c001", "f001", "m001", "u001")
K = 20
GENERATION_CEILING = 4_000_000
JUDGE_CHECKPOINT = PROJECT_ROOT / ".cache" / "faithfulness.jsonl"

# USD per million tokens, checked 30 Oct 2026 against third-party listings;
# the OpenAI dashboard is authoritative. Used only to print a projection.
PRICES = {"gpt-4o-mini-2024-07-18": (0.15, 0.60), "gpt-4o-2024-08-06": (2.50, 10.00)}

_CITATION = re.compile(r"\[(\d+)\]")


def spans_in_context(q, hits) -> set[int]:
    """Gold span indices overlapping any retrieved chunk (half-open)."""
    found = set()
    for i, e in enumerate(q.evidence):
        for h in hits:
            if h.arxiv_id == e.arxiv_id and h.char_start < e.char_end and e.char_start < h.char_end:
                found.add(i)
                break
    return found


def out_of_range(answer: str, n_hits: int) -> int:
    return sum(1 for m in _CITATION.finditer(answer) if not 1 <= int(m.group(1)) <= n_hits)


def cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="baseline.yaml")
    ap.add_argument("--dry-run", action="store_true", help="the five protocol questions only")
    ap.add_argument("--budget-tokens", type=int, required=True,
                    help="generation token budget for uncached calls in this invocation")
    ap.add_argument("--judge", action="store_true", help="also run RAGAS faithfulness")
    ap.add_argument("--judge-budget-tokens", type=int, default=None)
    ap.add_argument("--judge-retest", action="store_true",
                    help="dry run only: judge every judged answer once more (protocol amendment)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
    for noisy in ("httpx", "openai", "ragas", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    if args.budget_tokens > GENERATION_CEILING:
        print(f"--budget-tokens {args.budget_tokens:,} is above the protocol's "
              f"ceiling of {GENERATION_CEILING:,}.")
        return 2
    if args.judge and args.judge_budget_tokens is None:
        print("--judge needs --judge-budget-tokens.")
        return 2
    if args.judge_retest and not (args.dry_run and args.judge):
        print("--judge-retest runs on the dry run only, with --judge.")
        return 2

    with open(CONFIG_DIR / args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    gen = cfg["generation"]
    judge_model = cfg["evaluation"]["judge_model"]
    if cfg["retrieval"]["mode"] != "bm25" or cfg["retrieval"]["k"] != K:
        print(f"config retrieval is {cfg['retrieval']}; the protocol fixes bm25 at k = {K}.")
        return 2
    for name in (gen["model"], judge_model):
        if not re.search(r"-\d{4}-\d{2}-\d{2}$", name):
            print(f"{name} is an alias, not a dated version (protocol amendment).")
            return 2

    gold = load_gold(EVAL_DIR / "gold_questions.jsonl")
    if args.dry_run:
        gold = [q for q in gold if q.id in DRY_RUN_IDS]
    suffix = "_dryrun" if args.dry_run else ""
    answers_path = TABLES_DIR / f"generation_answers{suffix}.csv"
    faith_path = TABLES_DIR / f"generation_faithfulness{suffix}.csv"
    # A stale output never reads as a result; only the tables this run writes.
    for p in (answers_path, faith_path) if args.judge else (answers_path,):
        p.unlink(missing_ok=True)

    store = ChromaStore(CHROMA_DIR, cfg["index"]["collection"])
    retriever = build_retriever(cfg, store)
    llm = LLMClient(provider=gen.get("provider", "openai"), model=gen["model"],
                    temperature=gen["temperature"], max_tokens=gen["max_tokens"],
                    cache_path=RESPONSE_CACHE, max_total_tokens=args.budget_tokens)
    pipe = RAGPipeline(retriever, llm, k=K)

    bm25_table = pd.read_csv(TABLES_DIR / "retrieval_bm25.csv", dtype={"hit_spans@20": str})
    harness = {r["id"]: span_indices(r["hit_spans@20"]) for _, r in bm25_table.iterrows()}

    print(f"Phase 7 {'DRY RUN' if args.dry_run else 'FULL RUN'}: {len(gold)} questions x "
          f"{len(STRATEGIES)} strategies, BM25 k = {K}, generator {gen['model']}, "
          f"max_tokens {gen['max_tokens']}, budget {args.budget_tokens:,} tokens\n")

    rows, problems, texts = [], [], {}
    isolation_failures = retrieval_mismatches = 0
    try:
        for q in gold:
            users = set()
            hit_ids = set()
            for strategy in STRATEGIES:
                a = pipe.answer(q.question, strategy=strategy)
                _, user = build_prompt(q.question, a.hits, strategy)
                users.add(user)
                hit_ids.add(tuple(h.chunk_id for h in a.hits))
                for h in a.hits:
                    texts[h.chunk_id] = " ".join(h.text.split())
                found = spans_in_context(q, a.hits)
                # The protocol's split. Never a null-like label (bug museum).
                if q.question_type == "unanswerable":
                    label = "unanswerable"
                else:
                    label = "complete" if len(found) == len(q.evidence) else "incomplete"
                # Citation accuracy, mechanical (secondary 3): which cited
                # passages overlap a gold span, and which gold spans they cover.
                cited = [a.hits[i - 1] for i in a.citations]
                cited_on_gold = sum(1 for h in cited if spans_in_context(q, [h]))
                spans_cited = spans_in_context(q, cited)
                row = a.to_row()
                row.update(id=q.id, question_type=q.question_type, evidence=label,
                           gold_spans_in_context=";".join(str(i) for i in sorted(found)),
                           n_gold_spans=len(q.evidence),
                           cited_on_gold=cited_on_gold,
                           gold_spans_cited=";".join(str(i) for i in sorted(spans_cited)),
                           out_of_range_citations=out_of_range(a.answer, len(a.hits)))
                rows.append(row)
                print(f"  {q.id} {strategy:<19} {label:<12} "
                      f"{'REFUSAL(regex)' if a.abstained else 'answer':<14} "
                      f"in {a.prompt_tokens:>6,} out {a.completion_tokens:>4} "
                      f"{a.finish_reason or '-':<6} {'cached' if a.cached else ''}")
                if q.question_type != "unanswerable" and found != harness[q.id]:
                    retrieval_mismatches += 1
                    problems.append(f"check 2: {q.id} {strategy} spans {sorted(found)} vs "
                                    f"harness {sorted(harness[q.id])}")
            if len(users) != 1 or len(hit_ids) != 1:
                isolation_failures += 1
                problems.append(f"check 1: {q.id} user messages or hits differ across strategies")
    except TokenBudgetExceeded as e:
        # Nothing partial is written: every finished answer is in the cache,
        # so a re-run with a larger budget resumes from here at no cost.
        print(f"\nGENERATION STOPPED: {e}")
        print(f"  {len(rows)} answers finished this run; re-run to resume.")
        return 3

    out = pd.DataFrame(rows)
    lead = ["id", "question_type", "strategy", "evidence", "gold_spans_in_context", "n_gold_spans"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    out.to_csv(answers_path, index=False, encoding="utf-8")

    # --- integrity checks ---------------------------------------------------
    system = {s: PROMPT_STRATEGIES[s] for s in STRATEGIES}
    prompt_ok = (system["abstain"].startswith(system["naive"])
                 and system["abstain_confidence"].startswith(system["abstain"]))
    truncated = int((out["finish_reason"] == "length").sum())
    served = sorted(set(out["served_model"]) - {""})
    unknown_served = int((out["served_model"] == "").sum())

    print("\nINTEGRITY")
    print(f"  1 prompt isolation      : system prompts nest {'yes' if prompt_ok else 'NO'}; "
          f"user message and hits identical across strategies on "
          f"{len(gold) - isolation_failures}/{len(gold)} questions")
    n_scorable = int((out["question_type"] != "unanswerable").sum())
    print(f"  2 retrieval identity    : {n_scorable} scorable rows checked, "
          f"{retrieval_mismatches} differ from retrieval_bm25.csv at k = {K}")
    print(f"  4 cut off at max_tokens : {truncated}")
    print(f"  5 served versions       : {served or 'none recorded'}"
          f"{f'; {unknown_served} rows without a served version' if unknown_served else ''}")
    if len(served) > 1:
        problems.append(f"check 5: answers came from {len(served)} versions: {served}")
    if not prompt_ok:
        problems.append("check 1: system prompts do not nest naive < abstain < abstain_confidence")
    if truncated:
        problems.append(f"check 4: {truncated} answers stopped at max_tokens")

    # --- check 3, dry run only ---------------------------------------------
    if args.dry_run:
        fresh = LLMClient(provider=gen.get("provider", "openai"), model=gen["model"],
                          temperature=gen["temperature"], max_tokens=gen["max_tokens"],
                          cache_path=None,
                          max_total_tokens=max(0, args.budget_tokens - llm.tokens_used))
        same = 0
        repeat = RAGPipeline(retriever, fresh, k=K)
        try:
            for q in gold:
                again = repeat.answer(q.question, strategy="abstain").answer
                first = out[(out["id"] == q.id) & (out["strategy"] == "abstain")]["answer"].iloc[0]
                same += again == first
                if again != first:
                    print(f"  3 {q.id}: second call differs\n    first : {first[:200]}\n"
                          f"    second: {again[:200]}")
            print(f"  3 determinism           : {same}/{len(gold)} abstain answers identical "
                  f"on a second, uncached call")
        except TokenBudgetExceeded:
            print("  3 determinism           : SKIPPED, generation budget exhausted")
            problems.append("check 3: determinism not measured (budget)")
        print(f"    (determinism calls: {fresh.calls_made}, {fresh.tokens_used:,} tokens)")

    new = out[~out["cached"]]
    p_tok, c_tok = int(new["prompt_tokens"].sum()), int(new["completion_tokens"].sum())
    print(f"\nGENERATION: {len(new)} new answers this run, {p_tok:,} in / {c_tok:,} out "
          f"(~${cost(gen['model'], p_tok, c_tok):.3f}); {len(out) - len(new)} from cache")
    if args.dry_run:
        n_full = 74 * len(STRATEGIES)
        per = out["prompt_tokens"].sum() / len(out), out["completion_tokens"].sum() / len(out)
        proj = cost(gen["model"], per[0] * n_full, per[1] * n_full)
        print(f"  projected full generation: {per[0] * n_full + per[1] * n_full:,.0f} tokens, "
              f"~${proj:.2f}; protocol budget at 1.5x = "
              f"{1.5 * (per[0] + per[1]) * n_full:,.0f}")

    # --- judging -------------------------------------------------------------
    if args.judge:
        rc = run_judge(out, gold, texts, judge_model, args.judge_budget_tokens,
                       faith_path, args.dry_run, problems)
        if rc:
            return rc
    if args.judge_retest:
        rc = run_retest(out, gold, texts, judge_model, args.judge_budget_tokens)
        if rc:
            return rc

    print(f"\nwrote {answers_path.relative_to(PROJECT_ROOT)}")
    if problems:
        print("\nREPORT BEFORE READING ANY NUMBER:")
        for p in problems:
            print(f"  - {p}")
    return 0


def refusals(out: pd.DataFrame, dry_run: bool) -> pd.Series:
    """Which answers are refusals: the regex, corrected by the hand checks.

    Before labelling the regex is all there is. Once generation_checks.csv
    exists (27 --read), the owner's REFUSAL/ANSWER label overrides it, so a
    re-run of --judge picks up any answer the regex wrongly called a refusal;
    everything already judged comes from the checkpoint.
    """
    refused = out["abstained"].astype(str) == "True"
    path = TABLES_DIR / "generation_checks.csv"
    if dry_run or not path.exists():
        return refused
    checks = pd.read_csv(path, dtype=str)
    checks = checks[checks["check"] == "refusal"]
    hand = {(r["id"], r["strategy"]): r["final"] == "REFUSAL" for _, r in checks.iterrows()}
    keys = list(zip(out["id"], out["strategy"]))
    return pd.Series([hand.get(k, bool(f)) for k, f in zip(keys, refused)], index=out.index)


def run_judge(out, gold, texts, judge_model, budget, faith_path, dry_run, problems) -> int:
    questions = {q.id: q for q in gold}
    refused = refusals(out, dry_run)
    todo = out[(out["question_type"] != "unanswerable") & (~refused)]
    meter = fa.UsageMeter(max_total_tokens=budget)
    metric = fa.make_metric(judge_model, meter)
    ckpt = fa.Checkpoint(JUDGE_CHECKPOINT)
    print(f"\nJUDGING: {len(todo)} non-refusal scorable answers, judge {judge_model}, "
          f"budget {budget:,} tokens")
    records, new = [], 0
    for _, r in todo.iterrows():
        judged = fa.strip_confidence(r["answer"])   # key on the text actually judged
        key = fa.answer_key(r["id"], r["strategy"], judged, judge_model)
        if key not in ckpt:
            contexts = [texts[c] for c in r["retrieved_chunk_ids"].split("|")]
            try:
                j = asyncio.run(fa.judge(metric, key, questions[r["id"]].question,
                                         judged, contexts, meter))
            except Exception as e:          # stop; nothing partial is written
                print(f"\nJUDGING STOPPED at {r['id']} {r['strategy']}: "
                      f"{type(e).__name__}: {e}")
                print(f"  {len(ckpt.done)} judgments are in the checkpoint; "
                      f"re-run to resume.")
                return 3
            ckpt.add(j)
            new += 1
            print(f"  {r['id']} {r['strategy']:<19} {r['evidence']:<11} "
                  f"{j.n_supported}/{j.n_statements} supported  [{j.served_model}]")
        j = ckpt.done[key]
        records.append({"id": r["id"], "strategy": r["strategy"], "evidence": r["evidence"],
                        "key": key, "n_statements": j.n_statements,
                        "n_supported": j.n_supported,
                        "faithfulness": j.score if j.score is not None else "",
                        "judge_served_model": j.served_model})
    df = pd.DataFrame(records)
    df.to_csv(faith_path, index=False, encoding="utf-8")
    served = sorted({m for s in df["judge_served_model"] for m in str(s).split(";") if m})
    no_claims = int((df["n_statements"] == 0).sum()) if len(df) else 0
    print(f"\n  judged this run: {new}; from checkpoint: {len(df) - new}; "
          f"answers with no claims: {no_claims}")
    print(f"  5 judge served versions: {served or 'none recorded'}")
    if len(served) > 1:
        problems.append(f"check 5: judgments came from {len(served)} versions: {served}")
    print(f"  judge tokens this run: {meter.tokens_used:,} in {meter.calls} calls")
    spent = cost(judge_model, meter.prompt_tokens, meter.completion_tokens)
    print(f"    {meter.prompt_tokens:,} in / {meter.completion_tokens:,} out, ~${spent:.3f}")
    if dry_run and new:
        # Upper bound: every scorable answer judged, none refused.
        print(f"  projected: ~${spent / new:.4f} per judged answer; at most 174 answers, "
              f"~${174 * spent / new:.2f}, tokens {174 * meter.tokens_used / new:,.0f}")
    print(f"wrote {faith_path.relative_to(PROJECT_ROOT)}")
    return 0


def run_retest(out, gold, texts, judge_model, budget) -> int:
    """Judge each judged dry-run answer once more and compare (amendment)."""
    questions = {q.id: q for q in gold}
    ckpt = fa.Checkpoint(JUDGE_CHECKPOINT)
    meter = fa.UsageMeter(max_total_tokens=budget)
    metric = fa.make_metric(judge_model, meter)
    todo = out[(out["question_type"] != "unanswerable") & (~out["abstained"])]
    print(f"\nJUDGE TEST-RETEST: {len(todo)} answers judged a second time, same text, "
          f"same contexts, checkpoint bypassed")
    rows = []
    for _, r in todo.iterrows():
        judged = fa.strip_confidence(r["answer"])
        key = fa.answer_key(r["id"], r["strategy"], judged, judge_model)
        first = ckpt.done[key]
        contexts = [texts[c] for c in r["retrieved_chunk_ids"].split("|")]
        try:
            second = asyncio.run(fa.judge(metric, key, questions[r["id"]].question,
                                          judged, contexts, meter))
        except Exception as e:
            print(f"  RETEST STOPPED at {r['id']} {r['strategy']}: {type(e).__name__}: {e}")
            return 3
        fa.validate(second)
        same_split = first.statements == second.statements
        agree = (sum(a == b for a, b in zip(first.verdicts, second.verdicts))
                 if same_split else None)
        d = (abs(first.score - second.score)
             if first.score is not None and second.score is not None else None)
        rows.append({"id": r["id"], "strategy": r["strategy"],
                     "first": first.score, "second": second.score, "abs_diff": d,
                     "n_first": first.n_statements, "n_second": second.n_statements,
                     "same_statements": same_split,
                     "verdicts_agree": agree if agree is not None else "",
                     "judge_served_model": second.served_model})
        print(f"  {r['id']} {r['strategy']:<19} {first.n_supported}/{first.n_statements} -> "
              f"{second.n_supported}/{second.n_statements}  |d| "
              f"{'-' if d is None else f'{d:.3f}'}"
              f"{f'  same statements, {agree}/{first.n_statements} verdicts agree' if same_split else ''}")
    df = pd.DataFrame(rows)
    path = TABLES_DIR / "judge_retest_dryrun.csv"
    path.unlink(missing_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    diffs = df["abs_diff"].dropna()
    same = df[df["same_statements"]]
    n_v = int(same["n_first"].sum())
    print(f"\n  mean |d| {diffs.mean():.3f}, max |d| {diffs.max():.3f} over {len(diffs)} answers; "
          f"{int((diffs > 0).sum())} answers changed score")
    print(f"  same statement split on {len(same)}/{len(df)} answers"
          + (f"; on those, {int(same['verdicts_agree'].sum())}/{n_v} verdicts agree"
             if n_v else ""))
    spent = cost(judge_model, meter.prompt_tokens, meter.completion_tokens)
    print(f"  tokens {meter.tokens_used:,}, ~${spent:.3f}; wrote {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
