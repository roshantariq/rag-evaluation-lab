"""Validate the gold evaluation set against the frozen extraction output."""

from __future__ import annotations

import argparse
import json
import sys

from rageval.config import EVAL_DIR, INTERIM_DIR
from rageval.evaluation.gold import TARGET_COUNTS, load_gold, validate_set


def load_all_texts() -> dict[str, str]:
    texts = {}
    for path in sorted(INTERIM_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        texts[doc["arxiv_id"]] = doc["text"]
    return texts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", default="gold_questions.jsonl")
    p.add_argument("--quiet", action="store_true", help="Only show problems.")
    args = p.parse_args()

    path = EVAL_DIR / args.file
    questions = load_gold(path)
    if not questions:
        print(f"No questions found at {path}")
        return 1

    texts = load_all_texts()
    report = validate_set(questions, texts)

    print(f"\n{'=' * 70}\nGOLD SET VALIDATION  ({report['n_total']} questions)\n{'=' * 70}")
    print(f"  passing        {report['n_ok']}/{report['n_total']}")
    print(f"  papers covered {report['papers_covered']}/{len(texts)}")

    print("\n  composition          have  target")
    for qtype, target in TARGET_COUNTS.items():
        have = report["counts"].get(qtype, 0)
        flag = "" if have >= target else f"   need {target - have} more"
        print(f"    {qtype:<18} {have:4d}  {target:4d}{flag}")

    failing = {qid: probs for qid, probs in report["per_question"].items() if probs}
    if failing:
        print(f"\n{'=' * 70}\n{len(failing)} QUESTION(S) WITH PROBLEMS\n{'=' * 70}")
        for qid, probs in failing.items():
            print(f"\n  {qid}")
            for prob in probs:
                print(f"    - {prob}")
    elif not args.quiet:
        print("\n  All questions valid.")

    total = sum(TARGET_COUNTS.values())
    if report["n_ok"] == report["n_total"] and report["n_total"] >= total:
        print(f"\n  Gold set complete and valid. Phase 4 done.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())