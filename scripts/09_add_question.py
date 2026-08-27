"""Append a gold question, fetching quotes from source so they are never copied.

Every quote corruption in this project came from moving text through a
console. Here the spans are the only thing typed by hand - short ASCII
numbers that cannot be silently mangled - and the quote is read straight
from the frozen extraction output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from rageval.config import EVAL_DIR, INTERIM_DIR
from rageval.evaluation.gold import (
    Evidence,
    GoldQuestion,
    QUESTION_TYPES,
    load_gold,
    save_gold,
    validate_question,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_SPAN = re.compile(r"^(?P<paper>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")


def load_texts() -> dict[str, str]:
    texts = {}
    for path in sorted(INTERIM_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        texts[doc["arxiv_id"]] = doc["text"]
    return texts


def parse_evidence(specs: list[str], texts: dict[str, str]) -> list[Evidence]:
    out = []
    for spec in specs:
        m = _SPAN.match(spec.strip())
        if not m:
            raise SystemExit(f"Bad span '{spec}'. Expected ARXIVID:START-END, "
                             f"e.g. 2212.12794v2:63364-63662")
        paper, start, end = m["paper"], int(m["start"]), int(m["end"])
        if paper not in texts:
            raise SystemExit(f"Paper '{paper}' not in corpus.")
        text = texts[paper]
        if end > len(text) or start >= end:
            raise SystemExit(f"Span {start}-{end} invalid for {paper} "
                             f"(document is {len(text)} chars).")
        out.append(Evidence(paper, start, end, " ".join(text[start:end].split())))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id", required=True)
    p.add_argument("--type", required=True, choices=QUESTION_TYPES)
    p.add_argument("--question", required=True)
    p.add_argument("--answer", required=True)
    p.add_argument("--evidence", nargs="*", default=[], metavar="PAPER:START-END",
                   help="Spans from 06_find_evidence.py. Omit for unanswerable questions.")
    p.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    p.add_argument("--note", default="")
    p.add_argument("--file", default="gold_questions.jsonl")
    p.add_argument("--replace", action="store_true", help="Overwrite an existing id.")
    args = p.parse_args()

    texts = load_texts()
    path = EVAL_DIR / args.file
    questions = load_gold(path)

    existing = [q for q in questions if q.id == args.id]
    if existing and not args.replace:
        raise SystemExit(f"'{args.id}' already exists. Use --replace to overwrite.")

    question = GoldQuestion(
        id=args.id,
        question=args.question,
        question_type=args.type,
        reference_answer=args.answer,
        evidence=parse_evidence(args.evidence, texts),
        difficulty=args.difficulty,
        note=args.note,
        verified=True,
    )

    problems = validate_question(question, texts)
    if problems:
        print(f"REJECTED - {args.id} has problems:")
        for prob in problems:
            print(f"  - {prob}")
        return 1

    questions = [q for q in questions if q.id != args.id] + [question]
    questions.sort(key=lambda q: q.id)
    save_gold(questions, path)

    print(f"Added {args.id} ({args.type}, {len(question.evidence)} span(s)) "
          f"-> {len(questions)} questions total\n")
    for ev in question.evidence:
        preview = ev.quote if len(ev.quote) <= 200 else ev.quote[:200] + " ..."
        print(f"  {ev.arxiv_id}  {ev.char_start}-{ev.char_end}")
        print(f"    {preview}\n")
    print("READ THE QUOTES ABOVE. The span is right only if they support your answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())