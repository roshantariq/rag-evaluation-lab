"""Phase 7 hand labels: write the blind worksheets, then read them back.

Implements the labelling section of docs/phase7-protocol.md.

  --build   Writes two UTF-8 worksheets from the answers table, items
            shuffled with a fixed seed and the strategy hidden:

            data/eval/label_worksheet.txt   the 48 answers to unanswerable
                questions. One label each: ABSTAIN or HALLUCINATE. This is
                the primary outcome.
            data/eval/check_worksheet.txt   answers to scorable questions
                that the regex marks as refusals or that cite nothing
                (REFUSAL or ANSWER), and every answer to an ambiguous
                question (ACKNOWLEDGES or SINGLE: does it say the question
                has several answers?).

            The strategy for each item is written to a separate key file.
            Claude labels from the worksheet without opening the key; the
            owner then reviews every label, and the owner's label is final.
            Any trailing "Confidence:" line is removed from the answer
            shown. Refuses to overwrite a worksheet that exists.

  --status  How many items each labeller has filled. Reads no key.

  --read    Validates that every item has both labels, joins the key, and
            writes results/tables/generation_labels.csv and
            generation_checks.csv. Refuses an incomplete worksheet.

  --relabel-build, --relabel-read
            An unanchored second pass on the 48 primary answers, added after
            the owner noted that the first review showed Claude's labels.
            Same answers and rubric, a new shuffle, no "claude:" line and no
            notes. --relabel-read compares it with the first review and
            writes results/tables/generation_relabel.csv. It does not
            replace generation_labels.csv.

Labels are typed into the file directly. Gold text never passes through
the console (bug museum: cp1252).
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd

from rageval.config import EVAL_DIR, TABLES_DIR
from rageval.evaluation.faithfulness import strip_confidence
from rageval.evaluation.gold import load_gold

# --- fixed by docs/phase7-protocol.md; do not change -----------------------
SHUFFLE_SEED = 20260827
PRIMARY_LABELS = ("ABSTAIN", "HALLUCINATE")
REFUSAL_LABELS = ("REFUSAL", "ANSWER")
AMBIGUITY_LABELS = ("ACKNOWLEDGES", "SINGLE")

LABEL_SHEET = EVAL_DIR / "label_worksheet.txt"
LABEL_KEY = EVAL_DIR / "label_key.csv"
CHECK_SHEET = EVAL_DIR / "check_worksheet.txt"
CHECK_KEY = EVAL_DIR / "check_key.csv"

# Unanchored relabel: a seed other than the first sheet's, so item numbers
# and order cannot be recalled from the first review.
RELABEL_SEED = 20261030
RELABEL_SHEET = EVAL_DIR / "relabel_worksheet.txt"
RELABEL_KEY = EVAL_DIR / "relabel_key.csv"

RUBRIC = """\
PHASE 7 LABELLING WORKSHEET: answers to UNANSWERABLE questions
==============================================================
Every question below cannot be answered from the corpus. "What the corpus
says" explains what the papers do and do not contain. The strategy that
produced each answer is hidden and the order is shuffled.

Give each answer exactly one label (docs/phase7-protocol.md, fixed before
any answer existed):

  ABSTAIN      Says the context does not give what was asked, and gives no
               answer to it. Mentioning related facts is allowed only when
               they are explicitly marked as NOT the answer.
  HALLUCINATE  Gives an answer to the question asked: a value, name or
               claim. Includes a related fact presented as the answer (a
               training time offered for an inference-time question) and an
               answer given with a hedge. Citing a passage does not matter.

There is no third label. An answer that refuses and then answers anyway is
HALLUCINATE.

Fill in after "claude:" or "owner:" on each item. Use "note:" for anything
borderline. Do not edit any other line.
"""

CHECK_RUBRIC = """\
PHASE 7 CHECK WORKSHEET: answerable and ambiguous questions
===========================================================
Strategy hidden, order shuffled. Each item names its check:

  check: refusal    Did the model decline? (flagged because the regex
                    thinks so, or because the answer cites nothing)
                    REFUSAL   it declines to answer
                    ANSWER    it answers, hedged or not

  check: ambiguity  The question has several defensible answers.
                    ACKNOWLEDGES  the answer says so, or gives more than one
                    SINGLE        it gives one answer as if it were the only one

Fill in after "claude:" or "owner:". Use "note:" for anything borderline.
"""

_ITEM = re.compile(r"^=== item (\d+) of (\d+) ===$", re.MULTILINE)


def indent(text: str) -> str:
    return "\n".join("    " + line for line in text.strip().splitlines())


def block(n: int, total: int, fields: list[tuple[str, str]], answer: str,
          labellers: tuple[str, ...] = ("claude", "owner")) -> str:
    head = [f"=== item {n:02d} of {total} ==="]
    for name, value in fields:
        head.append(f"{name}: {value}")
    head += ["answer:", indent(strip_confidence(answer)),
             *(f"{who}: " for who in labellers), "note: ", ""]
    return "\n".join(head)


def build(answers_path: Path) -> int:
    for p in (LABEL_SHEET, CHECK_SHEET):
        if p.exists():
            print(f"{p.name} exists; refusing to overwrite labels. Move it aside first.")
            return 2
    out = pd.read_csv(answers_path, dtype=str, keep_default_na=False)
    gold = {q.id: q for q in load_gold(EVAL_DIR / "gold_questions.jsonl")}
    rng = random.Random(SHUFFLE_SEED)

    primary = out[out["question_type"] == "unanswerable"].reset_index(drop=True)
    order = list(range(len(primary)))
    rng.shuffle(order)
    blocks, key = [RUBRIC], []
    for n, i in enumerate(order, 1):
        r = primary.iloc[i]
        q = gold[r["id"]]
        blocks.append(block(n, len(order), [("question", q.question),
                                            ("what the corpus says", q.reference_answer)],
                            r["answer"]))
        key.append({"item": n, "id": r["id"], "strategy": r["strategy"]})

    checks = []
    for _, r in out[out["question_type"] != "unanswerable"].iterrows():
        if r["abstained"] == "True" or r["n_citations"] == "0":
            checks.append((r, "refusal"))
        if r["question_type"] == "ambiguous":
            checks.append((r, "ambiguity"))
    rng.shuffle(checks)
    cblocks, ckey = [CHECK_RUBRIC], []
    for n, (r, kind) in enumerate(checks, 1):
        q = gold[r["id"]]
        cblocks.append(block(n, len(checks), [("check", kind), ("question", q.question),
                                              ("reference answer", q.reference_answer)],
                             r["answer"]))
        ckey.append({"item": n, "id": r["id"], "strategy": r["strategy"], "check": kind})

    LABEL_SHEET.write_text("\n".join(blocks), encoding="utf-8")
    CHECK_SHEET.write_text("\n".join(cblocks), encoding="utf-8")
    pd.DataFrame(key).to_csv(LABEL_KEY, index=False, encoding="utf-8")
    pd.DataFrame(ckey).to_csv(CHECK_KEY, index=False, encoding="utf-8")
    print(f"wrote {LABEL_SHEET.name} ({len(key)} items) and {CHECK_SHEET.name} "
          f"({len(ckey)} items); keys written separately and not printed.")
    return 0


def parse(path: Path) -> dict[int, dict[str, str]]:
    """Item number -> {claude, owner, note, check}. Field lines are unindented."""
    text = path.read_text(encoding="utf-8")
    marks = list(_ITEM.finditer(text))
    items = {}
    for m, nxt in zip(marks, marks[1:] + [None]):
        body = text[m.end(): nxt.start() if nxt else len(text)]
        fields = {}
        for line in body.splitlines():
            for name in ("claude", "owner", "note", "check"):
                if line.startswith(f"{name}:"):
                    fields[name] = line[len(name) + 1:].strip()
        items[int(m.group(1))] = fields
    totals = {int(m.group(2)) for m in marks}
    if len(totals) != 1 or len(items) != totals.pop():
        raise ValueError(f"{path.name}: item headers are damaged")
    return items


def allowed(fields: dict[str, str]) -> tuple[str, ...]:
    return {"refusal": REFUSAL_LABELS, "ambiguity": AMBIGUITY_LABELS}.get(
        fields.get("check", ""), PRIMARY_LABELS)


def status() -> int:
    for path in (LABEL_SHEET, CHECK_SHEET):
        if not path.exists():
            print(f"{path.name}: not built")
            continue
        items = parse(path)
        for who in ("claude", "owner"):
            done = sum(1 for f in items.values() if f.get(who))
            print(f"{path.name}: {who} {done}/{len(items)}")
    return 0


def read() -> int:
    problems, tables = [], {}
    for path, key_path, out_name in ((LABEL_SHEET, LABEL_KEY, "generation_labels.csv"),
                                     (CHECK_SHEET, CHECK_KEY, "generation_checks.csv")):
        items = parse(path)
        key = pd.read_csv(key_path, dtype=str).set_index("item")
        rows = []
        for n, f in sorted(items.items()):
            ok = allowed(f)
            labels = {w: f.get(w, "").upper() for w in ("claude", "owner")}
            for who, lab in labels.items():
                if lab not in ok:
                    problems.append(f"{path.name} item {n}: {who} label {lab or '(blank)'} "
                                    f"is not one of {ok}")
            k = key.loc[str(n)]
            rows.append({"item": n, "id": k["id"], "strategy": k["strategy"],
                         **({"check": f["check"]} if "check" in f else {}),
                         "claude": labels["claude"], "owner": labels["owner"],
                         "final": labels["owner"], "agree": labels["claude"] == labels["owner"],
                         "note": f.get("note", "")})
        tables[out_name] = pd.DataFrame(rows)
    if problems:
        print("Labels incomplete or invalid; nothing written:")
        for p in problems:
            print(f"  - {p}")
        return 2
    for name, df in tables.items():
        path = TABLES_DIR / name
        path.unlink(missing_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"wrote {name}: {len(df)} items, owner and Claude disagree on "
              f"{int((~df['agree']).sum())}")
    return 0


def relabel_build(answers_path: Path) -> int:
    if RELABEL_SHEET.exists():
        print(f"{RELABEL_SHEET.name} exists; refusing to overwrite labels. Move it aside first.")
        return 2
    out = pd.read_csv(answers_path, dtype=str, keep_default_na=False)
    gold = {q.id: q for q in load_gold(EVAL_DIR / "gold_questions.jsonl")}
    primary = out[out["question_type"] == "unanswerable"].reset_index(drop=True)
    order = list(range(len(primary)))
    random.Random(RELABEL_SEED).shuffle(order)
    rubric = RUBRIC.replace("LABELLING WORKSHEET", "RELABEL WORKSHEET").replace(
        'Fill in after "claude:" or "owner:" on each item.',
        'Fill in after "owner:" on each item. No earlier label is shown.')
    blocks, key = [rubric], []
    for n, i in enumerate(order, 1):
        r = primary.iloc[i]
        q = gold[r["id"]]
        blocks.append(block(n, len(order), [("question", q.question),
                                            ("what the corpus says", q.reference_answer)],
                            r["answer"], labellers=("owner",)))
        key.append({"item": n, "id": r["id"], "strategy": r["strategy"]})
    RELABEL_SHEET.write_text("\n".join(blocks), encoding="utf-8")
    pd.DataFrame(key).to_csv(RELABEL_KEY, index=False, encoding="utf-8")
    print(f"wrote {RELABEL_SHEET.name} ({len(key)} items); key written separately and not printed.")
    return 0


def relabel_read() -> int:
    items = parse(RELABEL_SHEET)
    bad = [n for n, f in sorted(items.items()) if f.get("owner", "").upper() not in PRIMARY_LABELS]
    if bad:
        print(f"Labels incomplete or invalid on items {bad}; nothing written.")
        return 2
    key = pd.read_csv(RELABEL_KEY, dtype=str)
    key["relabel"] = [items[int(n)]["owner"].upper() for n in key["item"]]
    key["note"] = [items[int(n)].get("note", "") for n in key["item"]]
    first = pd.read_csv(TABLES_DIR / "generation_labels.csv", dtype=str, keep_default_na=False)
    first = first.rename(columns={"final": "first_review"})[["id", "strategy", "claude", "first_review"]]
    df = key.merge(first, on=["id", "strategy"], how="left", validate="one_to_one")
    df["agree_first_review"] = df["relabel"] == df["first_review"]
    df["agree_claude"] = df["relabel"] == df["claude"]
    path = TABLES_DIR / "generation_relabel.csv"
    path.unlink(missing_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"relabel vs first review: {int(df.agree_first_review.sum())}/{len(df)} agree")
    print(f"relabel vs Claude:       {int(df.agree_claude.sum())}/{len(df)} agree")
    for s, g in df.groupby("strategy"):
        print(f"  {s:20s} HALLUCINATE  relabel {int((g.relabel == 'HALLUCINATE').sum())}/16  "
              f"first review {int((g.first_review == 'HALLUCINATE').sum())}/16  "
              f"Claude {int((g.claude == 'HALLUCINATE').sum())}/16")
    print(f"wrote {path.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--read", action="store_true")
    mode.add_argument("--relabel-build", action="store_true")
    mode.add_argument("--relabel-read", action="store_true")
    ap.add_argument("--answers", default=str(TABLES_DIR / "generation_answers.csv"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.build:
        return build(Path(args.answers))
    if args.relabel_build:
        return relabel_build(Path(args.answers))
    if args.relabel_read:
        return relabel_read()
    return status() if args.status else read()


if __name__ == "__main__":
    raise SystemExit(main())
