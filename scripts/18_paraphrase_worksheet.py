"""Build - and then validate - the paraphrase test's rewritten questions.

WHY THIS EXISTS

Sweep 2 found BM25 beats dense retrieval by +0.168 span recall, and then
found that advantage concentrated in questions that reuse their evidence's
vocabulary (low-overlap tercile +0.037, CI [-0.115, +0.208]). Every gold
question was authored while reading its evidence, so the gold set hands a
lexical retriever the exact tokens the writer had in front of them. That
confound bounds the headline number and no amount of resampling can see it.

This test attacks the confound instead of describing it: rewrite the
highest-overlap questions to ask the same thing in different words, and
measure whether BM25's advantage survives.

WHY IT HAS MORE POWER THAN THE TERCILE SPLIT

The tercile comparison was BETWEEN groups - 19 questions against 20
different ones, differing in content as well as in overlap. This is WITHIN
question: the same question in two phrasings, scored by both retrievers.
The question's difficulty, evidence and type are held constant by
construction, so the only thing that varies is the wording.

HOW THE RULES ARE KEPT HONEST

The writer knows the hypothesis, which is a bias risk. Three defences:

  1. The banned-token list is COMPUTED, not judged. For each question the
     script bans the shared terms carrying the top 80% of the overlapping
     IDF mass - the exact quantity BM25 scores on - so "avoid the evidence's
     vocabulary" becomes a checkable list rather than an instinct.
  2. Compliance is checked mechanically on `--build`. A paraphrase still
     containing a banned term is rejected, not quietly accepted.
  3. The worksheet carries NO retrieval results. Which questions BM25
     currently gets right is deliberately withheld, so a rewrite cannot be
     unconsciously aimed.

An escape hatch exists because some questions become ambiguous once their
subject is unnameable: `keep: <term>` declares a retained term. Declared
exceptions are counted and reported; silent ones are violations.

USAGE

    python scripts/18_paraphrase_worksheet.py                 # emit worksheet
    ... fill in every `paraphrase>` block by hand ...
    python scripts/18_paraphrase_worksheet.py --build         # validate + write gold
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from math import log
from pathlib import Path

import pandas as pd

from rageval.config import CHROMA_DIR, EVAL_DIR, TABLES_DIR
from rageval.retrieve.bm25 import tokenize
from rageval.store.chroma_store import ChromaStore

WORKSHEET = EVAL_DIR / "paraphrase_worksheet.txt"
PARA_GOLD = EVAL_DIR / "gold_paraphrased.jsonl"
BLOCK = re.compile(r"^### (\S+)\s*$")


# --- overlap machinery, shared with 17_lexical_overlap.py ------------------

def build_idf(texts):
    df, n = Counter(), 0
    for t in texts:
        n += 1
        for term in set(tokenize(t)):
            df[term] += 1
    idf = {t: log((n - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}
    return idf, n, df


def idf_of(term, idf, ceiling):
    return idf.get(term, ceiling)


def overlap_and_shared(question, quotes, idf, ceiling):
    """IDF-weighted overlap, and the shared terms sorted by weight."""
    q_terms = set(tokenize(question))
    if not q_terms:
        return 0.0, []
    e_terms: set[str] = set()
    for quote in quotes:
        e_terms |= set(tokenize(quote))
    total = sum(idf_of(t, idf, ceiling) for t in q_terms)
    shared = sorted(q_terms & e_terms, key=lambda t: -idf_of(t, idf, ceiling))
    mass = sum(idf_of(t, idf, ceiling) for t in shared)
    return (mass / total if total else 0.0), shared


def banned_terms(shared, idf, ceiling, df, n_docs, mass_share=0.80, cap=20,
                 max_df=0.20):
    """The shared terms carrying the top `mass_share` of overlapping IDF.

    Banning every shared term would forbid "the" and "of"; banning only the
    rarest would leave most of the signal intact. This bans the mass BM25
    actually scores on, which is the quantity the confound is made of.

    Terms appearing in more than `max_df` of the corpus are never banned.
    They are function words and generic domain vocabulary that BM25 already
    scores at almost nothing, and forbidding them would make the rewriting
    task absurd without removing any lexical signal.
    """
    eligible = [t for t in shared if df.get(t, 0) <= max_df * n_docs]
    total = sum(idf_of(t, idf, ceiling) for t in eligible)
    if total <= 0:
        return []
    out, acc = [], 0.0
    for t in eligible:
        out.append(t)
        acc += idf_of(t, idf, ceiling)
        if acc / total >= mass_share or len(out) >= cap:
            break
    return out


def load_raw_gold(path: Path) -> list[dict]:
    """Read the gold file as plain dicts.

    Deliberately not `load_gold`: the paraphrased file must preserve every
    field byte-for-byte except `question`, and round-tripping through a
    dataclass would silently drop anything the dataclass does not model.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  gold line {i} is not valid JSON: {exc}", file=sys.stderr)
                raise
    return rows


def corpus_idf(collection: str):
    store = ChromaStore(CHROMA_DIR, collection)
    records = store.all_records()
    if not records:
        raise SystemExit("Collection is empty; build the index first.")
    idf, n, df = build_idf(r.text for r in records)
    return idf, n, max(idf.values()), df


# --- emit ------------------------------------------------------------------

def emit(args) -> int:
    idf, n_docs, ceiling, df = corpus_idf(args.collection)
    gold = [g for g in load_raw_gold(EVAL_DIR / "gold_questions.jsonl")
            if g.get("evidence")]

    scored = []
    for g in gold:
        quotes = [e["quote"] for e in g["evidence"]]
        ov, shared = overlap_and_shared(g["question"], quotes, idf, ceiling)
        scored.append((ov, shared, g))
    scored.sort(key=lambda t: -t[0])
    chosen = scored[: args.n]

    lines = [
        "# PARAPHRASE WORKSHEET",
        "#",
        "# Rewrite each question to ask exactly the same thing in different",
        "# words. Type your version on the blank line after `paraphrase>`.",
        "#",
        "# RULES (fixed before any rewriting; see docs/paraphrase-protocol.md)",
        "#  1. The answer must not change. The same evidence must still",
        "#     answer it, and the reference answer must still be correct.",
        "#  2. Use NONE of the terms listed under `avoid:`. Sub-forms count -",
        "#     banning `ecmwf-ifs` also bans `ECMWF` and `IFS`.",
        "#  3. Describe rather than name. 'the graph neural network weather",
        "#     model' instead of a product name, where the description is",
        "#     unambiguous within this corpus.",
        "#  4. Keep it a natural question someone would actually type.",
        "#  5. If a question becomes genuinely ambiguous without one banned",
        "#     term, add a line `keep: <term>` in the block. Declared",
        "#     exceptions are counted; undeclared ones are rejected.",
        "#",
        "# Do not reorder or delete blocks. Run --build when finished.",
        "",
    ]
    for ov, shared, g in chosen:
        ban = banned_terms(shared, idf, ceiling, df, n_docs,
                           args.mass, args.cap, args.max_df)
        lines.append(f"### {g['id']}")
        lines.append(f"# type    : {g.get('question_type', '?')}")
        lines.append(f"# overlap : {ov:.3f}")
        lines.append(f"# original: {g['question']}")
        ans = g.get("reference_answer", "")
        lines.append(f"# answer  : {ans}")
        for i, e in enumerate(g["evidence"]):
            quote = " ".join(e["quote"].split())
            lines.append(f"# evidence{i}: {quote[:400]}{'...' if len(quote) > 400 else ''}")
        lines.append(f"avoid: {', '.join(ban)}")
        lines.append("paraphrase>")
        lines.append("")
        lines.append("")

    WORKSHEET.parent.mkdir(parents=True, exist_ok=True)
    WORKSHEET.write_text("\n".join(lines), encoding="utf-8")

    print(f"{'=' * 78}\nPARAPHRASE WORKSHEET\n{'=' * 78}")
    print(f"  corpus        {n_docs} chunks, {len(idf):,} terms")
    print(f"  selected      {len(chosen)} of {len(gold)} questions, highest overlap first")
    print(f"  overlap range {chosen[-1][0]:.3f} to {chosen[0][0]:.3f}")
    types = Counter(g.get("question_type", "?") for _, _, g in chosen)
    print(f"  types         {', '.join(f'{k} {v}' for k, v in sorted(types.items()))}")
    print(f"  banned terms  {args.mass:.0%} of shared IDF mass, capped at {args.cap}, "
          f"skipping terms in >{args.max_df:.0%} of the corpus")
    bans = [len(banned_terms(sh, idf, ceiling, df, n_docs, args.mass, args.cap,
                             args.max_df)) for _, sh, _ in chosen]
    print(f"  per question  min {min(bans)}, median {sorted(bans)[len(bans) // 2]}, "
          f"max {max(bans)} terms to avoid")
    print(f"\n  -> {WORKSHEET}")
    print("\n  The worksheet contains NO retrieval results, deliberately: a")
    print("  rewrite must not be aimable at questions BM25 currently misses.")
    return 0


# --- build -----------------------------------------------------------------

def parse_worksheet(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"No worksheet at {path}. Run without --build first.")
    blocks, cur = [], None
    for raw in path.read_text(encoding="utf-8").split("\n"):
        m = BLOCK.match(raw)
        if m:
            if cur:
                blocks.append(cur)
            cur = {"id": m.group(1), "avoid": [], "keep": [], "para": [],
                   "collecting": False}
            continue
        if cur is None:
            continue
        if raw.startswith("#"):
            continue
        if raw.startswith("avoid:"):
            cur["avoid"] = [t.strip() for t in raw[6:].split(",") if t.strip()]
            continue
        if raw.startswith("keep:"):
            cur["keep"] += [t.strip().lower() for t in raw[5:].split(",") if t.strip()]
            continue
        if raw.startswith("paraphrase>"):
            cur["collecting"] = True
            rest = raw[len("paraphrase>"):].strip()
            if rest:
                cur["para"].append(rest)
            continue
        if cur["collecting"]:
            cur["para"].append(raw)
    if cur:
        blocks.append(cur)
    for b in blocks:
        b["paraphrase"] = " ".join(" ".join(b["para"]).split())
    return blocks


def build(args) -> int:
    idf, n_docs, ceiling, _df = corpus_idf(args.collection)
    gold = {g["id"]: g for g in load_raw_gold(EVAL_DIR / "gold_questions.jsonl")}
    blocks = parse_worksheet(WORKSHEET)

    rows, problems, kept_total = [], [], 0
    for b in blocks:
        g = gold.get(b["id"])
        if g is None:
            problems.append(f"{b['id']}: not in the gold set")
            continue
        if not b["paraphrase"]:
            problems.append(f"{b['id']}: paraphrase is empty")
            continue

        terms = set(tokenize(b["paraphrase"]))
        declared = set(b["keep"])
        violations = sorted((set(b["avoid"]) & terms) - declared)
        kept = sorted(set(b["avoid"]) & terms & declared)
        kept_total += len(kept)
        if violations:
            problems.append(f"{b['id']}: uses banned term(s) {', '.join(violations)}")

        quotes = [e["quote"] for e in g["evidence"]]
        before, _ = overlap_and_shared(g["question"], quotes, idf, ceiling)
        after, _ = overlap_and_shared(b["paraphrase"], quotes, idf, ceiling)
        rows.append({
            "id": b["id"], "question_type": g.get("question_type", "?"),
            "overlap_before": before, "overlap_after": after,
            "delta": after - before, "n_banned": len(b["avoid"]),
            "kept_declared": ";".join(kept),
            "original": g["question"], "paraphrase": b["paraphrase"],
        })

    if problems:
        print(f"{'=' * 78}\nNOT WRITTEN - {len(problems)} problem(s)\n{'=' * 78}")
        for p in problems:
            print(f"  {p}")
        print("\n  Fix the worksheet and re-run. Nothing is written while any")
        print("  block fails: a partially-compliant set would quietly weaken")
        print("  the manipulation without showing up in the result.")
        return 1

    df = pd.DataFrame(rows)
    out_csv = TABLES_DIR / "paraphrase_overlap.csv"
    df.to_csv(out_csv, index=False)

    para_by_id = {r["id"]: r["paraphrase"] for r in rows}
    with open(PARA_GOLD, "w", encoding="utf-8") as fh:
        for g in sorted(gold.values(), key=lambda d: d["id"]):
            out = dict(g)
            if g["id"] in para_by_id:
                out["question"] = para_by_id[g["id"]]
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"{'=' * 78}\nMANIPULATION CHECK\n{'=' * 78}")
    print(f"  rewritten            {len(df)} questions")
    print(f"  overlap before       mean {df['overlap_before'].mean():.3f}  "
          f"median {df['overlap_before'].median():.3f}")
    print(f"  overlap after        mean {df['overlap_after'].mean():.3f}  "
          f"median {df['overlap_after'].median():.3f}")
    drop = -df["delta"].mean()
    print(f"  mean drop            {drop:.3f}")
    print(f"  declared exceptions  {kept_total} term(s) across "
          f"{(df['kept_declared'] != '').sum()} question(s)")
    worst = df.nsmallest(3, "delta", keep="all").iloc[::-1]
    print(f"\n  smallest drops (check these read as genuine rewrites):")
    for _, r in worst.iterrows():
        print(f"    {r['id']:<6} {r['overlap_before']:.3f} -> {r['overlap_after']:.3f}")

    print(f"\n{'-' * 78}\nVERDICT\n{'-' * 78}")
    if drop < args.min_drop:
        print(f"  MANIPULATION FAILED. Mean drop {drop:.3f} is below the "
              f"pre-registered {args.min_drop:.2f}.")
        print("  The rewrites did not remove enough of the shared vocabulary, so")
        print("  the test cannot distinguish anything. Rewrite further - do NOT")
        print("  reinterpret the threshold after seeing it.")
    else:
        print(f"  MANIPULATION SUCCEEDED. Mean drop {drop:.3f} clears the "
              f"pre-registered {args.min_drop:.2f}.")
        print("  The paraphrased gold set is valid to score against.")

    print(f"\n  -> {PARA_GOLD}")
    print(f"  -> {out_csv}")
    print("\n  Next, holding everything else identical:")
    print(f"    python scripts/10_eval_retrieval.py --config baseline.yaml "
          f"--gold {PARA_GOLD} --tag para_dense --k-max 50")
    print(f"    python scripts/10_eval_retrieval.py --config sweep_bm25.yaml "
          f"--gold {PARA_GOLD} --tag para_bm25 --k-max 50")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true",
                   help="Validate the filled worksheet and write the gold file.")
    p.add_argument("-n", type=int, default=20, help="Questions to rewrite.")
    p.add_argument("--collection", default="baseline_fixed512_minilm")
    p.add_argument("--mass", type=float, default=0.80,
                   help="Share of overlapping IDF mass to ban.")
    p.add_argument("--cap", type=int, default=20, help="Max banned terms per question.")
    p.add_argument("--max-df", type=float, default=0.20,
                   help="Never ban a term appearing in more than this share "
                        "of the corpus; BM25 scores those near zero anyway.")
    p.add_argument("--min-drop", type=float, default=0.25,
                   help="Pre-registered minimum mean overlap drop for the "
                        "manipulation to count as successful.")
    args = p.parse_args()
    return build(args) if args.build else emit(args)


if __name__ == "__main__":
    raise SystemExit(main())