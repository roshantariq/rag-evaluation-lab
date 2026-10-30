"""Apply the Phase 7 protocol's outcome table to the labelled answers.

Implements docs/phase7-protocol.md and nothing else. Written and tested
before the full run exists, like 19-25. The constants that decide the
outcome are module constants, not flags, and tests pin them.

WHAT IT READS

    generation_answers.csv        from 26 (answers, evidence split, citations)
    generation_labels.csv         from 27 --read (48 hand labels, owner final)
    generation_checks.csv         from 27 --read (refusal and ambiguity checks)
    generation_faithfulness.csv   from 26 --judge

WHAT IT DECIDES (primary, 16 unanswerable questions)

    floor             naive hallucinates on 2 or fewer of 16: checked first
    reduces           CI on naive - abstain excludes zero, positive
    not_separable     CI includes zero ("not separable" is not "no effect")
    increases         CI excludes zero, negative

  Bootstrap over the 16 questions, 10,000 resamples, seed 20260827, 95%
  percentile interval. McNemar exact is reported beside it and decides
  nothing; Holm corrects it across the two pairs (naive vs abstain,
  abstain_confidence vs abstain). Only naive vs abstain decides.

WHAT IT REPORTS AND DOES NOT DECIDE

  1 false abstention on scorable questions, split complete / incomplete;
    abstain - naive on complete questions, same bootstrap
  2 faithfulness (refusals excluded), split; naive vs abstain paired on the
    questions both answered; a difference below the judge's measured
    test-retest noise (0.083) is labelled within_judge_noise
  3 citation precision and recall against gold spans, out-of-range cites
  4 stated confidence against the label or the evidence split
  and the ambiguity acknowledgement count.

Writes results/tables/generation_result.csv.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rageval.config import TABLES_DIR
from rageval.evaluation.spans import span_indices
from rageval.evaluation.stats import binom_two_sided, holm
from rageval.provenance import record

# --- pre-registered in docs/phase7-protocol.md; do not change --------------
STRATEGIES = ("naive", "abstain", "abstain_confidence")
DECIDING_PAIR = ("naive", "abstain")
PAIRS = (("naive", "abstain"), ("abstain_confidence", "abstain"))
N_UNANSWERABLE = 16
FLOOR = 2
BOOT = 10_000
SEED = 20260827
JUDGE_NOISE = 0.083          # mean test-retest |delta|, judge_retest_dryrun.csv
N_ANSWERS = 222

_CONFIDENCE = re.compile(r"Confidence:\s*(high|medium|low)", re.IGNORECASE)


def classify(naive_count: int, lo: float, hi: float) -> str:
    """The protocol's outcome table. Floor is checked first, on naive alone."""
    if naive_count <= FLOOR:
        return "floor"
    if lo > 0:
        return "reduces"
    if hi < 0:
        return "increases"
    return "not_separable"


def paired_ci(a: np.ndarray, b: np.ndarray, draws: np.ndarray) -> tuple[float, float, float]:
    """Mean of a - b with a percentile CI; draws are shared question indices."""
    diff = a - b
    boot = diff[draws].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def draws_for(n: int, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(BOOT, n))


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Discordant counts (a only, b only) and the exact two-sided p."""
    only_a = int(((a == 1) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    return only_a, only_b, binom_two_sided(only_a, only_b)


def wide(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """id x strategy table, strategies in protocol order."""
    return df.pivot(index="id", columns="strategy", values=value)[list(STRATEGIES)]


def primary(labels: pd.DataFrame, rows: list[dict]) -> str:
    h = wide(labels.assign(h=(labels["final"] == "HALLUCINATE").astype(int)), "h")
    if len(h) != N_UNANSWERABLE or h.isna().any().any():
        raise ValueError(f"expected {N_UNANSWERABLE} questions x 3 labels, got {h.shape}")
    print(f"PRIMARY: hallucination on {N_UNANSWERABLE} unanswerable questions "
          f"(owner's labels final)")
    for s in STRATEGIES:
        n = int(h[s].sum())
        print(f"  {s:<19} {n:>2}/{N_UNANSWERABLE}  {n / N_UNANSWERABLE:.3f}")
        rows.append({"section": "primary", "measure": "hallucination_rate", "arm": s,
                     "value": n / N_UNANSWERABLE, "n": N_UNANSWERABLE, "count": n})
    agree = labels["agree"].astype(str) == "True"
    print(f"  labeller agreement: {int(agree.sum())}/{len(labels)}")
    rows.append({"section": "primary", "measure": "labeller_agreement", "arm": "all",
                 "value": float(agree.mean()), "n": len(labels), "count": int(agree.sum())})

    draws = draws_for(N_UNANSWERABLE)
    tests = []
    for a, b in PAIRS:
        est, lo, hi = paired_ci(h[a].to_numpy(float), h[b].to_numpy(float), draws)
        only_a, only_b, p = mcnemar(h[a].to_numpy(), h[b].to_numpy())
        tests.append((a, b, est, lo, hi, only_a, only_b, p))
    adjusted = holm([t[-1] for t in tests])
    naive_count = int(h["naive"].sum())
    outcome = "floor"
    for (a, b, est, lo, hi, only_a, only_b, p), p_holm in zip(tests, adjusted):
        deciding = (a, b) == DECIDING_PAIR
        label = classify(naive_count, lo, hi) if deciding else "reported_only"
        if deciding:
            outcome = label
        print(f"  {a} - {b}: {est:+.3f} [{lo:+.3f}, {hi:+.3f}]  discordant {only_a}/{only_b}"
              f"  McNemar p {p:.4f}, Holm {p_holm:.4f}  -> {label}")
        rows.append({"section": "primary", "measure": "hallucination_diff", "arm": f"{a}-{b}",
                     "value": est, "lo": lo, "hi": hi, "n": N_UNANSWERABLE,
                     "count": only_a - only_b, "p_mcnemar": p, "p_holm": p_holm,
                     "outcome": label})
    if outcome == "floor":
        print(f"  FLOOR: naive hallucinates on {naive_count} <= {FLOOR}. The effect is "
              f"unmeasurable on this set, not absent; the fallback headline applies.")
    return outcome


def refusal_table(answers: pd.DataFrame, checks: pd.DataFrame) -> pd.Series:
    refused = answers["abstained"].astype(str) == "True"
    hand = checks[checks["check"] == "refusal"]
    override = {(r["id"], r["strategy"]): r["final"] == "REFUSAL" for _, r in hand.iterrows()}
    return pd.Series([override.get((i, s), bool(f)) for i, s, f in
                      zip(answers["id"], answers["strategy"], refused)], index=answers.index)


def secondary(answers, checks, faith, rows) -> None:
    scorable = answers[answers["question_type"] != "unanswerable"].copy()
    scorable["refused"] = refusal_table(scorable, checks).astype(int)

    print("\nSECONDARY 1: refusal on scorable questions (regex, hand-corrected)")
    for ev in ("complete", "incomplete"):
        part = scorable[scorable["evidence"] == ev]
        for s in STRATEGIES:
            r = part[part["strategy"] == s]["refused"]
            print(f"  {ev:<10} {s:<19} {int(r.sum()):>2}/{len(r)}")
            rows.append({"section": "false_abstention", "measure": f"refusal_rate_{ev}",
                         "arm": s, "value": float(r.mean()), "n": len(r), "count": int(r.sum())})
    comp = wide(scorable[scorable["evidence"] == "complete"], "refused")
    est, lo, hi = paired_ci(comp["abstain"].to_numpy(float), comp["naive"].to_numpy(float),
                            draws_for(len(comp)))
    print(f"  complete, abstain - naive: {est:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    rows.append({"section": "false_abstention", "measure": "refusal_diff_complete",
                 "arm": "abstain-naive", "value": est, "lo": lo, "hi": hi, "n": len(comp)})

    print(f"\nSECONDARY 2: faithfulness (judge), refusals excluded; "
          f"judge test-retest noise {JUDGE_NOISE}")
    merged = scorable.merge(faith[["id", "strategy", "faithfulness", "n_statements"]],
                            on=["id", "strategy"], how="left")
    answered = merged[merged["refused"] == 0]
    missing = int(answered["n_statements"].isna().sum())
    if missing:
        print(f"  WARNING: {missing} answered rows have no judgment; re-run 26 --judge")
    scored = answered.dropna(subset=["faithfulness"])
    scored = scored[scored["faithfulness"].astype(str) != ""]
    scored = scored.assign(faithfulness=scored["faithfulness"].astype(float))
    no_claims = int((answered["n_statements"] == 0).sum())
    for ev in ("complete", "incomplete"):
        for s in STRATEGIES:
            f = scored[(scored["evidence"] == ev) & (scored["strategy"] == s)]["faithfulness"]
            print(f"  {ev:<10} {s:<19} mean {f.mean():.3f} over {len(f)}")
            rows.append({"section": "faithfulness", "measure": f"mean_{ev}", "arm": s,
                         "value": float(f.mean()), "n": len(f)})
    both = wide(scored, "faithfulness")[["naive", "abstain"]].dropna()
    est, lo, hi = paired_ci(both["naive"].to_numpy(), both["abstain"].to_numpy(),
                            draws_for(len(both)))
    within = abs(est) < JUDGE_NOISE
    print(f"  naive - abstain, {len(both)} questions both answered: {est:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}]{'  within judge noise' if within else ''}")
    print(f"  answers with no claims (not scored): {no_claims}")
    rows.append({"section": "faithfulness", "measure": "diff_both_answered",
                 "arm": "naive-abstain", "value": est, "lo": lo, "hi": hi, "n": len(both),
                 "outcome": "within_judge_noise" if within else "beyond_judge_noise"})

    print("\nSECONDARY 3: citation accuracy (answered scorable questions)")
    for s in STRATEGIES:
        a = answered[answered["strategy"] == s]
        n_cites = int(a["n_citations"].sum())
        precision = a["cited_on_gold"].sum() / n_cites if n_cites else float("nan")
        in_ctx = sum(len(span_indices(c)) for c in a["gold_spans_in_context"])
        cited = sum(len(span_indices(c)) for c in a["gold_spans_cited"])
        recall = cited / in_ctx if in_ctx else float("nan")
        oor = int(a["out_of_range_citations"].sum())
        print(f"  {s:<19} precision {precision:.3f} ({int(a['cited_on_gold'].sum())}/{n_cites})"
              f"  recall {recall:.3f} ({cited}/{in_ctx})  out of range {oor}")
        rows += [{"section": "citations", "measure": "precision", "arm": s, "value": precision,
                  "n": n_cites},
                 {"section": "citations", "measure": "recall", "arm": s, "value": recall,
                  "n": in_ctx},
                 {"section": "citations", "measure": "out_of_range", "arm": s, "value": oor}]


def calibration(answers, labels, checks, rows) -> None:
    conf = answers[answers["strategy"] == "abstain_confidence"].copy()
    conf["stated"] = [m.group(1).lower() if (m := _CONFIDENCE.search(a)) else "none"
                      for a in conf["answer"]]
    final = labels[labels["strategy"] == "abstain_confidence"].set_index("id")["final"]
    conf["truth"] = [final.get(i, ev) if qt == "unanswerable" else ev
                     for i, qt, ev in zip(conf["id"], conf["question_type"], conf["evidence"])]
    table = pd.crosstab(conf["truth"], conf["stated"])
    print("\nSECONDARY 4: abstain_confidence stated confidence\n" + table.to_string())
    for truth, row in table.iterrows():
        for stated, n in row.items():
            rows.append({"section": "calibration", "measure": f"stated_{stated}",
                         "arm": str(truth), "value": int(n)})

    amb = checks[checks["check"] == "ambiguity"]
    print("\nAMBIGUOUS QUESTIONS: answers acknowledging several answers")
    for s in STRATEGIES:
        a = amb[amb["strategy"] == s]
        n = int((a["final"] == "ACKNOWLEDGES").sum())
        print(f"  {s:<19} {n}/{len(a)}")
        rows.append({"section": "ambiguity", "measure": "acknowledges", "arm": s,
                     "value": n / len(a) if len(a) else float("nan"), "n": len(a), "count": n})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default=str(TABLES_DIR))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    tables = Path(args.tables)

    answers = pd.read_csv(tables / "generation_answers.csv",
                          dtype={"gold_spans_in_context": str, "gold_spans_cited": str})
    labels = pd.read_csv(tables / "generation_labels.csv", dtype=str)
    checks = pd.read_csv(tables / "generation_checks.csv", dtype=str)
    faith = pd.read_csv(tables / "generation_faithfulness.csv", dtype={"faithfulness": str})

    served = sorted(set(answers["served_model"].dropna()))
    if len(answers) != N_ANSWERS or len(served) != 1:
        print(f"REPORT FIRST: {len(answers)} answers (expected {N_ANSWERS}), "
              f"served versions {served}")

    rows: list[dict] = []
    outcome = primary(labels, rows)
    secondary(answers, checks, faith, rows)
    calibration(answers, labels, checks, rows)

    out = tables / "generation_result.csv"
    out.unlink(missing_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    record([out])
    print(f"\nOUTCOME: {outcome}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
