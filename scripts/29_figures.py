"""Draw the write-up's figures from committed results tables only.

Nothing here computes a statistic. Every interval, p-value and outcome is
read from a table that a pre-registered script already wrote; span recall
for the chunk-size figure is summed from committed retrieval tables
through rageval.evaluation.spans, the one hit_spans parser. A figure that
recomputed its own numbers could drift from the table it illustrates,
which is the two-harnesses failure in another form.

Writes PNGs to results/figures/:

  fig1_hallucination.png   Phase 7: 16 unanswerable questions x 3 prompts
  fig2_effects.png         every pre-registered comparison, 20k budget, 95% CI
  fig3_paraphrase.png      BM25 vs dense, original and rewritten wording
  fig4_chunk_size.png      chunk size at fixed k vs at equal character budget
  fig5_k_curve.png         BM25 span recall and complete coverage by k

  python scripts\\29_figures.py
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rageval.config import PROJECT_ROOT, TABLES_DIR  # noqa: E402
from rageval.evaluation.spans import count_spans  # noqa: E402
from rageval.provenance import record  # noqa: E402

FIG_DIR = PROJECT_ROOT / "results" / "figures"

# Reference palette (light surface); two categorical slots, validated together.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"     # BM25, the selected retriever; "decisive"
ORANGE = "#eb6834"   # dense
NEUTRAL = "#b5b3ab"  # not separable / not hallucinated

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK_2, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
})


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / name, dtype=str, keep_default_na=False)


def num(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c])
    return df


def save(fig, name: str, written: list) -> None:
    path = FIG_DIR / name
    path.unlink(missing_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    print(f"  -> {path}")


# --- figure 1 ------------------------------------------------------------
def fig_hallucination(written: list) -> None:
    labels = table("generation_labels.csv")
    res = table("generation_result.csv")
    diff = res[(res.measure == "hallucination_diff") & (res.arm == "naive-abstain")].iloc[0]
    arms = [("naive", "naive"), ("abstain", "abstain"),
            ("abstain_confidence", "abstain + confidence")]
    ids = sorted(labels.id.unique())
    final = {(r.id, r.strategy): r.final for r in labels.itertuples()}
    disagreed = {(r.id, r.strategy) for r in labels.itertuples() if r.agree != "True"}

    fig, ax = plt.subplots(figsize=(8.2, 2.7))
    for row, (arm, name) in enumerate(arms):
        y = len(arms) - 1 - row
        n = 0
        for x, qid in enumerate(ids):
            hall = final[(qid, arm)] == "HALLUCINATE"
            n += hall
            ax.add_patch(plt.Rectangle((x + 0.06, y + 0.12), 0.88, 0.76,
                                       facecolor=ORANGE if hall else SURFACE,
                                       edgecolor=ORANGE if hall else NEUTRAL, linewidth=1.2))
            if (qid, arm) in disagreed:
                ax.plot(x + 0.5, y + 0.5, marker="o", markersize=4.5, color=SURFACE,
                        markeredgecolor=INK, markeredgewidth=0.9)
        ax.text(-0.3, y + 0.5, name, ha="right", va="center", color=INK_2)
        ax.text(len(ids) + 0.3, y + 0.5, f"{n} / {len(ids)}", ha="left", va="center",
                color=INK, fontweight="bold")
    ax.set_xlim(-0.1, len(ids) + 1.6)
    ax.set_ylim(-0.05, len(arms))
    ax.set_xticks([i + 0.5 for i in range(len(ids))], ids, rotation=90, fontsize=7.5)
    ax.tick_params(axis="x", length=0, colors=MUTED)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Unanswerable questions answered anyway (filled = hallucinated)", pad=10)
    fig.text(0.01, -0.13,
             f"naive − abstain = +{float(diff.value):.3f}, 95% CI [+{float(diff.lo):.3f}, "
             f"+{float(diff.hi):.3f}], McNemar p = {float(diff.p_mcnemar):.4f}. "
             "Circle: labellers disagreed; the owner's final label is shown.\n"
             "gpt-4o-mini, BM25 top 20 passages, each question's evidence deliberately absent "
             "from the corpus but adjacent to it.",
             color=INK_2, fontsize=8)
    save(fig, "fig1_hallucination.png", written)


# --- figure 2 ------------------------------------------------------------
def fig_effects(written: list) -> None:
    rows = []

    def add(group, label, df, a, b, decided=False, sign=1):
        r = num(df[(df.a == a) & (df.b == b)].copy(), "delta", "ci_lo", "ci_hi").iloc[0]
        lo, hi = sorted((sign * r.ci_lo, sign * r.ci_hi))
        rows.append((group, label, sign * r.delta, lo, hi, decided))

    s1 = table("paired_baseline_chunk256_chunk1024_B20000.csv")
    add("Chunk size, dense", "fixed_256 vs fixed_512", s1, "baseline", "chunk256")
    add("Chunk size, dense", "fixed_1024 vs fixed_512", s1, "baseline", "chunk1024")
    s2 = table("paired_baseline_bm25_hybrid_B20000.csv")
    add("Retrieval function", "BM25 vs dense", s2, "baseline", "bm25", decided=True)
    add("Retrieval function", "hybrid (RRF) vs dense", s2, "baseline", "hybrid")
    s3 = table("paired_baseline_bm25_rerank_B20000.csv")
    add("Reranking", "cross-encoder vs BM25", s3, "bm25", "rerank")
    f = table("paired_bm25_filter030_filter040_filter050_B20000.csv")
    for t in ("030", "040", "050"):
        add("Low-prose chunk filter", f"filter at 0.{t[1:]} vs none", f, "bm25", f"filter{t}")
    c = table("paired_bm25_chunk_section_chunk_recursive_chunk_semantic_B20000.csv")
    for name in ("section", "recursive", "semantic"):
        add("Chunker, size-matched", f"{name} vs fixed_512", c, "bm25", f"chunk_{name}")
    r = table("paired_bm25_bm25_chunk256_B20000.csv")
    add("Chunk size, BM25", "fixed_256 vs fixed_512", r, "bm25", "bm25_chunk256")
    p = num(table("paraphrase_result.csv"), "delta_original", "delta_original_lo",
            "delta_original_hi", "delta_paraphrase", "delta_paraphrase_lo",
            "delta_paraphrase_hi")
    p = p[p.metric == "budget 20,000"].iloc[0]
    rows.append(("BM25 lead, 20 highest-overlap questions", "original wording",
                 p.delta_original, p.delta_original_lo, p.delta_original_hi, False))
    rows.append(("BM25 lead, 20 highest-overlap questions", "rewritten wording",
                 p.delta_paraphrase, p.delta_paraphrase_lo, p.delta_paraphrase_hi, False))

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    y, ticks, labels, group_prev = 0, [], [], None
    for group, label, d, lo, hi, decided in rows:
        if group != group_prev:
            y -= 0.6 if group_prev else 0
            ax.text(-0.375, y, group, ha="left", va="center", color=INK, fontweight="bold",
                    fontsize=9.5)
            y -= 1
            group_prev = group
        excludes = lo > 0 or hi < 0
        color = BLUE if decided else (INK_2 if excludes else NEUTRAL)
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(d, y, "o", color=color, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5)
        ax.text(0.405, y, f"{d:+.3f}  [{lo:+.3f}, {hi:+.3f}]", ha="left", va="center",
                color=INK_2, fontsize=8.5, family="monospace")
        ticks.append(y)
        labels.append(label)
        y -= 1
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.375, 0.4)
    ax.set_ylim(y + 0.4, 0.6)
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Difference in span recall at a 20,000-character context budget (95% CI)")
    ax.set_title("Only one design change cleared the pre-registered switching rule", pad=12)
    fig.text(0.01, 0.005,
             "Blue: replaced the baseline (CI excludes zero and half-split win rate 93.2%). "
             "Dark grey: CI excludes zero, not adopted:\nfixed_1024 is worse; hybrid is not "
             "separable from BM25 and lost on parsimony; the last group is a subgroup. "
             "Light grey: not separable.\nCluster bootstrap over questions, 10,000 resamples; "
             "58 questions, "
             "101 evidence spans\n(the last group: 20 questions, 40 spans). Reranking's lower "
             "bound is exactly 0.000, so it does not exclude zero.",
             color=INK_2, fontsize=8, va="top")
    save(fig, "fig2_effects.png", written)


# --- figure 3 ------------------------------------------------------------
def fig_paraphrase(written: list) -> None:
    p = num(table("paraphrase_result.csv"), "orig_dense", "orig_bm25", "para_dense",
            "para_bm25", "effect", "effect_lo", "effect_hi")
    p = p[p.metric == "budget 20,000"].iloc[0]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for name, a, b, color in (("BM25", p.orig_bm25, p.para_bm25, BLUE),
                              ("dense", p.orig_dense, p.para_dense, ORANGE)):
        ax.plot([0, 1], [a, b], color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(-0.06, a, f"{name}  {a:.3f}", ha="right", va="center", color=INK_2)
        ax.text(1.06, b, f"{b:.3f}  {name}", ha="left", va="center", color=INK_2)
    ax.set_xticks([0, 1], ["original wording", "rewritten wording"])
    ax.tick_params(axis="x", length=0, colors=INK_2)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 0.7)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.grid(axis="y")
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Span recall, 20,000-character budget")
    ax.set_title("BM25's lead disappears when the question's words do", pad=12)
    fig.text(0.01, -0.06,
             f"The 20 gold questions sharing most vocabulary with their evidence, rewritten by\n"
             f"hand. The lead shrinks by +{p.effect:.3f}, 95% CI [+{p.effect_lo:.3f}, "
             f"+{p.effect_hi:.3f}]; dense search at ef_search 500 gives the same result.",
             color=INK_2, fontsize=8, va="top")
    save(fig, "fig3_paraphrase.png", written)


# --- figure 4 ------------------------------------------------------------
def span_recall(tag: str, col: str) -> float:
    run = table(f"retrieval_{tag}.csv")
    run = run[run.scorable == "True"]
    return sum(count_spans(c) for c in run[col]) / pd.to_numeric(run.n_evidence).sum()


def fig_chunk_size(written: list) -> None:
    arms = (("fixed_256", "chunk256"), ("fixed_512", "baseline"), ("fixed_1024", "chunk1024"))
    views = (("Same number of chunks (k = 10)", "hit_spans@10"),
             ("Same amount of text (20,000 characters)", "hit_spans@B20000"))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    for ax, (title, col) in zip(axes, views):
        vals = [span_recall(tag, col) for _, tag in arms]
        best = max(range(3), key=lambda i: vals[i])
        bars = ax.bar(range(3), vals, width=0.62,
                      color=[BLUE if i == best else NEUTRAL for i in range(3)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center",
                    va="bottom", color=INK, fontsize=9)
        ax.set_xticks(range(3), [name for name, _ in arms])
        ax.tick_params(axis="x", length=0, colors=INK_2)
        ax.set_title(title, fontsize=10, pad=8)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Span recall, dense retrieval")
    axes[0].set_ylim(0, 0.55)
    fig.suptitle("Ranking chunk sizes at fixed k measures text volume, not chunking",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, -0.04,
             "1,024-token chunks hand the scorer about four times the characters of 256-token "
             "chunks at the same k. At equal text the order\ninverts. "
             "At the budget, the primary metric, 256 and 512 are not separable "
             "(Holm p = 0.50); 1,024 is worse (Holm p = 0.009).",
             color=INK_2, fontsize=8, va="top")
    fig.tight_layout()
    save(fig, "fig4_chunk_size.png", written)


# --- figure 5 ------------------------------------------------------------
def fig_k_curve(written: list) -> None:
    cur = num(table("k_sweep_curve.csv"), "k", "span_recall", "complete_share")
    steps = num(table("k_sweep.csv"), "k_from", "k_to", "gain", "ci_lo", "ci_hi")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    series = (("all", "span_recall", BLUE, "all 58 questions: span recall"),
              ("all", "complete_share", BLUE, "all 58: every gold span present"),
              ("rewritten_20_reworded", "span_recall", ORANGE, "20 rewritten questions: span recall"))
    for subset, col, color, name in series:
        s = cur[(cur.arm == "bm25") & (cur.subset == subset)].sort_values("k")
        dashed = col == "complete_share"
        ax.plot(s.k, s[col], color=color, linewidth=2, linestyle="--" if dashed else "-",
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)
        last = s.iloc[-1]
        ax.text(20.8, last[col], f"{last[col]:.3f}  {name}", va="center", color=INK_2,
                fontsize=8.5)
    for st in steps.itertuples():
        ax.text((st.k_from + st.k_to) / 2, 0.015, f"+{st.gain:.3f}", ha="center",
                color=MUTED, fontsize=7.5)
    ax.set_xticks([1, 3, 5, 10, 20])
    ax.set_xlim(0, 20.5)
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y")
    ax.set_xlabel("Passages retrieved (k)")
    ax.set_ylabel("Share")
    ax.set_title("Every step to k = 20 still adds evidence", pad=12)
    fig.text(0.01, -0.07,
             "BM25, fixed 512-token chunks. Grey numbers: gain in all-58 span recall per step, "
             "each with a 95% CI excluding zero.\nA larger k does not fix the rewritten "
             "questions: their shortfall is in level, not in rate.",
             color=INK_2, fontsize=8, va="top")
    save(fig, "fig5_k_curve.png", written)


def main() -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list = []
    fig_hallucination(written)
    fig_effects(written)
    fig_paraphrase(written)
    fig_chunk_size(written)
    fig_k_curve(written)
    record(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
