"""Prompt strategies for the abstention experiment.

The headline finding of this project is how much explicit abstention
instruction changes hallucination rate on unanswerable questions. That
requires the prompts to differ in exactly one respect - the instruction -
with context format, citation requirement and answer style held constant.
"""

from __future__ import annotations

ABSTENTION_SENTINEL = "INSUFFICIENT CONTEXT"

_BASE_RULES = """You answer questions about deep learning for weather and climate
forecasting, using only the numbered context passages provided.

Rules:
- Every factual claim must cite the passage it came from, as [1], [2], etc.
- Cite only passages you actually used.
- Quote specific numbers, model names and metrics where the context gives them.
- Be concise. Three sentences is usually enough."""

_NAIVE = _BASE_RULES

_ABSTAIN = _BASE_RULES + f"""
- If the context does not contain enough information to answer, reply with
  exactly "{ABSTENTION_SENTINEL}" and nothing else. Do not guess, do not
  answer from your own knowledge, and do not offer a partial answer."""

_ABSTAIN_CONFIDENCE = _ABSTAIN + """
- When you do answer, end with a final line of the form
  "Confidence: high" / "Confidence: medium" / "Confidence: low",
  reflecting how directly the context supports your answer."""

PROMPT_STRATEGIES: dict[str, str] = {
    "naive": _NAIVE,
    "abstain": _ABSTAIN,
    "abstain_confidence": _ABSTAIN_CONFIDENCE,
}

_USER_TEMPLATE = """Context passages:

{context}

Question: {question}

Answer:"""


def format_context(hits: list) -> str:
    """Render retrieved chunks as numbered passages the model can cite.

    Every passage is sent whole. A 2,000-character default cap used to sit
    here, invisible to every config, and would have hidden part of 17 of
    the 67 gold spans retrieved at k = 20. If a cap is ever needed it is a
    declared config value that reports when it fires, not a default.
    """
    blocks = []
    for i, h in enumerate(hits, 1):
        # Whitespace collapse: measured at 0.9996 of raw characters over the
        # 20k-budget context, in every chunk set (build log, Phase 3).
        body = " ".join(h.text.split())
        # Cosmetic label truncation: the title is a display label only and
        # touches no measured quantity.
        blocks.append(f"[{i}] ({h.arxiv_id} - {h.title[:70]})\n{body}")
    return "\n\n".join(blocks) if blocks else "(no passages retrieved)"


def build_prompt(question: str, hits: list, strategy: str = "abstain") -> tuple[str, str]:
    if strategy not in PROMPT_STRATEGIES:
        raise ValueError(f"Unknown prompt strategy: {strategy}. "
                         f"Options: {sorted(PROMPT_STRATEGIES)}")
    system = PROMPT_STRATEGIES[strategy]
    user = _USER_TEMPLATE.format(context=format_context(hits), question=question)
    return system, user