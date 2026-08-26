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


def format_context(hits: list, max_chars_per_hit: int = 2000) -> str:
    """Render retrieved chunks as numbered passages the model can cite."""
    blocks = []
    for i, h in enumerate(hits, 1):
        body = " ".join(h.text.split())
        if len(body) > max_chars_per_hit:
            body = body[:max_chars_per_hit] + " ..."
        blocks.append(f"[{i}] ({h.arxiv_id} - {h.title[:70]})\n{body}")
    return "\n\n".join(blocks) if blocks else "(no passages retrieved)"


def build_prompt(question: str, hits: list, strategy: str = "abstain") -> tuple[str, str]:
    if strategy not in PROMPT_STRATEGIES:
        raise ValueError(f"Unknown prompt strategy: {strategy}. "
                         f"Options: {sorted(PROMPT_STRATEGIES)}")
    system = PROMPT_STRATEGIES[strategy]
    user = _USER_TEMPLATE.format(context=format_context(hits), question=question)
    return system, user