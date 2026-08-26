"""Gold evaluation set: schema, IO and validation.

Evidence is anchored to (arxiv_id, char_start, char_end) against the frozen
extraction output, never to chunk IDs. Chunk IDs differ across the six
chunking strategies being ablated, so chunk-anchored ground truth would be
valid for exactly one arm of the experiment and meaningless for the rest.

Every span also carries the quote it points at, which makes it
self-verifying: if the text at those offsets no longer matches, the
extraction changed and the whole gold set is suspect.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

QUESTION_TYPES = ("factual", "multi_hop", "comparative", "unanswerable", "ambiguous")

# Composition target from the build plan.
TARGET_COUNTS = {
    "factual": 20,
    "multi_hop": 18,
    "comparative": 14,
    "unanswerable": 16,
    "ambiguous": 6,
}


@dataclass
class Evidence:
    """One passage supporting an answer, anchored by span and quote."""

    arxiv_id: str
    char_start: int
    char_end: int
    quote: str

    def resolves_against(self, text: str) -> bool:
        actual = text[self.char_start : self.char_end]
        return " ".join(actual.split()) == " ".join(self.quote.split())


@dataclass
class GoldQuestion:
    id: str
    question: str
    question_type: str
    reference_answer: str
    evidence: list[Evidence] = field(default_factory=list)
    difficulty: str = "medium"
    note: str = ""
    verified: bool = False

    @property
    def papers(self) -> set[str]:
        return {e.arxiv_id for e in self.evidence}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [asdict(e) for e in self.evidence]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GoldQuestion":
        d = dict(d)
        d["evidence"] = [Evidence(**e) for e in d.get("evidence", [])]
        return cls(**d)


def load_gold(path: Path) -> list[GoldQuestion]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(GoldQuestion.from_dict(json.loads(line)))
    return out


def save_gold(questions: list[GoldQuestion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_question(q: GoldQuestion, texts: dict[str, str]) -> list[str]:
    """Return a list of problems. Empty means the question is sound."""
    problems: list[str] = []

    if q.question_type not in QUESTION_TYPES:
        problems.append(f"unknown type '{q.question_type}'")
    if not q.question.strip():
        problems.append("empty question")
    if not q.reference_answer.strip():
        problems.append("empty reference answer")

    if q.question_type == "unanswerable":
        if q.evidence:
            problems.append("unanswerable question has evidence attached")
    elif not q.evidence:
        problems.append(f"{q.question_type} question has no evidence")

    if q.question_type == "multi_hop":
        if len(q.evidence) < 2:
            problems.append("multi_hop needs at least two evidence spans")
        elif len(q.papers) < 2:
            problems.append("multi_hop evidence all comes from one paper")

    if q.question_type == "comparative" and len(q.papers) < 2:
        problems.append("comparative evidence should span at least two papers")

    for i, ev in enumerate(q.evidence):
        label = f"evidence[{i}] {ev.arxiv_id}"
        if ev.arxiv_id not in texts:
            problems.append(f"{label}: paper not in corpus")
            continue
        text = texts[ev.arxiv_id]
        if ev.char_start < 0 or ev.char_end > len(text):
            problems.append(f"{label}: span {ev.char_start}-{ev.char_end} "
                            f"outside document (length {len(text)})")
        elif ev.char_start >= ev.char_end:
            problems.append(f"{label}: empty or inverted span")
        elif not ev.resolves_against(text):
            actual = " ".join(text[ev.char_start:ev.char_end].split())[:60]
            problems.append(f"{label}: quote does not match text at span. "
                            f"Found: {actual!r}")

    if not q.verified:
        problems.append("not marked verified")

    return problems


def validate_set(questions: list[GoldQuestion], texts: dict[str, str]) -> dict:
    """Validate the whole set, including cross-question checks."""
    per_question = {q.id: validate_question(q, texts) for q in questions}

    ids = [q.id for q in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    for dup in duplicates:
        per_question.setdefault(dup, []).append("duplicate question id")

    seen_text: dict[str, str] = {}
    for q in questions:
        key = " ".join(q.question.lower().split())
        if key in seen_text:
            per_question[q.id].append(f"duplicate question text (same as {seen_text[key]})")
        seen_text[key] = q.id

    counts = {t: sum(1 for q in questions if q.question_type == t) for t in QUESTION_TYPES}
    return {
        "per_question": per_question,
        "counts": counts,
        "n_total": len(questions),
        "n_ok": sum(1 for p in per_question.values() if not p),
        "papers_covered": len({e.arxiv_id for q in questions for e in q.evidence}),
    }