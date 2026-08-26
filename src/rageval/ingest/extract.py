"""Extract and clean text from academic PDFs, scoring extraction quality.

Academic PDFs are the hard case: two-column layouts that naive extractors
interleave, subset fonts that decode to (cid:NN), reference sections that
add thousands of tokens of noise, and equations that fragment into
punctuation soup. This module handles those and, critically, *measures*
how well it did so failures are visible rather than silent.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber
import pymupdf

logger = logging.getLogger(__name__)

pymupdf.TOOLS.mupdf_display_errors(False)

# Headings that mark the end of body content worth retrieving over.
# Strict first: a line containing only "References". Falls back to a looser
# match for PDFs where the first citation lands on the heading's line.
_END_STRICT = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]*)?(?:references|bibliography)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_END_LOOSE = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]*)?(?:references|bibliography)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Where body content resumes after a bibliography block.
_RESUME = re.compile(r"^[ \t]*(?:appendix|supplement(?:ary|al)?)\b", re.IGNORECASE | re.MULTILINE) 
_MIN_BODY_BEFORE_REFS = 3000

# Numbered headings (1 Introduction / 2.3 Method) or known unnumbered ones.
_KNOWN_HEADINGS = (
    r"abstract|introduction|background|related work|literature review|"
    r"methodology|methods?|approach|model|architecture|data|dataset|"
    r"experiments?|experimental setup|results?|evaluation|discussion|"
    r"ablation|limitations?|conclusions?|future work|acknowledgements?"
)
_HEADING = re.compile(
    rf"^[ \t]*(?:(?:\d+(?:\.\d+)*)\.?[ \t]+[A-Z][^\n]{{2,70}}|(?:{_KNOWN_HEADINGS}))[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

_CID = re.compile(r"\(cid:\d+\)")
_WORDLIKE = re.compile(r"^[A-Za-z][A-Za-z'-]{1,}$")


@dataclass
class Section:
    heading: str
    text: str
    char_count: int = 0


@dataclass
class ExtractedDoc:
    arxiv_id: str
    engine: str
    n_pages: int
    raw_chars: int
    clean_chars: int
    references_found: bool
    n_sections: int
    status: str
    quality: dict = field(default_factory=dict)
    error: str | None = None
    sections: list[Section] = field(default_factory=list)
    text: str = ""

    def to_manifest_row(self) -> dict:
        """Audit row - everything except the bulky text."""
        row = asdict(self)
        row.pop("text")
        row.pop("sections")
        return row


# --------------------------------------------------------------------------
# Column-aware block ordering
# --------------------------------------------------------------------------

def _order_blocks(blocks: list, page_width: float) -> list:
    """Sort text blocks into human reading order, handling two-column layouts.

    A naive top-to-bottom sort interleaves the two columns of a typical
    academic paper, producing text that alternates between unrelated
    sentences. That destroys chunk coherence and quietly wrecks retrieval.

    Blocks are (x0, y0, x1, y1, text, block_no, block_type).
    """
    text_blocks = [b for b in blocks if len(b) >= 5 and b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return []

    mid = page_width / 2.0
    left, right, spanning = [], [], []
    for b in text_blocks:
        if b[2] < mid * 1.05:      # ends before centre
            left.append(b)
        elif b[0] > mid * 0.95:    # starts after centre
            right.append(b)
        else:                      # crosses the gutter
            spanning.append(b)

    # Treat as two-column only when both sides carry real weight. A title
    # page with one stray sidebar should not trigger column splitting.
    total = len(text_blocks)
    if len(left) >= 0.25 * total and len(right) >= 0.25 * total:
        left.sort(key=lambda b: b[1])
        right.sort(key=lambda b: b[1])
        spanning.sort(key=lambda b: b[1])
        # Spanning blocks (titles, full-width figures) lead the page.
        return spanning + left + right

    return sorted(text_blocks, key=lambda b: (b[1], b[0]))


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------

def extract_pymupdf(path: Path) -> tuple[str, int]:
    parts: list[str] = []
    with pymupdf.open(path) as doc:
        n_pages = doc.page_count
        for page in doc:
            blocks = page.get_text("blocks")
            ordered = _order_blocks(blocks, page.rect.width)
            parts.append("\n".join(b[4].strip() for b in ordered))
    return "\n\n".join(parts), n_pages


def extract_pdfplumber(path: Path) -> tuple[str, int]:
    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n\n".join(parts), n_pages


ENGINES = {"pymupdf": extract_pymupdf, "pdfplumber": extract_pdfplumber}


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Repair extraction damage while PRESERVING line structure.

    Line breaks are load-bearing: headings and the references marker are
    only identifiable as lines. Collapsing them before parsing structure
    silently destroys both.
    """
    for bad, good in (("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"), ("ﬃ", "ffi"), ("ﬄ", "ffl")):
        text = text.replace(bad, good)

    # Hyphenation across line breaks: "convo-\nlutional" -> "convolutional".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collapse_paragraphs(text: str) -> str:
    """Turn soft wraps into spaces. Run only AFTER structural parsing."""
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def strip_references(text: str) -> tuple[str, bool]:
    """Excise the bibliography block, preserving any appendix that follows.

    Reference lists are noise for retrieval - author names and venues that
    match queries lexically while containing nothing answerable. Appendices
    are the opposite: methods detail, extra experiments, real answers.

    Cutting from References to end-of-document destroys both. So when a
    resumption marker follows the bibliography, only the block between them
    is removed, and the position guard relaxes - in appendix-heavy papers
    the bibliography can sit as early as 20% of the way through.
    """
    n = len(text)
    for pattern in (_END_STRICT, _END_LOOSE):
        for match in reversed(list(pattern.finditer(text))):
            start = match.start()
            resume = _RESUME.search(text, match.end())

            if resume:
                if start < _MIN_BODY_BEFORE_REFS:
                    continue
                kept = text[:start].rstrip() + "\n\n" + text[resume.start():]
                return kept.strip(), True

            # No resumption marker: the bibliography runs to the end, so the
            # stricter guard still applies against mid-prose false positives.
            if start >= 0.40 * n:
                return text[:start].strip(), True

    return text, False


def split_sections(text: str) -> list[Section]:
    """Split on detected headings, preserving them as retrievable metadata."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [Section(heading="(untitled)", text=text, char_count=len(text))]

    sections: list[Section] = []
    if matches[0].start() > 0:
        head = text[: matches[0].start()].strip()
        if head:
            sections.append(Section("(front matter)", head, len(head)))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = " ".join(m.group(0).split())
        body = text[m.end() : end].strip()
        if body:
            sections.append(Section(heading, body, len(body)))
    return sections


# --------------------------------------------------------------------------
# Quality scoring
# --------------------------------------------------------------------------

def score_quality(text: str, n_pages: int) -> dict:
    """Cheap heuristics that correlate with unusable extractions."""
    n = len(text)
    if n == 0:
        return {
            "chars": 0, "chars_per_page": 0.0, "alpha_ratio": 0.0,
            "wordlike_ratio": 0.0, "cid_hits": 0, "mean_word_len": 0.0,
        }

    tokens = text.split()
    wordlike = sum(1 for t in tokens if _WORDLIKE.match(t.strip(".,;:()[]")))
    return {
        "chars": n,
        "chars_per_page": round(n / max(n_pages, 1), 1),
        "alpha_ratio": round(sum(c.isalpha() or c.isspace() for c in text) / n, 3),
        "wordlike_ratio": round(wordlike / max(len(tokens), 1), 3),
        "cid_hits": len(_CID.findall(text)),
        "mean_word_len": round(sum(len(t) for t in tokens) / max(len(tokens), 1), 2),
    }


def classify(q: dict, n_sections: int) -> str:
    """clean | degraded | failed - thresholds tuned on academic PDFs."""
    if q["chars"] < 3000:
        return "failed"
    if q["cid_hits"] > 50 or q["alpha_ratio"] < 0.55:
        return "failed"
    if q["chars_per_page"] < 1200 or q["wordlike_ratio"] < 0.55 or n_sections < 2:
        return "degraded"
    return "clean"


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def extract_document(path: Path, arxiv_id: str, engine: str = "pymupdf") -> ExtractedDoc:
    """Run one PDF end to end. Never raises - failures become status='failed'."""
    try:
        raw, n_pages = ENGINES[engine](path)
    except Exception as exc:  # noqa: BLE001
        return ExtractedDoc(
            arxiv_id=arxiv_id, engine=engine, n_pages=0, raw_chars=0, clean_chars=0,
            references_found=False, n_sections=0, status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )

    # Order is critical: structure is parsed while line breaks still exist,
    # and soft wraps are only collapsed afterwards, per section.
    normalized = normalize_text(raw)
    body, refs_found = strip_references(normalized)
    sections = split_sections(body)
    for s in sections:
        s.text = collapse_paragraphs(s.text)
        s.char_count = len(s.text)

    flat = collapse_paragraphs(body)
    quality = score_quality(flat, n_pages)
    status = classify(quality, len(sections))

    return ExtractedDoc(
        arxiv_id=arxiv_id, engine=engine, n_pages=n_pages,
        raw_chars=len(raw), clean_chars=len(flat),
        references_found=refs_found, n_sections=len(sections),
        status=status, quality=quality, sections=sections, text=flat,
    )


def compare_engines(path: Path, arxiv_id: str) -> dict:
    """Run both engines on one document and report the difference."""
    out: dict = {"arxiv_id": arxiv_id}
    for name in ENGINES:
        t0 = time.perf_counter()
        doc = extract_document(path, arxiv_id, engine=name)
        out[f"{name}_chars"] = doc.clean_chars
        out[f"{name}_status"] = doc.status
        out[f"{name}_sections"] = doc.n_sections
        out[f"{name}_seconds"] = round(time.perf_counter() - t0, 2)
    a, b = out["pymupdf_chars"], out["pdfplumber_chars"]
    out["char_delta_pct"] = round(100 * (a - b) / max(b, 1), 1)
    return out