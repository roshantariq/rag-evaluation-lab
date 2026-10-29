"""Rerank a first-stage candidate list with a cross-encoder.

A cross-encoder reads the question and one passage together and scores the
pair, instead of comparing two separately computed vectors (dense) or
counting shared terms (BM25). It can only reorder what the first stage
hands it: anything outside the top `depth` candidates is unreachable, so
the first stage's recall within that depth is the ceiling of this arm.

Truncation is declared, not inherited. The model's default input length
would cut the passage to fit question + passage into the window, and
Phase 5 showed that a silently truncated encoder can hide half the corpus.
`max_length` is therefore always passed explicitly and printed by the
harness. docs/reranking-protocol.md records the measured share of text
each choice would leave unseen.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence


class Reranker:
    """Reorder candidates by cross-encoder score.

    `score_fn` exists so the reordering logic can be tested without
    downloading a model: it takes a list of (question, passage) pairs and
    returns one float per pair. In real use it is built from a
    sentence-transformers CrossEncoder.
    """

    def __init__(self, model_name: str, max_length: int, batch_size: int = 16,
                 score_fn: Callable[[list[tuple[str, str]]], Sequence[float]] | None = None):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        if score_fn is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, max_length=max_length)
            # Read back what the model will actually use: a silent cap below
            # the requested length would reintroduce the truncation the
            # protocol measured and ruled out.
            self.max_length = getattr(model, "max_seq_length", None) or model.max_length

            def score_fn(pairs):
                return model.predict(pairs, batch_size=batch_size,
                                     show_progress_bar=False)
        self._score = score_fn

    def rerank(self, question: str, candidates: Sequence, top_k: int | None = None) -> list:
        """Candidates reordered by score, best first, with score and rank replaced.

        Ties keep first-stage order (Python's sort is stable), so a reranker
        that scores everything equally returns the first stage unchanged
        rather than an arbitrary permutation.
        """
        if not candidates:
            return []
        scores = [float(s) for s in self._score([(question, c.text) for c in candidates])]
        if len(scores) != len(candidates):
            raise ValueError(f"score_fn returned {len(scores)} scores for "
                             f"{len(candidates)} candidates")
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        if top_k is not None:
            order = order[:top_k]
        return [replace(candidates[i], score=scores[i], rank=r)
                for r, i in enumerate(order, 1)]
