"""Build the retriever a config names, for generation.

Until Phase 7, generation was dense-only: RAGPipeline queried the vector
store directly and ignored `retrieval.mode`, while the harness had moved to
BM25 in sweep 2. Measuring abstention on the arm that lost sweep 2 would
describe a pipeline nobody selected. This module closes that gap: the
pipeline asks for "the retriever the config names" and gets exactly the
object the harness scores. For BM25 that is BM25Retriever over the
collection's stored records, the same corpus and code path as
`10_eval_retrieval.py --retriever bm25`.

Every retriever here has one method, `query(text, k) -> list[hit]`, where
hits carry chunk_id, arxiv_id, text, char span, title, score and rank.
"""

from __future__ import annotations

from typing import Callable

from rageval.retrieve.bm25 import BM25Retriever

GENERATION_MODES = ("dense", "bm25")


class DenseRetriever:
    """The vector store and its query encoder, behind the common interface."""

    def __init__(self, store, encoder):
        self.store = store
        self.encoder = encoder

    def query(self, text: str, k: int = 10) -> list:
        return self.store.query(self.encoder.encode_query(text), k=k)


def build_retriever(cfg: dict, store, make_encoder: Callable[[], object] | None = None):
    """The retriever `cfg["retrieval"]["mode"]` names, over `store`'s collection.

    `make_encoder` builds the dense query encoder only when dense is asked
    for, so BM25 generation never loads an embedding model. Hybrid fusion
    is a harness arm, not a selected pipeline, and is refused here rather
    than half-supported.
    """
    mode = cfg.get("retrieval", {}).get("mode", "dense")
    if mode == "bm25":
        return BM25Retriever(store.all_records())
    if mode == "dense":
        if make_encoder is None:
            raise ValueError("dense retrieval needs an encoder")
        return DenseRetriever(store, make_encoder())
    raise ValueError(f"retrieval.mode {mode!r} is not supported for generation; "
                     f"expected one of {GENERATION_MODES}")
