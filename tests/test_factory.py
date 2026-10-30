"""Tests for the generation retriever factory, with stub stores only."""

from __future__ import annotations

import pytest

from rageval.retrieve.bm25 import BM25Retriever
from rageval.retrieve.factory import DenseRetriever, build_retriever
from rageval.store.chroma_store import Retrieved


def rec(cid: str, text: str) -> Retrieved:
    return Retrieved(chunk_id=cid, arxiv_id="2101.00001", text=text, char_start=0,
                     char_end=len(text), section="", title="", score=0.0, rank=0)


class StubStore:
    RECORDS = [rec("a", "convlstm radar nowcasting"), rec("b", "graphcast weather model"),
               rec("c", "precipitation skill score")]

    def all_records(self):
        return list(self.RECORDS)

    def query(self, emb, k=5):
        return self.RECORDS[:k]


class StubEncoder:
    def __init__(self):
        self.seen = []

    def encode_query(self, text):
        self.seen.append(text)
        return [0.0]


def test_bm25_mode_builds_bm25_over_the_stored_records():
    r = build_retriever({"retrieval": {"mode": "bm25"}}, StubStore())
    assert isinstance(r, BM25Retriever) and len(r) == 3
    assert r.query("graphcast", k=2)[0].chunk_id == "b"


def test_bm25_mode_never_builds_an_encoder():
    def boom():
        raise AssertionError("BM25 generation must not load an embedding model")
    build_retriever({"retrieval": {"mode": "bm25"}}, StubStore(), make_encoder=boom)


def test_dense_mode_encodes_the_question():
    enc = StubEncoder()
    r = build_retriever({"retrieval": {"mode": "dense"}}, StubStore(), make_encoder=lambda: enc)
    assert isinstance(r, DenseRetriever)
    assert [h.chunk_id for h in r.query("what is radar", k=2)] == ["a", "b"]
    assert enc.seen == ["what is radar"]


def test_dense_without_encoder_is_refused():
    with pytest.raises(ValueError):
        build_retriever({"retrieval": {"mode": "dense"}}, StubStore())


def test_hybrid_is_refused_for_generation():
    with pytest.raises(ValueError):
        build_retriever({"retrieval": {"mode": "hybrid"}}, StubStore())
