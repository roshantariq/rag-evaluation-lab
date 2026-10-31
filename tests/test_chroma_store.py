"""Tests for ChromaStore's declared ef_search, on a throwaway Chroma directory."""

from __future__ import annotations

import pytest

from rageval.store.chroma_store import ChromaStore


def test_creation_persists_the_declared_ef_search_and_cosine_space(tmp_path):
    store = ChromaStore(tmp_path, "col", ef_search=500)
    hnsw = store.collection.configuration["hnsw"]
    assert store.ef_search == 500 and hnsw["space"] == "cosine"


def test_reopening_with_another_value_is_refused(tmp_path):
    ChromaStore(tmp_path, "col", ef_search=500)
    with pytest.raises(ValueError, match="persists ef_search=500"):
        ChromaStore(tmp_path, "col", ef_search=100)


def test_no_declaration_opens_without_a_check(tmp_path):
    ChromaStore(tmp_path, "col", ef_search=500)
    assert ChromaStore(tmp_path, "col").ef_search == 500


def test_set_ef_search_persists_across_clients(tmp_path):
    ChromaStore(tmp_path, "col", ef_search=100).set_ef_search(500)
    assert ChromaStore(tmp_path, "col", ef_search=500).ef_search == 500


def test_reset_recreates_with_the_declared_value(tmp_path):
    store = ChromaStore(tmp_path, "col", ef_search=300)
    store.reset()
    assert store.ef_search == 300
