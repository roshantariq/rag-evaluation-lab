"""Tests for rageval.embed.encoder: the cache path and the device passthrough."""

from __future__ import annotations

import sys
import types

import numpy as np

from rageval.embed.encoder import Encoder


class FakeModel:
    calls: list[list[str]] = []

    def __init__(self, name, device=None):
        self.name, self.device = name, device

    def encode(self, texts, batch_size, show_progress_bar, normalize_embeddings):
        FakeModel.calls.append(list(texts))
        return np.ones((len(texts), 3), dtype=np.float32) * len(texts)


def install_fake(monkeypatch):
    FakeModel.calls = []
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeModel))


def test_device_reaches_the_model(monkeypatch):
    install_fake(monkeypatch)
    assert Encoder("m", device="cpu").model.device == "cpu"
    assert Encoder("m").model.device is None


def test_cached_text_is_not_recomputed(tmp_path, monkeypatch):
    install_fake(monkeypatch)
    cache = tmp_path / "e.sqlite"
    Encoder("m", cache_path=cache).encode(["a", "b"])
    Encoder("m", cache_path=cache).encode(["a", "b", "c"])
    assert FakeModel.calls == [["a", "b"], ["c"]]


def test_no_cache_always_computes_and_writes_nothing(tmp_path, monkeypatch):
    install_fake(monkeypatch)
    enc = Encoder("m", cache_path=None)
    enc.encode(["a"])
    enc.encode(["a"])
    assert FakeModel.calls == [["a"], ["a"]] and enc.cache is None
    assert list(tmp_path.iterdir()) == []
