"""Embedding with a content-addressed cache.

Six chunking strategies times three embedding models means the same text is
embedded many times across the ablation grid. Caching on a hash of
(model, text) makes every repeat run free, which matters when a full sweep
is otherwise tens of minutes of CPU.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key   TEXT PRIMARY KEY,
    dim   INTEGER NOT NULL,
    vec   BLOB NOT NULL
)
"""


class EmbeddingCache:
    """SQLite-backed cache. stdlib only, single file, safe to delete."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(_SCHEMA)
        self.conn.commit()

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()

    def get_many(self, model: str, texts: list[str]) -> dict[int, np.ndarray]:
        keys = [self.key(model, t) for t in texts]
        found: dict[str, np.ndarray] = {}
        # Chunked IN clause: SQLite caps variables per statement.
        for i in range(0, len(keys), 500):
            batch = keys[i : i + 500]
            q = f"SELECT key, dim, vec FROM embeddings WHERE key IN ({','.join('?' * len(batch))})"
            for k, dim, blob in self.conn.execute(q, batch):
                found[k] = np.frombuffer(blob, dtype=np.float32, count=dim)
        return {i: found[k] for i, k in enumerate(keys) if k in found}

    def put_many(self, model: str, texts: list[str], vectors: np.ndarray) -> None:
        rows = [
            (self.key(model, t), int(v.shape[0]), v.astype(np.float32).tobytes())
            for t, v in zip(texts, vectors)
        ]
        self.conn.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)", rows)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class Encoder:
    """sentence-transformers wrapper that consults the cache first."""

    def __init__(self, model_name: str, cache_path: Path | None = None, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self.cache = EmbeddingCache(cache_path) if cache_path else None

    @property
    def model(self):
        """Loaded lazily - importing torch costs seconds even on a cache hit."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        cached = self.cache.get_many(self.model_name, texts) if self.cache else {}
        missing = [i for i in range(len(texts)) if i not in cached]
        logger.info("Embedding %d texts (%d cached, %d to compute)",
                    len(texts), len(cached), len(missing))

        if missing:
            fresh = self.model.encode(
                [texts[i] for i in missing],
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
            ).astype(np.float32)
            if self.cache:
                self.cache.put_many(self.model_name, [texts[i] for i in missing], fresh)
            for slot, i in enumerate(missing):
                cached[i] = fresh[slot]

        return np.vstack([cached[i] for i in range(len(texts))])

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]