"""ChromaDB wrapper. Embeddings are supplied by us, never computed by Chroma.

Chroma can embed documents itself, but then the embedding cache is bypassed
and the model in use becomes implicit. Passing vectors in keeps one code
path for embedding and makes the ablation's model axis explicit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

logger = logging.getLogger(__name__)

_MAX_BATCH = 1000


@dataclass
class Retrieved:
    """One retrieval hit, carrying provenance for span-overlap scoring."""

    chunk_id: str
    arxiv_id: str
    text: str
    char_start: int
    char_end: int
    section: str
    title: str
    score: float
    rank: int

    def overlaps(self, start: int, end: int) -> bool:
        return self.char_start < end and start < self.char_end


class ChromaStore:
    def __init__(self, path: Path, collection: str):
        self.path = path
        self.name = collection
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        """Drop and recreate. Indexes are derived data; never patch in place."""
        try:
            self.client.delete_collection(self.name)
        except Exception:  # noqa: BLE001 - absent collection is fine
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list, embeddings: np.ndarray) -> None:
        for i in range(0, len(chunks), _MAX_BATCH):
            batch = chunks[i : i + _MAX_BATCH]
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=[e.tolist() for e in embeddings[i : i + _MAX_BATCH]],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "arxiv_id": c.arxiv_id,
                        "char_start": int(c.char_start),
                        "char_end": int(c.char_end),
                        "section": c.section or "(unknown)",
                        "title": c.title or "",
                        "published": c.published or "",
                        "strategy": c.strategy,
                    }
                    for c in batch
                ],
            )
            logger.info("  indexed %d/%d", min(i + _MAX_BATCH, len(chunks)), len(chunks))

    def query(self, embedding: np.ndarray, k: int = 5) -> list[Retrieved]:
        res = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]), 1
        ):
            out.append(
                Retrieved(
                    chunk_id=cid,
                    arxiv_id=meta["arxiv_id"],
                    text=doc,
                    char_start=int(meta["char_start"]),
                    char_end=int(meta["char_end"]),
                    section=meta.get("section", ""),
                    title=meta.get("title", ""),
                    score=1.0 - float(dist),  # cosine distance -> similarity
                    rank=rank,
                )
            )
        return out

    def count(self) -> int:
        return self.collection.count()