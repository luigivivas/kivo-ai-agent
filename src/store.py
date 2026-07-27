"""Índice vectorial minimalista: numpy + similitud coseno, sin dependencias pesadas."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .loaders import Chunk

_META_FILE = "chunks.json"
_VECTORS_FILE = "vectors.npy"


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.chunks = chunks
        self.vectors = vectors / norms

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        q = query_vector / (np.linalg.norm(query_vector) or 1.0)
        scores = self.vectors @ q
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / _VECTORS_FILE, self.vectors)
        with open(index_dir / _META_FILE, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False)

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        vectors = np.load(index_dir / _VECTORS_FILE)
        with open(index_dir / _META_FILE, encoding="utf-8") as f:
            raw_chunks = json.load(f)
        chunks = [Chunk(**c) for c in raw_chunks]
        store = cls.__new__(cls)
        store.chunks = chunks
        store.vectors = vectors
        return store

    @staticmethod
    def exists(index_dir: Path) -> bool:
        return (index_dir / _VECTORS_FILE).exists() and (index_dir / _META_FILE).exists()
