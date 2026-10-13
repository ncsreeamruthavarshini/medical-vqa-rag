"""Text embeddings for retrieval.

An embedding is a list of numbers that represents the meaning of a text.
Two texts with similar meanings usually have vectors that are close together
in that number space. This project uses a small sentence-transformers model
to turn medical passages and questions into vectors of the same length.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class TextEncoder(Protocol):
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape (n_texts, dimension)."""


class SentenceTransformerEncoder:
    """Wrapper around sentence-transformers so tests can swap in a fake encoder."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self._model = SentenceTransformer(model_id)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dim = self.dimension
            return np.zeros((0, dim), dtype=np.float32)
        vectors = self._model.encode(list(texts), convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


class HashOverlapEncoder:
    """Deterministic encoder for unit tests. Similar words yield closer vectors."""

    def __init__(self, dimension: int = 32):
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension
        self.model_id = "hash-overlap-test-encoder"

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        rows = [self._vector(text) for text in texts]
        if not rows:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack(rows, axis=0)

    def _vector(self, text: str) -> np.ndarray:
        import hashlib

        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = str(text).lower().split()
        if not tokens:
            vec[0] = 1.0
            return vec
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest, 16) % self.dimension
            vec[index] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)
