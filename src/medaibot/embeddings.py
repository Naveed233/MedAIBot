"""Embedding backends. All return L2-normalised float32 rows so inner product = cosine."""

from collections.abc import Sequence
from hashlib import blake2b
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from medaibot.text import tokenize

if TYPE_CHECKING:
    from medaibot.config import Settings


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)


class HashingEmbedder:
    """Feature hashing over tokens. Deterministic and offline, but lexical, not semantic.

    Used by tests and CI so that nothing downloads a model.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            h = int.from_bytes(blake2b(token.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalise(self._vector(text)[None, :])[0]


class FastEmbedEmbedder:
    """Local ONNX embedding model through fastembed (no PyTorch). Downloads once to cache_dir."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        self.dim = int(self._model.embedding_size)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = list(self._model.passage_embed(list(texts), batch_size=64))
        return _l2_normalise(np.asarray(rows, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        row = next(iter(self._model.query_embed(text)))
        return _l2_normalise(np.asarray(row, dtype=np.float32)[None, :])[0]


def make_embedder(settings: "Settings") -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    return FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir)
