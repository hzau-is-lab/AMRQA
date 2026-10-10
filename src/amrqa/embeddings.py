"""Embedding backends with an offline deterministic alternative for tests."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import numpy as np

from .config import EmbeddingConfig


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalized vector per input text."""


_TOKEN_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Small deterministic lexical embedder for tests and smoke runs only."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN_PATTERN.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "little") % self.dimensions
                matrix[row, index] += 1.0
        return _normalize_rows(matrix)


class SentenceTransformerEmbedder:
    """Sentence-Transformers wrapper that always returns CPU NumPy vectors."""

    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "SentenceTransformer is required for the configured embedding model. "
                "Install with: pip install -e '.[models]'"
            ) from exc
        resolved_device = None if device == "auto" else device
        self.model = SentenceTransformer(model_name, device=resolved_device)
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(config: EmbeddingConfig) -> Embedder:
    if config.model_name.lower() == "hashing":
        return HashingEmbedder()
    model_name = os.getenv(config.model_path_env) or config.model_name
    return SentenceTransformerEmbedder(model_name, config.device, config.batch_size)


def cosine_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Return cosine scores for normalized or unnormalized NumPy vectors."""
    if candidates.size == 0:
        return np.array([], dtype=np.float32)
    query = query.reshape(1, -1)
    query = _normalize_rows(query)
    candidates = _normalize_rows(candidates)
    return (query @ candidates.T).reshape(-1)
