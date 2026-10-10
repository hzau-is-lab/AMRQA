"""Dense retrieval followed by a configurable cross-encoder or lexical reranker."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import RerankerConfig
from .domain import Document, Evidence
from .embeddings import Embedder, cosine_scores


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score query-passage pairs in the same order as `passages`."""


class LexicalReranker:
    """Token-overlap reranker used only for offline tests or unavailable model dependencies."""

    _token_pattern = re.compile(r"[\w'-]+", flags=re.UNICODE)

    def score(self, query: str, passages: list[str]) -> list[float]:
        query_terms = set(self._token_pattern.findall(query.lower()))
        if not query_terms:
            return [0.0] * len(passages)
        results = []
        for passage in passages:
            passage_terms = set(self._token_pattern.findall(passage.lower()))
            results.append(
                len(query_terms & passage_terms) / len(query_terms | passage_terms or {"_"})
            )
        return results


class FlagEmbeddingReranker:
    """FlagEmbedding cross-encoder wrapper."""

    def __init__(self, config: RerankerConfig) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "FlagEmbedding is required for the configured reranker. "
                "Install with: pip install -e '.[models]'"
            ) from exc
        device = None if config.device == "auto" else config.device
        kwargs = {"use_fp16": config.use_fp16}
        if device:
            kwargs["devices"] = [device]
        self.model = FlagReranker(config.model_name, **kwargs)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.compute_score([[query, passage] for passage in passages])
        if isinstance(scores, float):
            scores = [scores]
        return [float(score) for score in scores]


def build_reranker(config: RerankerConfig) -> Reranker:
    if config.model_name.lower() == "lexical":
        return LexicalReranker()
    if local_path := os.getenv(config.model_path_env):
        config = RerankerConfig(
            model_name=local_path,
            model_path_env=config.model_path_env,
            device=config.device,
            use_fp16=config.use_fp16,
        )
    return FlagEmbeddingReranker(config)


@dataclass(frozen=True)
class RetrievalIndex:
    documents: tuple[Document, ...]
    embeddings: np.ndarray


class EvidenceRetriever:
    def __init__(self, embedder: Embedder, reranker: Reranker) -> None:
        self.embedder = embedder
        self.reranker = reranker

    def index(self, documents: tuple[Document, ...]) -> RetrievalIndex:
        passages = [document.text for document in documents]
        return RetrievalIndex(documents=documents, embeddings=self.embedder.encode(passages))

    def retrieve(
        self, index: RetrievalIndex, query: str, retrieve_k: int, rerank_k: int
    ) -> list[Evidence]:
        if not index.documents:
            return []
        query_embedding = self.embedder.encode([query])[0]
        dense_scores = cosine_scores(query_embedding, index.embeddings)
        candidate_count = min(retrieve_k, len(index.documents))
        candidate_indices = np.argsort(-dense_scores)[:candidate_count].tolist()
        passages = [index.documents[item].text for item in candidate_indices]
        rerank_scores = self.reranker.score(query, passages)
        if len(rerank_scores) != len(candidate_indices):
            raise RuntimeError("Reranker returned an unexpected number of scores")
        ranked = sorted(
            zip(candidate_indices, rerank_scores),
            key=lambda item: (-item[1], -float(dense_scores[item[0]]), item[0]),
        )[: min(rerank_k, candidate_count)]
        return [
            Evidence(
                document_id=index.documents[item].id,
                title=index.documents[item].title,
                text=index.documents[item].text,
                dense_score=round(float(dense_scores[item]), 6),
                rerank_score=round(float(rerank_score), 6),
                native_unit_ids=tuple(unit.id for unit in index.documents[item].native_units),
            )
            for item, rerank_score in ranked
        ]
