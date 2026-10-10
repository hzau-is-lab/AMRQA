"""High-level experiment runner shared by the CLI and programmatic users."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from typing import Optional

import numpy as np

from .amr import build_reference_provider
from .config import AMRQAConfig
from .datasets import iter_jsonl
from .embeddings import build_embedder
from .llm import build_chat_client
from .reasoner import AMRQAReasoner
from .retrieval import EvidenceRetriever, build_reranker


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_reasoner(config: AMRQAConfig) -> AMRQAReasoner:
    client = build_chat_client(config.model, config.seed)
    embedder = build_embedder(config.embedding)
    references = build_reference_provider(
        config.amr,
        client=client,
        model_config=config.model,
        timeout_seconds=config.runtime.timeout_seconds,
    )
    retriever = EvidenceRetriever(embedder, build_reranker(config.reranker))
    return AMRQAReasoner(config, client, embedder, references, retriever)


def run_predictions(
    config: AMRQAConfig, data_path: str, limit: Optional[int] = None
) -> Iterator[dict[str, object]]:
    seed_everything(config.seed)
    reasoner = build_reasoner(config)
    logger = logging.getLogger(__name__)
    for index, example in enumerate(iter_jsonl(data_path, limit), start=1):
        logger.info("Processing %s (%d): %s", example.id, index, example.question)
        yield reasoner.predict(example).to_dict()
