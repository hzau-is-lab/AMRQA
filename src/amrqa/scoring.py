"""Joint AMR-reference similarity and semantic-relevance evaluation."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Optional

import numpy as np

from .domain import CandidateScore
from .embeddings import Embedder
from .llm import ChatClient, Message
from .prompts import SYSTEM_PROMPT, semantic_score_prompt

_SCORE_PATTERN = re.compile(r"(?:^|[^0-9])([01](?:\.\d+)?)")


def parse_score(text: str) -> float:
    match = _SCORE_PATTERN.search(text.strip())
    if not match:
        return 0.0
    return min(1.0, max(0.0, float(match.group(1))))


class CandidateEvaluator:
    def __init__(
        self,
        *,
        embedder: Embedder,
        client: ChatClient,
        amr_reference_weight: float,
        scoring_mode: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.embedder = embedder
        self.client = client
        self.amr_reference_weight = amr_reference_weight
        self.scoring_mode = scoring_mode
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _semantic_score(
        self,
        original_question: str,
        candidate: str,
        timeout_provider: Optional[Callable[[], float]],
    ) -> float:
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": semantic_score_prompt(original_question, candidate)},
        ]
        text = self.client.complete(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=timeout_provider() if timeout_provider else None,
        )
        return parse_score(text)

    def evaluate(
        self,
        original_question: str,
        candidates: list[str],
        references: list[str],
        timeout_provider: Optional[Callable[[], float]] = None,
    ) -> list[CandidateScore]:
        if not candidates:
            return []
        if self.scoring_mode in {"joint", "amr_only"}:
            candidate_embeddings = self.embedder.encode(candidates)
            if not references:
                raise ValueError(
                    "AMR-reference scoring requires at least one reference sub-question"
                )
            reference_embeddings = self.embedder.encode(references)
            reference_matrix = candidate_embeddings @ reference_embeddings.T
            reference_scores = np.max(reference_matrix, axis=1)
            if len(reference_scores) != len(candidates):
                raise RuntimeError("Embedding backend returned an unexpected candidate count")
        else:
            reference_scores = np.zeros(len(candidates), dtype=np.float32)
        results: list[CandidateScore] = []
        for candidate, reference_similarity in zip(candidates, reference_scores):
            semantic = (
                self._semantic_score(original_question, candidate, timeout_provider)
                if self.scoring_mode in {"joint", "semantic_only"}
                else 0.0
            )
            if self.scoring_mode == "joint":
                joint = (
                    self.amr_reference_weight * float(reference_similarity)
                    + (1 - self.amr_reference_weight) * semantic
                )
            elif self.scoring_mode == "amr_only":
                joint = float(reference_similarity)
            elif self.scoring_mode == "semantic_only":
                joint = semantic
            else:
                joint = 0.0
            results.append(
                CandidateScore(
                    question=candidate,
                    amr_reference_similarity=round(float(reference_similarity), 6),
                    semantic_relevance=round(semantic, 6),
                    joint=round(joint, 6),
                )
            )
        if self.scoring_mode == "none":
            return results
        return sorted(results, key=lambda item: (-item.joint, item.question))
