"""Typed records passed between AMRQA pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class EvidenceUnit:
    """A benchmark-native evidence unit retained inside one retrievable passage."""

    id: str
    text: str


@dataclass(frozen=True)
class Document:
    """A retrievable evidence unit associated with one QA example."""

    id: str
    text: str
    title: str = ""
    native_units: tuple[EvidenceUnit, ...] = ()


@dataclass(frozen=True)
class QAExample:
    """A normalized multi-hop QA example."""

    id: str
    question: str
    answer: Optional[str]
    documents: tuple[Document, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateScore:
    """The two scoring signals and joint score for one sub-question."""

    question: str
    amr_reference_similarity: float
    semantic_relevance: float
    joint: float


@dataclass(frozen=True)
class Evidence:
    """A retrieved passage and its reranking score."""

    document_id: str
    title: str
    text: str
    dense_score: float
    rerank_score: float
    native_unit_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Fact:
    """An atomic fact selected for an observation."""

    text: str
    relevance: float
    source_document_id: str
    source_unit_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReasoningStep:
    """One Thought–Action–Observation transition in a reasoning path."""

    depth: int
    thought: str
    sub_question: str
    candidates: tuple[CandidateScore, ...]
    evidence: tuple[Evidence, ...]
    facts: tuple[Fact, ...]
    observation: str


@dataclass(frozen=True)
class ReasoningPath:
    """A complete or partial reasoning trajectory."""

    id: str
    steps: tuple[ReasoningStep, ...] = ()
    score: float = 0.0
    completed: bool = False
    answer: Optional[str] = None

    def history(self, question: str) -> str:
        """Render the trace as compact, model-readable context."""
        sections = [f"Original question: {question}"]
        for step in self.steps:
            sections.append(
                "\n".join(
                    (
                        f"Thought {step.depth}: {step.thought}",
                        f"Action {step.depth}: {step.sub_question}",
                        f"Observation {step.depth}: {step.observation}",
                    )
                )
            )
        return "\n\n".join(sections)


@dataclass(frozen=True)
class Prediction:
    """Serialized result of one AMRQA inference run."""

    id: str
    question: str
    gold_answer: Optional[str]
    prediction: str
    amr_references: tuple[str, ...]
    selected_path: Optional[ReasoningPath]
    completed_paths: tuple[ReasoningPath, ...]
    explored_paths: tuple[ReasoningPath, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
