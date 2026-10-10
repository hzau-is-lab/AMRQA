"""The AMRQA multi-path Thought–Action–Observation inference loop."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import replace
from typing import Any, Optional

from .amr import ReferenceQuestionProvider
from .config import AMRQAConfig
from .domain import (
    CandidateScore,
    Document,
    Fact,
    Prediction,
    QAExample,
    ReasoningPath,
    ReasoningStep,
)
from .embeddings import Embedder, cosine_scores
from .facts import AtomicFactExtractor
from .llm import ChatClient, LLMUsage, Message
from .prompt_assets import load_atomic_fact_demonstrations, load_react_demonstrations
from .prompts import (
    SYSTEM_PROMPT,
    candidate_prompt,
    final_answer_prompt,
    sufficiency_prompt,
)
from .retrieval import EvidenceRetriever, RetrievalIndex
from .scoring import CandidateEvaluator

LOGGER = logging.getLogger(__name__)
_NUMBERED_ITEM = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$")


class QuestionTimeoutError(TimeoutError):
    """Raised when a question exceeds the configured end-to-end runtime limit."""


def _json_value(text: str) -> Optional[Any]:
    """Extract a JSON object or array even when a model wraps it in a code fence."""
    starts = [position for position in (text.find("["), text.find("{")) if position >= 0]
    if not starts:
        return None
    start = min(starts)
    closing = "]" if text[start] == "[" else "}"
    end = text.rfind(closing)
    if end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def parse_candidates(text: str, expected_count: int) -> list[str]:
    """Parse model output while rejecting duplicates and empty candidate questions."""
    value = _json_value(text)
    if isinstance(value, list):
        candidates = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    else:
        candidates = []
        for line in text.splitlines():
            match = _NUMBERED_ITEM.match(line)
            candidate = match.group(1).strip() if match else ""
            if candidate:
                candidates.append(candidate)
    unique = list(dict.fromkeys(candidate.rstrip() for candidate in candidates))
    return unique[:expected_count]


def parse_decision(text: str) -> tuple[bool, str]:
    value = _json_value(text)
    if not isinstance(value, dict):
        return False, "The sufficiency response was not valid JSON."
    sufficient = value.get("sufficient") is True
    thought = str(value.get("thought", "")).strip() or "No reasoning note was returned."
    return sufficient, thought


class AMRQAReasoner:
    """Implements AMR-guided candidate selection and evidence-grounded beam reasoning."""

    def __init__(
        self,
        config: AMRQAConfig,
        client: ChatClient,
        embedder: Embedder,
        references: ReferenceQuestionProvider,
        retriever: EvidenceRetriever,
    ) -> None:
        self.config = config
        self.client = client
        self.embedder = embedder
        self.references = references
        self.retriever = retriever
        self.candidate_evaluator = CandidateEvaluator(
            embedder=embedder,
            client=client,
            amr_reference_weight=config.reasoning.amr_reference_weight,
            scoring_mode=config.reasoning.candidate_scoring_mode,
            temperature=config.model.temperature,
            max_tokens=config.model.max_tokens,
        )
        self.demonstrations = (
            load_react_demonstrations() if config.prompts.use_react_demonstrations else ""
        )
        self.atomic_fact_demonstrations = (
            load_atomic_fact_demonstrations()
            if config.prompts.use_atomic_fact_demonstrations
            else ""
        )
        self.fact_extractor = AtomicFactExtractor(
            client,
            config.model.temperature,
            config.prompts.atomic_fact_max_tokens,
            self.atomic_fact_demonstrations,
        )
        self._deadline: Optional[float] = None
        self._retrieval_calls = 0

    def _remaining_timeout(self) -> float:
        if self._deadline is None:
            return self.config.runtime.timeout_seconds
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise QuestionTimeoutError(
                f"Question exceeded the {self.config.runtime.timeout_seconds:g}-second limit"
            )
        return remaining

    def _check_timeout(self) -> None:
        self._remaining_timeout()

    def _usage_snapshot(self) -> LLMUsage:
        snapshot = getattr(self.client, "usage_snapshot", None)
        return snapshot() if callable(snapshot) else LLMUsage()

    def _complete(self, prompt: str, *, temperature: Optional[float] = None) -> str:
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self.client.complete(
            messages,
            temperature=(self.config.model.temperature if temperature is None else temperature),
            max_tokens=self.config.model.max_tokens,
            timeout_seconds=self._remaining_timeout(),
        )

    def _assess(self, question: str, path: ReasoningPath) -> tuple[bool, str]:
        return parse_decision(self._complete(sufficiency_prompt(question, path.history(question))))

    def _generate_candidates(self, question: str, path: ReasoningPath) -> list[str]:
        history = (
            path.history(question)
            if self.config.reasoning.candidate_history_mode == "full"
            else f"Original question: {question}"
        )
        response = self._complete(
            candidate_prompt(
                question,
                history,
                self.config.reasoning.candidate_count,
                self.demonstrations,
            ),
            temperature=self.config.model.candidate_generation_temperature,
        )
        candidates = parse_candidates(response, self.config.reasoning.candidate_count)
        if not candidates:
            LOGGER.warning("No valid candidate sub-questions were returned for path %s", path.id)
        return candidates

    def _answer(self, question: str, path: ReasoningPath) -> str:
        answer = self._complete(final_answer_prompt(question, path.history(question))).strip()
        return answer or self.config.reasoning.no_answer

    def _observe(
        self, question: str, retrieval_index: RetrievalIndex
    ) -> tuple[tuple[Any, ...], tuple[Fact, ...], str]:
        self._check_timeout()
        evidence = self.retriever.retrieve(
            retrieval_index,
            question,
            retrieve_k=self.config.reasoning.retrieval_k,
            rerank_k=self.config.reasoning.rerank_k,
        )
        self._retrieval_calls += 1
        self._check_timeout()
        documents = {document.id: document for document in retrieval_index.documents}
        candidate_facts: list[tuple[str, Document, Optional[tuple[str, ...]]]] = []
        for item in evidence:
            document = documents[item.document_id]
            if self.config.reasoning.observation_mode == "sentences":
                if document.native_units:
                    candidate_facts.extend(
                        (unit.text, document, (unit.id,)) for unit in document.native_units
                    )
                else:
                    candidate_facts.append((document.text, document, ()))
                continue
            try:
                facts = self.fact_extractor.extract(item.text, self._remaining_timeout())
            except Exception as exc:
                if isinstance(exc, (QuestionTimeoutError, TimeoutError)):
                    raise QuestionTimeoutError(str(exc)) from exc
                LOGGER.warning("Atomic fact extraction failed for %s: %s", item.document_id, exc)
                facts = []
            candidate_facts.extend((fact, document, None) for fact in facts)

        if not candidate_facts:
            observation = "No atomic facts were extracted from the retrieved evidence."
            return tuple(evidence), (), observation

        texts = [fact for fact, _, _ in candidate_facts]
        query_embedding = self.embedder.encode([question])[0]
        fact_embeddings = self.embedder.encode(texts)
        scores = cosine_scores(query_embedding, fact_embeddings)
        ranked_indices = sorted(
            range(len(texts)), key=lambda index: (-float(scores[index]), texts[index])
        )
        selected: list[Fact] = []
        seen: set[str] = set()
        for index in ranked_indices:
            text, document, explicit_unit_ids = candidate_facts[index]
            if text in seen:
                continue
            seen.add(text)
            source_unit_ids: tuple[str, ...] = explicit_unit_ids or ()
            if explicit_unit_ids is None and document.native_units:
                unit_texts = [unit.text for unit in document.native_units]
                unit_scores = cosine_scores(
                    self.embedder.encode([text])[0], self.embedder.encode(unit_texts)
                )
                best_unit = int(max(range(len(unit_scores)), key=lambda item: unit_scores[item]))
                source_unit_ids = (document.native_units[best_unit].id,)
            selected.append(
                Fact(
                    text=text,
                    relevance=round(float(scores[index]), 6),
                    source_document_id=document.id,
                    source_unit_ids=source_unit_ids,
                )
            )
            if len(selected) == self.config.reasoning.facts_per_observation:
                break
        self._check_timeout()
        label = (
            "Selected atomic facts"
            if self.config.reasoning.observation_mode == "atomic_facts"
            else "Selected evidence sentences"
        )
        observation = f"{label}:\n" + "\n".join(f"- {fact.text}" for fact in selected)
        return tuple(evidence), tuple(selected), observation

    @staticmethod
    def _extend_score(path: ReasoningPath, candidate: CandidateScore) -> float:
        previous = len(path.steps)
        return round((path.score * previous + candidate.joint) / (previous + 1), 6)

    def _expand_path(
        self,
        question: str,
        references: list[str],
        retrieval_index: RetrievalIndex,
        path: ReasoningPath,
        depth: int,
        thought: str,
    ) -> list[ReasoningPath]:
        candidates = self._generate_candidates(question, path)
        scored = self.candidate_evaluator.evaluate(
            question,
            candidates,
            references,
            timeout_provider=self._remaining_timeout,
        )
        self._check_timeout()
        retained = scored[: self.config.reasoning.path_width]
        children: list[ReasoningPath] = []
        for branch, candidate in enumerate(retained, start=1):
            evidence, facts, observation = self._observe(candidate.question, retrieval_index)
            step = ReasoningStep(
                depth=depth,
                thought=thought,
                sub_question=candidate.question,
                candidates=tuple(scored),
                evidence=evidence,
                facts=facts,
                observation=observation,
            )
            children.append(
                ReasoningPath(
                    id=f"{path.id}.{branch}",
                    steps=(*path.steps, step),
                    score=self._extend_score(path, candidate),
                )
            )
        return children

    def predict(self, example: QAExample) -> Prediction:
        # Per-question dense index construction is excluded from the timed online stages, matching
        # the efficiency protocol in the manuscript.
        retrieval_index = self.retriever.index(example.documents)
        started = time.monotonic()
        self._deadline = started + self.config.runtime.timeout_seconds
        self._retrieval_calls = 0
        usage_before = self._usage_snapshot()
        references: list[str] = []
        active = [ReasoningPath(id="p0")]
        completed: list[ReasoningPath] = []
        explored: list[ReasoningPath] = []
        selected: Optional[ReasoningPath] = None
        iterations_executed = 0
        termination_reason = "no_active_paths"

        try:
            if self.config.reasoning.candidate_scoring_mode in {"joint", "amr_only"}:
                references = self.references.extract(example.question)
            self._check_timeout()
            for depth in range(1, self.config.reasoning.max_depth + 1):
                iterations_executed = depth
                next_paths: list[ReasoningPath] = []
                for path in active:
                    self._check_timeout()
                    sufficient, thought = self._assess(example.question, path)
                    if sufficient:
                        completed.append(replace(path, completed=True))
                        continue
                    children = self._expand_path(
                        example.question,
                        references,
                        retrieval_index,
                        path,
                        depth,
                        thought,
                    )
                    next_paths.extend(children)
                    if self.config.runtime.capture_explored_paths:
                        explored.extend(children)
                if completed:
                    termination_reason = "answered"
                    break
                active = sorted(next_paths, key=lambda item: (-item.score, item.id))[
                    : self.config.reasoning.path_width
                ]
                if not active:
                    termination_reason = "no_active_paths"
                    break
            else:
                termination_reason = "max_steps"
            if completed:
                selected = max(completed, key=lambda item: (item.score, item.id))
                selected = replace(
                    selected,
                    answer=self._answer(example.question, selected),
                )
                completed = [selected if path.id == selected.id else path for path in completed]
        except (QuestionTimeoutError, TimeoutError) as exc:
            LOGGER.warning("Question %s timed out: %s", example.id, exc)
            completed = []
            selected = None
            termination_reason = "timeout"
        finally:
            elapsed_seconds = time.monotonic() - started
            self._deadline = None

        if elapsed_seconds > self.config.runtime.timeout_seconds:
            completed = []
            selected = None
            termination_reason = "timeout"
        usage = self._usage_snapshot() - usage_before
        return Prediction(
            id=example.id,
            question=example.question,
            gold_answer=example.answer,
            prediction=selected.answer
            if selected and selected.answer
            else self.config.reasoning.no_answer,
            amr_references=tuple(references),
            selected_path=selected,
            completed_paths=tuple(completed),
            explored_paths=tuple(explored),
            metadata={
                "seed": self.config.seed,
                "model": self.config.model.model_name,
                "non_candidate_temperature": self.config.model.temperature,
                "candidate_generation_temperature": (
                    self.config.model.candidate_generation_temperature
                ),
                "max_retries": self.config.model.max_retries,
                "provider_usage_required": self.config.model.require_usage,
                "max_depth": self.config.reasoning.max_depth,
                "candidate_count": self.config.reasoning.candidate_count,
                "candidate_history_mode": self.config.reasoning.candidate_history_mode,
                "reference_source": self.config.amr.reference_source,
                "path_width": self.config.reasoning.path_width,
                "amr_reference_weight": self.config.reasoning.amr_reference_weight,
                "candidate_scoring_mode": self.config.reasoning.candidate_scoring_mode,
                "retrieval_k": self.config.reasoning.retrieval_k,
                "rerank_k": self.config.reasoning.rerank_k,
                "facts_per_observation": self.config.reasoning.facts_per_observation,
                "observation_mode": self.config.reasoning.observation_mode,
                "elapsed_seconds": round(elapsed_seconds, 6),
                "reasoning_steps": iterations_executed,
                "llm_calls": usage.calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "retrieval_calls": self._retrieval_calls,
                "termination_reason": termination_reason,
                "timed_out": termination_reason == "timeout",
                "max_step_termination": termination_reason == "max_steps",
                "timeout_seconds": self.config.runtime.timeout_seconds,
                "index_build_in_timing": False,
                "few_shot_react_demonstrations": (self.config.prompts.use_react_demonstrations),
                "few_shot_atomic_fact_demonstrations": (
                    self.config.prompts.use_atomic_fact_demonstrations
                ),
                "remaining_active_paths": len(active),
            },
        )
