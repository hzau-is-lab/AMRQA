from typing import Optional

from amrqa.embeddings import HashingEmbedder
from amrqa.llm import LLMUsage
from amrqa.scoring import CandidateEvaluator


class ScoreClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        self.calls += 1
        return "0.8"

    def usage_snapshot(self) -> LLMUsage:
        return LLMUsage(calls=self.calls)


def _evaluator(mode: str, client: ScoreClient) -> CandidateEvaluator:
    return CandidateEvaluator(
        embedder=HashingEmbedder(),
        client=client,
        amr_reference_weight=0.5,
        scoring_mode=mode,
        temperature=0.0,
        max_tokens=128,
    )


def test_amr_only_does_not_call_semantic_evaluator() -> None:
    client = ScoreClient()
    values = _evaluator("amr_only", client).evaluate(
        "Original question?", ["Candidate question?"], ["Reference question?"]
    )
    assert client.calls == 0
    assert values[0].semantic_relevance == 0.0


def test_semantic_only_does_not_require_references() -> None:
    client = ScoreClient()
    values = _evaluator("semantic_only", client).evaluate(
        "Original question?", ["Candidate question?"], []
    )
    assert client.calls == 1
    assert values[0].joint == 0.8


def test_no_evaluator_preserves_generated_order() -> None:
    client = ScoreClient()
    values = _evaluator("none", client).evaluate(
        "Original question?", ["Z question?", "A question?"], []
    )
    assert client.calls == 0
    assert [value.question for value in values] == ["Z question?", "A question?"]
