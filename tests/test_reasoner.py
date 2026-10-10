from typing import Optional

from amrqa.config import AMRQAConfig
from amrqa.domain import Document, QAExample
from amrqa.embeddings import HashingEmbedder
from amrqa.llm import LLMUsage
from amrqa.reasoner import AMRQAReasoner
from amrqa.retrieval import EvidenceRetriever, LexicalReranker


class FixedReferences:
    def extract(self, question: str) -> list[str]:
        return ["Who directed Polish-Russian War?", "Who is the director's mother?"]


class ScriptedClient:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self.temperatures: list[tuple[str, float]] = []

    def complete(
        self,
        messages,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        self.calls += 1
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        self.temperatures.append((prompt, temperature))
        if "sufficient evidence" in prompt:
            if "Observation 1" in prompt:
                return '{"sufficient": true, "thought": "The answer is supported."}'
            return '{"sufficient": false, "thought": "I need the director and mother evidence."}'
        if "candidate sub-questions" in prompt:
            return '["Who directed Polish-Russian War?", "Who is Xawery Żuławski\'s mother?"]'
        if "semantic relevance" in prompt:
            return "0.900"
        if "final\nshort answer" in prompt:
            return "Małgorzata Braunek"
        if "atomic facts" in prompt:
            return (
                '["Polish-Russian War was directed by Xawery Żuławski.", '
                '"Xawery Żuławski is the son of Małgorzata Braunek."]'
            )
        raise AssertionError(f"Unexpected prompt: {prompt}")

    def usage_snapshot(self) -> LLMUsage:
        return LLMUsage(calls=self.calls)


def test_reasoner_returns_traceable_completed_path() -> None:
    config = AMRQAConfig()
    config.reasoning.max_depth = 2
    config.reasoning.candidate_count = 2
    config.reasoning.path_width = 1
    config.reasoning.retrieval_k = 2
    config.reasoning.rerank_k = 1
    config.reasoning.facts_per_observation = 2
    example = QAExample(
        id="example",
        question="Who is the mother of the director of Polish-Russian War?",
        answer="Małgorzata Braunek",
        documents=(
            Document("d1", "Polish-Russian War was directed by Xawery Żuławski."),
            Document("d2", "Xawery Żuławski is the son of Małgorzata Braunek."),
        ),
    )
    embedder = HashingEmbedder()
    client = ScriptedClient()
    reasoner = AMRQAReasoner(
        config,
        client,
        embedder,
        FixedReferences(),
        EvidenceRetriever(embedder, LexicalReranker()),
    )
    result = reasoner.predict(example)
    assert result.prediction == "Małgorzata Braunek"
    assert result.selected_path is not None
    assert result.selected_path.steps[0].facts
    assert result.selected_path.steps[0].candidates[0].joint > 0
    assert result.metadata["termination_reason"] == "answered"
    assert result.metadata["llm_calls"] > 0
    assert result.explored_paths
    candidate_prompts = [prompt for prompt in client.prompts if "candidate sub-questions" in prompt]
    assert candidate_prompts
    assert "Original question:" in candidate_prompts[0]
    assert all(
        temperature == 0.7
        for prompt, temperature in client.temperatures
        if "candidate sub-questions" in prompt
    )
    assert all(
        temperature == 0.0
        for prompt, temperature in client.temperatures
        if "candidate sub-questions" not in prompt
    )
