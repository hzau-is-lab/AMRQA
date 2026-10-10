import sys
from types import SimpleNamespace

import pytest

from amrqa.amr import AMRLibPathProvider, AMRPath, LLMReferenceProvider
from amrqa.config import AMRConfig, ModelConfig
from amrqa.llm import LLMUsage


class ReferenceClient:
    def complete(self, messages, *, temperature, max_tokens, timeout_seconds=None) -> str:
        return '["Who directed the film?", "Who is the director\'s mother?"]'

    def usage_snapshot(self) -> LLMUsage:
        return LLMUsage(calls=1)


def test_simple_path_enumeration_preserves_alternative_routes() -> None:
    adjacency = {
        "anchor": [("left", ":left"), ("right", ":right")],
        "left": [("target", ":to")],
        "right": [("target", ":to")],
        "target": [],
    }
    paths = AMRLibPathProvider._simple_paths(adjacency, "anchor", "target", 3, 8)
    assert {path.nodes for path in paths} == {
        ("anchor", "left", "target"),
        ("anchor", "right", "target"),
    }


def test_llm_reference_control_returns_json_subquestions() -> None:
    provider = LLMReferenceProvider(
        AMRConfig(reference_source="llm", max_paths=2),
        ModelConfig(),
        ReferenceClient(),
        180.0,
    )
    assert provider.extract("Who is the director's mother?") == [
        "Who directed the film?",
        "Who is the director's mother?",
    ]


def test_amr_verbalization_failure_is_not_silently_replaced(monkeypatch) -> None:
    class FailingGenerator:
        def generate(self, graphs):
            raise ValueError("generation failed")

    class FakeGraph:
        triples = [("a", ":instance", "author"), ("u", ":instance", "amr-unknown")]

    fake_penman = SimpleNamespace(
        Graph=lambda triples, top: SimpleNamespace(triples=triples, top=top),
        encode=lambda graph: "(u / amr-unknown)",
    )
    monkeypatch.setitem(sys.modules, "penman", fake_penman)
    provider = AMRLibPathProvider(AMRConfig())
    provider._gtos = FailingGenerator()

    with pytest.raises(RuntimeError, match="AMR-to-text generation failed"):
        provider._verbalize(FakeGraph(), AMRPath(("a", "u"), (":ARG0",)))
