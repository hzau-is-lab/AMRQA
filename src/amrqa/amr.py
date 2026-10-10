"""AMR anchor-to-unknown simple-path extraction and verbalization."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Protocol

from .config import AMRConfig, ModelConfig
from .llm import ChatClient
from .prompts import SYSTEM_PROMPT, reference_decomposition_prompt


class ReferenceQuestionProvider(Protocol):
    def extract(self, question: str) -> list[str]:
        """Return AMR-derived reference sub-questions for one input question."""


class QuestionFallbackProvider:
    """Question-reference control used only when AMR is explicitly disabled."""

    def extract(self, question: str) -> list[str]:
        return [question]


class LLMReferenceProvider:
    """Generate the non-AMR reference set used by the LLM-Ref control."""

    def __init__(
        self,
        config: AMRConfig,
        model_config: ModelConfig,
        client: ChatClient,
        timeout_seconds: float,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.client = client
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _parse_questions(text: str, limit: int) -> list[str]:
        match = re.search(r"\[[\s\S]*\]", text)
        if match is None:
            raise RuntimeError("The LLM reference generator did not return a JSON array")
        try:
            values = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise RuntimeError("The LLM reference generator returned invalid JSON") from exc
        if not isinstance(values, list):
            raise RuntimeError("The LLM reference generator did not return a list")
        references = list(
            dict.fromkeys(str(value).strip() for value in values if str(value).strip())
        )[:limit]
        if not references:
            raise RuntimeError("The LLM reference generator returned no usable sub-question")
        return references

    def extract(self, question: str) -> list[str]:
        response = self.client.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": reference_decomposition_prompt(question, self.config.max_paths),
                },
            ],
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens,
            timeout_seconds=self.timeout_seconds,
        )
        return self._parse_questions(response, self.config.max_paths)


@dataclass(frozen=True)
class AMRPath:
    nodes: tuple[str, ...]
    relations: tuple[str, ...]


class AMRLibPathProvider:
    """Extract and verbalize simple anchor-to-``amr-unknown`` paths.

    The AMR models are loaded lazily so importing or testing the package never downloads a
    checkpoint.
    """

    def __init__(self, config: AMRConfig) -> None:
        self.config = config
        self._stog = None
        self._gtos = None

    def _load_models(self) -> None:
        if self._stog is not None:
            return
        try:
            import amrlib
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("AMR support requires: pip install -e '.[amr]'") from exc
        parser_dir = os.getenv(self.config.parser_model_dir_env) or None
        generator_dir = os.getenv(self.config.generator_model_dir_env) or None
        try:
            self._stog = amrlib.load_stog_model(model_dir=parser_dir)
            self._gtos = amrlib.load_gtos_model(model_dir=generator_dir)
        except Exception as exc:  # pragma: no cover - depends on local AMR checkpoints
            raise RuntimeError(
                "AMRlib could not load its parser/generator checkpoints. Set "
                f"{self.config.parser_model_dir_env} and {self.config.generator_model_dir_env} "
                "to valid AMRlib model directories."
            ) from exc

    @staticmethod
    def _graph_parts(
        graph: object,
    ) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]], set[str]]:
        triples = graph.triples  # type: ignore[attr-defined]
        concepts = {
            str(source): str(target) for source, role, target in triples if role == ":instance"
        }
        adjacency: dict[str, list[tuple[str, str]]] = {node: [] for node in concepts}
        anchor_nodes: set[str] = set()
        for source, role, target in triples:
            source, role, target = str(source), str(role), str(target)
            if role == ":instance":
                continue
            if role in {":name", ":wiki", ":quant", ":value", ":year", ":month", ":day"}:
                anchor_nodes.add(source)
            if target in concepts:
                adjacency.setdefault(source, []).append((target, role))
                adjacency.setdefault(target, []).append((source, f"{role}-of"))
        # Named entities in AMR appear as a parent concept connected by :name.
        if not anchor_nodes:
            anchor_nodes = {
                node for node, concept in concepts.items() if concept not in {"amr-unknown", "name"}
            }
        return concepts, adjacency, anchor_nodes

    @staticmethod
    def _simple_paths(
        adjacency: dict[str, list[tuple[str, str]]],
        source: str,
        target: str,
        max_hops: int,
        limit: int,
    ) -> list[AMRPath]:
        """Enumerate cycle-free paths without collapsing alternatives to one shortest path."""
        paths: list[AMRPath] = []
        stack: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [(source, (source,), ())]
        while stack and len(paths) < limit:
            node, nodes, relations = stack.pop()
            if node == target:
                paths.append(AMRPath(nodes, relations))
                continue
            if len(relations) >= max_hops:
                continue
            neighbours = sorted(adjacency.get(node, []), reverse=True)
            for neighbour, relation in neighbours:
                if neighbour in nodes:
                    continue
                stack.append((neighbour, (*nodes, neighbour), (*relations, relation)))
        return paths

    def _verbalize(self, graph: object, path: AMRPath) -> str:
        """Verbalize a selected AMR subgraph with the configured graph-to-text model."""
        try:  # pragma: no cover - depends on AMR models and their output
            import penman

            selected = set(path.nodes)
            # Preserve entity names and scalar attributes attached to nodes on the path so the
            # verbalized reference remains grounded in the observed entities.
            changed = True
            while changed:
                changed = False
                for source, role, target in graph.triples:  # type: ignore[attr-defined]
                    source_text, role_text, target_text = str(source), str(role), str(target)
                    keep_attribute = role_text in {
                        ":name",
                        ":wiki",
                        ":quant",
                        ":value",
                        ":year",
                        ":month",
                        ":day",
                    } or role_text.startswith(":op")
                    if source_text in selected and keep_attribute and target_text not in selected:
                        selected.add(target_text)
                        changed = True
            triples = []
            for source, role, target in graph.triples:  # type: ignore[attr-defined]
                if str(source) in selected and (role == ":instance" or str(target) in selected):
                    triples.append((source, role, target))
            subgraph = penman.Graph(triples, top=path.nodes[-1])
            output = self._gtos.generate([penman.encode(subgraph)])
            sentences = output[0] if isinstance(output, tuple) else output
            generated = str(sentences[0]).strip() if sentences else ""
            if not generated:
                raise RuntimeError("The AMR-to-text model returned no text")
            generated = generated.rstrip(".")
            return generated if generated.endswith("?") else f"What is {generated}?"
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("AMR-to-text generation failed") from exc

    def extract(self, question: str) -> list[str]:
        self._load_models()
        try:
            try:
                import penman
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("AMR support requires: pip install -e '.[amr]'") from exc

            encoded = self._stog.parse_sents([question])[0]
            graph = penman.decode(encoded)
            concepts, adjacency, anchors = self._graph_parts(graph)
            unknowns = [node for node, concept in concepts.items() if concept == "amr-unknown"]
            if not unknowns:
                raise RuntimeError("The parsed AMR graph contains no amr-unknown node")
            paths: list[AMRPath] = []
            for unknown in unknowns:
                for anchor in anchors:
                    if anchor == unknown:
                        continue
                    pair_paths = self._simple_paths(
                        adjacency,
                        anchor,
                        unknown,
                        self.config.max_hops,
                        limit=max(self.config.max_paths * 4, self.config.max_paths),
                    )
                    paths.extend(path for path in pair_paths if path not in paths)
            # Prefer compositional paths; tie-break deterministically by node identifiers.
            paths.sort(key=lambda value: (-len(value.relations), value.nodes))
            references = [self._verbalize(graph, path) for path in paths[: self.config.max_paths]]
            unique_references = list(
                dict.fromkeys(reference for reference in references if reference.strip())
            )
            if not unique_references:
                raise RuntimeError("No AMR anchor-to-unknown reference could be constructed")
            return unique_references
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"AMR reference extraction failed: {exc}") from exc


def build_reference_provider(
    config: AMRConfig,
    *,
    client: Optional[ChatClient] = None,
    model_config: Optional[ModelConfig] = None,
    timeout_seconds: float = 180.0,
) -> ReferenceQuestionProvider:
    if config.reference_source == "amr":
        return AMRLibPathProvider(config)
    if config.reference_source == "question":
        return QuestionFallbackProvider()
    if client is None or model_config is None:
        raise ValueError("The LLM reference source requires a chat client and model configuration")
    return LLMReferenceProvider(config, model_config, client, timeout_seconds)
