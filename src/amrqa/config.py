"""Validated, serializable configuration for AMRQA experiments."""

from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass, field
from typing import Any, Optional, Union, get_type_hints

import yaml


@dataclass
class ModelConfig:
    provider: str = "openai_compatible"
    model_name: str = "Qwen2.5-32B-Instruct"
    api_key_env: str = "AMRQA_API_KEY"
    base_url_env: str = "AMRQA_BASE_URL"
    # Candidate generation is sampled; all other LLM operations use greedy decoding.
    temperature: float = 0.0
    candidate_generation_temperature: float = 0.7
    max_tokens: int = 1024
    max_retries: int = 0
    require_usage: bool = True


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-base-en-v1.5"
    model_path_env: str = "AMRQA_EMBEDDING_MODEL_DIR"
    device: str = "auto"
    batch_size: int = 32


@dataclass
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    model_path_env: str = "AMRQA_RERANKER_MODEL_DIR"
    device: str = "auto"
    use_fp16: bool = True


@dataclass
class AMRConfig:
    reference_source: str = "amr"
    parser_model_dir_env: str = "AMRQA_STOG_MODEL_DIR"
    generator_model_dir_env: str = "AMRQA_GTOS_MODEL_DIR"
    max_paths: int = 8
    max_hops: int = 8


@dataclass
class PromptConfig:
    use_react_demonstrations: bool = True
    use_atomic_fact_demonstrations: bool = True
    atomic_fact_max_tokens: int = 1024


@dataclass
class ReasoningConfig:
    max_depth: int = 6
    candidate_count: int = 3
    candidate_history_mode: str = "full"
    # The manuscript uses one k for both local retention and the global active beam.
    path_width: int = 2
    amr_reference_weight: float = 0.5
    candidate_scoring_mode: str = "joint"
    retrieval_k: int = 5
    rerank_k: int = 2
    facts_per_observation: int = 3
    observation_mode: str = "atomic_facts"
    no_answer: str = "No-answer"


@dataclass
class RuntimeConfig:
    log_level: str = "INFO"
    timeout_seconds: float = 180.0
    capture_explored_paths: bool = True


@dataclass
class AMRQAConfig:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    amr: AMRConfig = field(default_factory=AMRConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        cfg = self.reasoning
        for name in ("temperature", "candidate_generation_temperature"):
            value = getattr(self.model, name)
            if not 0.0 <= value <= 2.0:
                raise ValueError(f"model.{name} must be in [0, 2]")
        if not 0.0 <= cfg.amr_reference_weight <= 1.0:
            raise ValueError("reasoning.amr_reference_weight must be in [0, 1]")
        for name in (
            "max_depth",
            "candidate_count",
            "path_width",
            "retrieval_k",
            "rerank_k",
            "facts_per_observation",
        ):
            if getattr(cfg, name) < 1:
                raise ValueError(f"reasoning.{name} must be at least 1")
        if cfg.rerank_k > cfg.retrieval_k:
            raise ValueError("reasoning.rerank_k cannot exceed reasoning.retrieval_k")
        if cfg.path_width > cfg.candidate_count:
            raise ValueError("reasoning.path_width cannot exceed reasoning.candidate_count")
        if cfg.candidate_history_mode not in {"full", "question_only"}:
            raise ValueError("reasoning.candidate_history_mode must be 'full' or 'question_only'")
        if cfg.candidate_scoring_mode not in {
            "joint",
            "amr_only",
            "semantic_only",
            "none",
        }:
            raise ValueError(
                "reasoning.candidate_scoring_mode must be joint, amr_only, semantic_only, or none"
            )
        if cfg.observation_mode not in {"atomic_facts", "sentences"}:
            raise ValueError("reasoning.observation_mode must be 'atomic_facts' or 'sentences'")
        if self.amr.max_paths < 1 or self.amr.max_hops < 1:
            raise ValueError("amr.max_paths and amr.max_hops must be at least 1")
        if self.amr.reference_source not in {"amr", "llm", "question"}:
            raise ValueError("amr.reference_source must be 'amr', 'llm', or 'question'")
        if self.model.max_retries != 0:
            raise ValueError("model.max_retries must be 0 for the paper evaluation protocol")
        if self.runtime.timeout_seconds <= 0:
            raise ValueError("runtime.timeout_seconds must be greater than 0")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _update_dataclass(instance: Any, values: dict[str, Any]) -> None:
    """Recursively apply a validated mapping to a dataclass instance."""
    fields = {field.name: field for field in dataclasses.fields(instance)}
    type_hints = get_type_hints(type(instance))
    unknown = values.keys() - fields.keys()
    if unknown:
        raise ValueError(f"Unknown configuration key(s): {', '.join(sorted(unknown))}")
    for key, value in values.items():
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Configuration section '{key}' must be a mapping")
            _update_dataclass(current, value)
            continue
        expected_type = type_hints.get(key)
        if expected_type is not None and expected_type is not Any:
            try:
                value = expected_type(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid value for '{key}': {value!r}") from exc
        setattr(instance, key, value)


def apply_overrides(config: AMRQAConfig, overrides: list[str]) -> AMRQAConfig:
    """Apply `section.key=value` command-line overrides."""
    for override in overrides:
        if "=" not in override or "." not in override.partition("=")[0]:
            raise ValueError(f"Expected section.key=value, received: {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        section_name, field_name = dotted_key.split(".", 1)
        section = getattr(config, section_name, None)
        if section is None or not dataclasses.is_dataclass(section):
            raise ValueError(f"Unknown configuration section: {section_name}")
        if field_name not in {field.name for field in dataclasses.fields(section)}:
            raise ValueError(f"Unknown configuration key: {dotted_key}")
        value = yaml.safe_load(raw_value)
        _update_dataclass(section, {field_name: value})
    config.validate()
    return config


def load_config(
    path: Union[str, pathlib.Path], overrides: Optional[list[str]] = None
) -> AMRQAConfig:
    """Load a YAML experiment configuration and validate all values."""
    config_path = pathlib.Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("The configuration root must be a YAML mapping")
    config = AMRQAConfig()
    _update_dataclass(config, raw)
    config.validate()
    return apply_overrides(config, overrides or [])
