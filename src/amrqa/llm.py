"""Provider-independent chat-completion clients."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol

from .config import ModelConfig

Message = dict[str, str]


@dataclass(frozen=True)
class LLMUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __sub__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            calls=self.calls - other.calls,
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
        )


class ChatClient(Protocol):
    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        """Generate exactly one chat completion."""

    def usage_snapshot(self) -> LLMUsage:
        """Return cumulative calls and provider-reported token counts."""


class OpenAICompatibleClient:
    """Client for OpenAI and OpenAI-compatible inference servers."""

    def __init__(self, config: ModelConfig, seed: int) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install the LLM dependency with: pip install -e '.[llm]'") from exc

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {config.api_key_env} is required")
        base_url = os.getenv(config.base_url_env) or None
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=config.max_retries,
        )
        self.model_name = config.model_name
        self.seed = seed
        self.require_usage = config.require_usage
        self._usage = LLMUsage()

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        self._usage = LLMUsage(
            calls=self._usage.calls + 1,
            input_tokens=self._usage.input_tokens,
            output_tokens=self._usage.output_tokens,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=self.seed,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            if exc.__class__.__name__ == "APITimeoutError":
                raise TimeoutError(str(exc)) from exc
            raise
        usage = response.usage
        if usage is None and self.require_usage:
            raise RuntimeError(
                "The inference endpoint omitted token usage required by the paper protocol"
            )
        self._usage = LLMUsage(
            calls=self._usage.calls,
            input_tokens=self._usage.input_tokens + int(usage.prompt_tokens if usage else 0),
            output_tokens=self._usage.output_tokens + int(usage.completion_tokens if usage else 0),
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("The LLM returned an empty completion")
        return content.strip()

    def usage_snapshot(self) -> LLMUsage:
        return self._usage


class DryRunClient:
    """Explicitly non-semantic backend used only to validate I/O and trace serialization."""

    def __init__(self) -> None:
        self._calls = 0

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        self._calls += 1
        request = messages[-1]["content"]
        if "candidate sub-questions" in request:
            return '["What information is needed to answer the question?"]'
        if "atomic facts" in request:
            return "[]"
        if "sufficient evidence" in request:
            return '{"sufficient": false, "thought": "Dry-run mode does not answer questions."}'
        if "final answer" in request:
            return "No-answer"
        if "semantic relevance" in request:
            return "0.500"
        return ""

    def usage_snapshot(self) -> LLMUsage:
        # The dry-run backend has no tokenizer. Calls are exact; token counts are unavailable.
        return LLMUsage(calls=self._calls)


def build_chat_client(config: ModelConfig, seed: int = 42) -> ChatClient:
    if config.provider == "openai_compatible":
        return OpenAICompatibleClient(config, seed)
    if config.provider == "dry_run":
        return DryRunClient()
    raise ValueError(f"Unsupported model provider: {config.provider}")
