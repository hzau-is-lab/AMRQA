"""Robust extraction and parsing of atomic facts from retrieved passages."""

from __future__ import annotations

import json
import re
from typing import Optional

from .llm import ChatClient, Message
from .prompts import SYSTEM_PROMPT, atomic_fact_prompt

_NUMBERED_ITEM = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$")


def _extract_json_array(text: str) -> Optional[list[str]]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return [item.strip() for item in value if item.strip()]


def parse_facts(text: str) -> list[str]:
    parsed = _extract_json_array(text)
    if parsed is not None:
        return list(dict.fromkeys(parsed))
    facts: list[str] = []
    for line in text.splitlines():
        match = _NUMBERED_ITEM.match(line)
        fact = match.group(1).strip() if match else ""
        if fact:
            facts.append(fact)
    return list(dict.fromkeys(facts))


class AtomicFactExtractor:
    def __init__(
        self, client: ChatClient, temperature: float, max_tokens: int, demonstrations: str = ""
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.demonstrations = demonstrations

    def extract(self, context: str, timeout_seconds: Optional[float] = None) -> list[str]:
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": atomic_fact_prompt(context, self.demonstrations)},
        ]
        return parse_facts(
            self.client.complete(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout_seconds=timeout_seconds,
            )
        )
