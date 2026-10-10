"""Official-style normalized EM and token F1 metrics for QA."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Union

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = Counter(prediction_tokens) & Counter(reference_tokens)
    matches = sum(overlap.values())
    if not matches:
        return 0.0
    precision = matches / len(prediction_tokens)
    recall = matches / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def summarize(pairs: Iterable[tuple[str, str]]) -> dict[str, Union[float, int]]:
    values = list(pairs)
    if not values:
        return {"count": 0, "exact_match": 0.0, "f1": 0.0}
    em = sum(exact_match(prediction, reference) for prediction, reference in values) / len(values)
    f1 = sum(token_f1(prediction, reference) for prediction, reference in values) / len(values)
    return {"count": len(values), "exact_match": 100 * em, "f1": 100 * f1}


def summarize_efficiency(path: Union[str, Path]) -> dict[str, Union[float, int]]:
    """Macro-average per-question resource fields stored in inference traces."""
    metadata: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                values = item["metadata"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid prediction at {path}:{line_number}") from exc
            if not isinstance(values, dict):
                raise ValueError(f"Prediction metadata is not an object at {path}:{line_number}")
            metadata.append(values)
    if not metadata:
        raise ValueError("The prediction file contains no records")

    required = (
        "elapsed_seconds",
        "reasoning_steps",
        "input_tokens",
        "output_tokens",
        "llm_calls",
        "retrieval_calls",
        "timed_out",
        "max_step_termination",
    )
    missing = [name for name in required if any(name not in item for item in metadata)]
    if missing:
        raise ValueError(f"Trace metadata is missing efficiency fields: {', '.join(missing)}")
    count = len(metadata)

    def mean(name: str) -> float:
        return sum(float(item[name]) for item in metadata) / count

    return {
        "count": count,
        "latency_seconds_per_question": mean("elapsed_seconds"),
        "reasoning_steps_per_question": mean("reasoning_steps"),
        "input_tokens_per_question": mean("input_tokens"),
        "output_tokens_per_question": mean("output_tokens"),
        "llm_calls_per_question": mean("llm_calls"),
        "retrieval_calls_per_question": mean("retrieval_calls"),
        "timeout_rate_percent": 100 * mean("timed_out"),
        "max_step_termination_rate_percent": 100 * mean("max_step_termination"),
    }
