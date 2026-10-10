"""Supporting-evidence metrics over benchmark-native evidence identifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from .datasets import iter_jsonl


def _selected_unit_ids(record: dict[str, Any], representation: str) -> set[str]:
    selected_path = record.get("selected_path")
    if not isinstance(selected_path, dict):
        return set()
    selected: set[str] = set()
    for step in selected_path.get("steps", []):
        if not isinstance(step, dict):
            continue
        items = step.get("facts", []) if representation == "facts" else step.get("evidence", [])
        key = "source_unit_ids" if representation == "facts" else "native_unit_ids"
        for item in items:
            if isinstance(item, dict):
                selected.update(str(value) for value in item.get(key, []) if str(value))
    return selected


def _prf(selected: set[str], gold: set[str]) -> tuple[float, float, float]:
    overlap = len(selected & gold)
    precision = overlap / len(selected) if selected else float(not gold)
    recall = overlap / len(gold) if gold else float(not selected)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_supporting_evidence(
    predictions_path: Union[str, Path],
    data_path: Union[str, Path],
    representation: str = "facts",
) -> dict[str, Union[float, int, str]]:
    """Compute per-question supporting precision, recall, and F1, then macro-average."""
    if representation not in {"facts", "passages"}:
        raise ValueError("representation must be 'facts' or 'passages'")

    predictions: dict[str, dict[str, Any]] = {}
    with Path(predictions_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                identifier = str(record["id"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid prediction at {predictions_path}:{line_number}") from exc
            if identifier in predictions:
                raise ValueError(f"Duplicate prediction id: {identifier}")
            predictions[identifier] = record

    values: list[tuple[float, float, float]] = []
    for example in iter_jsonl(data_path):
        gold = set(str(item) for item in example.metadata.get("gold_evidence_unit_ids", []))
        if not gold:
            continue
        if example.id not in predictions:
            raise ValueError(f"Missing prediction for evidence-annotated example: {example.id}")
        selected = _selected_unit_ids(predictions[example.id], representation)
        values.append(_prf(selected, gold))

    if not values:
        raise ValueError("No examples with normalized supporting-evidence annotations were found")
    count = len(values)
    return {
        "count": count,
        "representation": representation,
        "precision": 100 * sum(value[0] for value in values) / count,
        "recall": 100 * sum(value[1] for value in values) / count,
        "f1": 100 * sum(value[2] for value in values) / count,
    }
