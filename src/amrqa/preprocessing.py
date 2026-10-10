"""Validation and JSONL conversion for official benchmark exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Union

from .datasets import normalize_example


def _read_records(path: Union[str, Path]) -> list[dict[str, Any]]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_path}:{line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object at {input_path}:{line_number}") from None
            records.append(item)
        return records
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("A JSON input must contain a list of example objects")
    return value


def prepare_jsonl(input_path: Union[str, Path], output_path: Union[str, Path]) -> dict[str, Any]:
    """Validate an official JSON/JSONL export and retain its source fields in JSONL form."""
    records = _read_records(input_path)
    for index, record in enumerate(records, start=1):
        try:
            normalize_example(record, index)
        except ValueError as exc:
            raise ValueError(f"Invalid benchmark example {index}: {exc}") from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {"output": str(output), "count": len(records), "sha256": digest}
