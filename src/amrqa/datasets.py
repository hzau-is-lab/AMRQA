"""Dataset normalization and JSONL input/output helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Optional, Union

from .domain import Document, EvidenceUnit, QAExample


def _sentences_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return ""


def _native_unit_id(title: str, position: Union[int, str]) -> str:
    """Create a stable, unambiguous identifier for benchmark-native evidence units."""
    return json.dumps([title, position], ensure_ascii=False, separators=(",", ":"))


def _units_from_sentences(title: str, sentences: list[Any]) -> tuple[EvidenceUnit, ...]:
    units = []
    for position, sentence in enumerate(sentences):
        text = str(sentence).strip()
        if text:
            units.append(EvidenceUnit(_native_unit_id(title, position), text))
    return tuple(units)


def _gold_evidence_unit_ids(record: dict[str, Any]) -> list[str]:
    """Normalize HotpotQA/2Wiki and MuSiQue supporting-evidence annotations."""
    normalized: list[str] = []
    supporting_facts = record.get("supporting_facts", record.get("supporting_evidence", []))
    if isinstance(supporting_facts, list):
        for item in supporting_facts:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                normalized.append(_native_unit_id(str(item[0]), item[1]))
            elif isinstance(item, dict):
                title = str(item.get("title", ""))
                position = item.get("sent_id", item.get("sentence_id", item.get("idx")))
                if position is not None:
                    normalized.append(_native_unit_id(title, position))

    paragraphs = record.get("paragraphs")
    if isinstance(paragraphs, list):
        for index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict) or not paragraph.get("is_supporting", False):
                continue
            title = str(paragraph.get("title", ""))
            position = paragraph.get("idx", paragraph.get("id", index))
            normalized.append(_native_unit_id(title, position))
    return list(dict.fromkeys(normalized))


def _normalize_documents(record: dict[str, Any]) -> tuple[Document, ...]:
    """Accept the context formats used by HotpotQA, 2WikiQA, and MuSiQue exports."""
    documents: list[Document] = []
    if isinstance(record.get("flat_contexts"), list):
        for index, text in enumerate(record["flat_contexts"]):
            normalized = _sentences_to_text(text)
            if normalized:
                units = (EvidenceUnit(_native_unit_id("", index), normalized),)
                documents.append(
                    Document(id=f"context-{index}", text=normalized, native_units=units)
                )
    else:
        contexts = record.get("context", record.get("contexts", record.get("paragraphs", [])))
        if not isinstance(contexts, list):
            raise ValueError("context/contexts/paragraphs must be a list")
        for index, item in enumerate(contexts):
            title = ""
            text = ""
            units: tuple[EvidenceUnit, ...] = ()
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                raw_text = item[1]
                text = _sentences_to_text(raw_text)
                if isinstance(raw_text, list):
                    units = _units_from_sentences(title, raw_text)
            elif isinstance(item, dict):
                title = str(item.get("title", ""))
                raw_text = item.get("text", item.get("sentences", item.get("paragraph_text", "")))
                text = _sentences_to_text(raw_text)
                if isinstance(raw_text, list):
                    units = _units_from_sentences(title, raw_text)
                else:
                    native_position = item.get("idx", item.get("id", index))
                    if text:
                        units = (EvidenceUnit(_native_unit_id(title, native_position), text),)
            elif isinstance(item, str):
                text = item.strip()
                if text:
                    units = (EvidenceUnit(_native_unit_id("", index), text),)
            if text:
                if not units:
                    units = (EvidenceUnit(_native_unit_id(title, index), text),)
                documents.append(
                    Document(
                        id=f"context-{index}",
                        title=title,
                        text=text,
                        native_units=units,
                    )
                )
    if not documents:
        raise ValueError("Example contains no non-empty retrieval contexts")
    return tuple(documents)


def normalize_example(record: dict[str, Any], index: int) -> QAExample:
    question = str(record.get("question", "")).strip()
    if not question:
        raise ValueError("Example is missing a non-empty 'question'")
    answer = record.get("answer")
    answer = str(answer).strip() if answer is not None else None
    identifier = str(record.get("id", record.get("_id", f"example-{index}")))
    metadata = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "id",
            "_id",
            "question",
            "answer",
            "context",
            "contexts",
            "paragraphs",
            "flat_contexts",
        }
    }
    gold_evidence_unit_ids = _gold_evidence_unit_ids(record)
    if gold_evidence_unit_ids:
        metadata["gold_evidence_unit_ids"] = gold_evidence_unit_ids
    return QAExample(
        id=identifier,
        question=question,
        answer=answer,
        documents=_normalize_documents(record),
        metadata=metadata,
    )


def iter_jsonl(path: Union[str, Path], limit: Optional[int] = None) -> Iterator[QAExample]:
    """Yield validated examples with line-aware errors."""
    input_path = Path(path)
    yielded = 0
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("JSONL line must contain an object")
                yield normalize_example(raw, line_number)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid example at {input_path}:{line_number}: {exc}") from exc
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def write_jsonl(path: Union[str, Path], records: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
