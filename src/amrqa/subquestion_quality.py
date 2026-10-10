"""Sub-question fluency, semantic-similarity, and form diagnostics."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Union

import numpy as np

_QUESTION_OPENERS = {
    "are",
    "how",
    "is",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
}
_FIRST_TOKEN = re.compile(r"[A-Za-z]+")


def rule_based_well_formed(question: str) -> int:
    """Return the binary syntactic-form label used by the released analysis."""

    match = _FIRST_TOKEN.search(question.strip())
    return int(
        bool(match)
        and match.group(0).lower() in _QUESTION_OPENERS
        and question.rstrip().endswith("?")
    )


def _subquestions(record: dict[str, Any]) -> list[str]:
    explicit = record.get("sub_questions")
    if isinstance(explicit, list):
        return [str(value).strip() for value in explicit if str(value).strip()]
    selected = record.get("selected_path")
    if isinstance(selected, dict) and isinstance(selected.get("steps"), list):
        return [
            str(step.get("sub_question", "")).strip()
            for step in selected["steps"]
            if isinstance(step, dict) and str(step.get("sub_question", "")).strip()
        ]
    legacy = record.get("all_path")
    if isinstance(legacy, dict):
        return [str(value).strip() for value in legacy.values() if str(value).strip()]
    return []


def evaluate_subquestion_quality(
    path: Union[str, Path],
    *,
    gpt2_model_name: str = "gpt2",
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "auto",
) -> dict[str, Union[float, int]]:
    """Evaluate explicit sub-questions against their original root questions."""

    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional model dependencies
        raise RuntimeError("Sub-question evaluation requires the 'models' dependencies") from exc

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
    language_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name).to(resolved_device)
    language_model.eval()
    semantic_model = SentenceTransformer(semantic_model_name, device=resolved_device)

    fluent: list[float] = []
    similarities: list[float] = []
    well_formed: list[int] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}") from exc
            original_question = str(
                record.get("question", record.get("original_question", ""))
            ).strip()
            if not original_question:
                raise ValueError(f"Line {line_number} has no original question")
            for subquestion in _subquestions(record):
                encoded = tokenizer(subquestion, return_tensors="pt").to(resolved_device)
                with torch.no_grad():
                    loss = language_model(**encoded, labels=encoded["input_ids"]).loss
                fluent.append(float(loss.item()))
                vectors = semantic_model.encode(
                    [subquestion, original_question],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                similarities.append(float(np.dot(vectors[0], vectors[1])))
                well_formed.append(rule_based_well_formed(subquestion))
    if not fluent:
        raise ValueError("The input contains no explicit sub-questions")
    return {
        "subquestion_count": len(fluent),
        "fluency_nll": float(np.mean(fluent)),
        "semantic_similarity": float(np.mean(similarities)),
        "well_formedness": float(np.mean(well_formed)),
    }
