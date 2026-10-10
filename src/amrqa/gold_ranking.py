"""Gold-decomposition ranking diagnostics used in the manuscript analysis."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Union

SCORE_FIELDS = (
    "amr_reference_similarity",
    "semantic_relevance",
    "joint_score",
)


def _safe_ratio(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _read_states(path: Union[str, Path]) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                state = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}") from exc
            candidates = state.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"Line {line_number} must contain a non-empty candidate list")
            for candidate in candidates:
                if not isinstance(candidate, dict) or not isinstance(
                    candidate.get("gold_aligned"), bool
                ):
                    raise ValueError(
                        f"Every candidate at line {line_number} requires boolean gold_aligned"
                    )
                missing = [field for field in SCORE_FIELDS if field not in candidate]
                if missing:
                    raise ValueError(
                        f"Candidate at line {line_number} is missing: {', '.join(missing)}"
                    )
                for field in SCORE_FIELDS:
                    try:
                        candidate[field] = float(candidate[field])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Candidate field {field} at line {line_number} must be numeric"
                        ) from exc
            states.append(state)
    if not states:
        raise ValueError("The annotation file contains no reasoning states")
    return states


def evaluate_gold_ranking(path: Union[str, Path]) -> dict[str, Any]:
    """Compute availability, ranking, and pairwise metrics from one annotated JSONL file.

    Hit and reciprocal-rank metrics use stable score ordering with candidate order as the tie
    break. Pairwise accuracy evaluates every eligible aligned/non-aligned pair, assigning half a
    point to an exact score tie.
    """

    states = _read_states(path)
    available = [state for state in states if any(c["gold_aligned"] for c in state["candidates"])]
    candidate_counts = {len(state["candidates"]) for state in states}
    if len(candidate_counts) != 1:
        raise ValueError("All states must contain the same number of generated candidates")
    candidates_per_state = next(iter(candidate_counts))
    multiplicity = Counter(
        sum(bool(candidate["gold_aligned"]) for candidate in state["candidates"])
        for state in available
    )

    score_results: dict[str, dict[str, Union[float, int]]] = {}
    for field in SCORE_FIELDS:
        hit_at_1 = 0
        hit_at_2 = 0
        reciprocal_rank = 0.0
        correct = 0
        ties = 0
        incorrect = 0
        for state in available:
            candidates = state["candidates"]
            ranked = sorted(
                enumerate(candidates),
                key=lambda item: (-item[1][field], item[0]),
            )
            first_gold_rank = next(
                rank
                for rank, (_, candidate) in enumerate(ranked, start=1)
                if candidate["gold_aligned"]
            )
            hit_at_1 += int(first_gold_rank <= 1)
            hit_at_2 += int(first_gold_rank <= 2)
            reciprocal_rank += 1.0 / first_gold_rank

            aligned = [candidate for candidate in candidates if candidate["gold_aligned"]]
            non_aligned = [candidate for candidate in candidates if not candidate["gold_aligned"]]
            for positive in aligned:
                for negative in non_aligned:
                    difference = positive[field] - negative[field]
                    if abs(difference) <= 1e-12:
                        ties += 1
                    elif difference > 0:
                        correct += 1
                    else:
                        incorrect += 1

        state_denominator = len(available)
        pair_denominator = correct + ties + incorrect
        score_results[field] = {
            "conditional_hit_at_1": _safe_ratio(hit_at_1, state_denominator),
            "conditional_hit_at_2": _safe_ratio(hit_at_2, state_denominator),
            "conditional_mrr": _safe_ratio(reciprocal_rank, state_denominator),
            "pairwise_correct": correct,
            "pairwise_ties": ties,
            "pairwise_incorrect": incorrect,
            "eligible_pair_count": pair_denominator,
            "conditional_pairwise_accuracy": _safe_ratio(correct + 0.5 * ties, pair_denominator),
        }

    return {
        "state_count": len(states),
        "candidates_per_state": candidates_per_state,
        "gold_path_candidate_count": len(available),
        "gold_path_candidate_availability": _safe_ratio(len(available), len(states)),
        "gold_aligned_candidate_multiplicity": {
            str(count): frequency for count, frequency in sorted(multiplicity.items())
        },
        "scores": score_results,
    }
