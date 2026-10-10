"""Question-clustered paired inference for the manuscript's primary comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml

from .metrics import exact_match, token_f1


@dataclass(frozen=True)
class Comparison:
    label: str
    system: tuple[str, ...]
    baseline: tuple[str, ...]


def _load_seed_predictions(path: str) -> dict[str, tuple[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                identifier = str(item["id"])
                prediction = str(item["prediction"])
                gold = item["gold_answer"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid prediction at {path}:{line_number}") from exc
            if gold is None:
                raise ValueError(f"Prediction {identifier} in {path} has no gold answer")
            if identifier in records:
                raise ValueError(f"Duplicate prediction id {identifier} in {path}")
            records[identifier] = (prediction, str(gold))
    return records


def _seed_metric(path: str, metric: str) -> tuple[dict[str, float], dict[str, str]]:
    scorer = exact_match if metric == "em" else token_f1
    records = _load_seed_predictions(path)
    scores = {
        identifier: scorer(prediction, gold) for identifier, (prediction, gold) in records.items()
    }
    gold = {identifier: value[1] for identifier, value in records.items()}
    return scores, gold


def _matched_differences(
    comparison: Comparison, metric: str
) -> tuple[np.ndarray, list[float], list[float]]:
    if len(comparison.system) != len(comparison.baseline):
        raise ValueError(f"{comparison.label}: system and baseline must have matched seed counts")
    if len(comparison.system) != 3:
        raise ValueError(f"{comparison.label}: the paper protocol requires three matched seeds")
    system_loaded = [_seed_metric(path, metric) for path in comparison.system]
    baseline_loaded = [_seed_metric(path, metric) for path in comparison.baseline]
    system_seeds = [item[0] for item in system_loaded]
    baseline_seeds = [item[0] for item in baseline_loaded]
    gold_maps = [item[1] for item in (*system_loaded, *baseline_loaded)]
    identifiers = set(system_seeds[0])
    all_runs = [*system_seeds, *baseline_seeds]
    if any(set(run) != identifiers for run in all_runs):
        raise ValueError(f"{comparison.label}: question ids differ across matched runs")
    first_gold = gold_maps[0]
    if any(gold != first_gold for gold in gold_maps[1:]):
        raise ValueError(f"{comparison.label}: gold answers differ across matched runs")
    ordered = sorted(identifiers)
    system_matrix = np.asarray(
        [[run[identifier] for identifier in ordered] for run in system_seeds], dtype=float
    )
    baseline_matrix = np.asarray(
        [[run[identifier] for identifier in ordered] for run in baseline_seeds], dtype=float
    )
    differences = 100 * (system_matrix.mean(axis=0) - baseline_matrix.mean(axis=0))
    system_seed_means = (100 * system_matrix.mean(axis=1)).tolist()
    baseline_seed_means = (100 * baseline_matrix.mean(axis=1)).tolist()
    return differences, system_seed_means, baseline_seed_means


def _bootstrap_ci(
    differences: np.ndarray, resamples: int, rng: np.random.Generator, batch_size: int = 256
) -> tuple[float, float]:
    estimates: list[np.ndarray] = []
    for start in range(0, resamples, batch_size):
        size = min(batch_size, resamples - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        estimates.append(differences[indices].mean(axis=1))
    values = np.concatenate(estimates)
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def _randomization_p(
    differences: np.ndarray, trials: int, rng: np.random.Generator, batch_size: int = 256
) -> float:
    observed = abs(float(differences.mean()))
    extreme = 0
    for start in range(0, trials, batch_size):
        size = min(batch_size, trials - start)
        signs = rng.integers(0, 2, size=(size, len(differences)), dtype=np.int8) * 2 - 1
        randomized = np.abs((signs * differences).mean(axis=1))
        extreme += int(np.count_nonzero(randomized >= observed - 1e-12))
    return (extreme + 1) / (trials + 1)


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _parse_manifest(path: Union[str, Path]) -> tuple[dict[str, Any], list[Comparison]]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    items = raw.get("comparisons")
    grid = raw.get("comparison_grid")
    if items is None and isinstance(grid, dict):
        backbones = grid.get("backbones", [])
        datasets = grid.get("datasets", [])
        baselines = grid.get("baselines", [])
        seeds = grid.get("seeds", [])
        pattern = str(grid.get("trace_pattern", ""))
        system_method = str(grid.get("system_method", "amrqa"))
        dimensions = (backbones, datasets, baselines)
        if not all(isinstance(values, list) and values for values in dimensions):
            raise ValueError(
                "comparison_grid requires non-empty backbone, dataset, and baseline lists"
            )
        if len(seeds) != 3 or not pattern:
            raise ValueError("comparison_grid requires three seeds and a trace_pattern")
        items = []
        for backbone in backbones:
            for dataset in datasets:
                for baseline in baselines:
                    common = {"backbone": backbone, "dataset": dataset}
                    items.append(
                        {
                            "label": f"{backbone}-{dataset}-amrqa-vs-{baseline}",
                            "system": [
                                pattern.format(method=system_method, seed=seed, **common)
                                for seed in seeds
                            ],
                            "baseline": [
                                pattern.format(method=baseline, seed=seed, **common)
                                for seed in seeds
                            ],
                        }
                    )
    if not isinstance(items, list) or not items:
        raise ValueError(
            "The statistics manifest requires comparisons or a non-empty comparison_grid"
        )
    comparisons = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each comparison must be a mapping")
        comparisons.append(
            Comparison(
                label=str(item["label"]),
                system=tuple(str(value) for value in item["system"]),
                baseline=tuple(str(value) for value in item["baseline"]),
            )
        )
    return raw, comparisons


def run_comparison_manifest(path: Union[str, Path]) -> dict[str, Any]:
    """Run all comparisons and apply one Holm correction across the declared family."""
    manifest, comparisons = _parse_manifest(path)
    bootstrap_resamples = int(manifest.get("bootstrap_resamples", 10_000))
    randomization_trials = int(manifest.get("randomization_trials", 100_000))
    analysis_seed = int(manifest.get("analysis_seed", 42))
    metrics = tuple(str(value).lower() for value in manifest.get("metrics", ["em", "f1"]))
    if any(metric not in {"em", "f1"} for metric in metrics):
        raise ValueError("metrics must contain only 'em' and/or 'f1'")
    if bootstrap_resamples < 1 or randomization_trials < 1:
        raise ValueError("bootstrap_resamples and randomization_trials must be positive")

    results: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    rng = np.random.default_rng(analysis_seed)
    for comparison in comparisons:
        for metric in metrics:
            differences, system_seed_means, baseline_seed_means = _matched_differences(
                comparison, metric
            )
            confidence_interval = _bootstrap_ci(differences, bootstrap_resamples, rng)
            p_value = _randomization_p(differences, randomization_trials, rng)
            raw_p_values.append(p_value)
            results.append(
                {
                    "comparison": comparison.label,
                    "metric": metric,
                    "question_count": len(differences),
                    "system_mean": float(np.mean(system_seed_means)),
                    "system_std": float(np.std(system_seed_means, ddof=1)),
                    "baseline_mean": float(np.mean(baseline_seed_means)),
                    "baseline_std": float(np.std(baseline_seed_means, ddof=1)),
                    "difference": float(differences.mean()),
                    "pointwise_95_ci": list(confidence_interval),
                    "randomization_p": p_value,
                }
            )
    adjusted_values = _holm_adjust(raw_p_values)
    if len(results) != len(adjusted_values):
        raise RuntimeError("Internal error while applying Holm correction")
    for result, adjusted in zip(results, adjusted_values):
        result["holm_adjusted_p"] = adjusted
    return {
        "bootstrap_resamples": bootstrap_resamples,
        "randomization_trials": randomization_trials,
        "analysis_seed": analysis_seed,
        "family_size": len(results),
        "results": results,
    }
