import json
from pathlib import Path

import yaml

from amrqa.gold_ranking import evaluate_gold_ranking
from amrqa.metrics import summarize_efficiency
from amrqa.statistics import _parse_manifest, run_comparison_manifest
from amrqa.supporting import evaluate_supporting_evidence


def _write_predictions(path, predictions) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for identifier, prediction, gold in predictions:
            handle.write(
                json.dumps({"id": identifier, "prediction": prediction, "gold_answer": gold}) + "\n"
            )


def test_question_clustered_statistics_and_holm_family(tmp_path) -> None:
    system_paths = []
    baseline_paths = []
    for seed in (1, 2, 3):
        system = tmp_path / f"system-{seed}.jsonl"
        baseline = tmp_path / f"baseline-{seed}.jsonl"
        _write_predictions(system, [("q1", "Ada", "Ada"), ("q2", "Byron", "Byron")])
        _write_predictions(baseline, [("q1", "wrong", "Ada"), ("q2", "Byron", "Byron")])
        system_paths.append(str(system))
        baseline_paths.append(str(baseline))
    manifest = tmp_path / "statistics.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "bootstrap_resamples": 20,
                "randomization_trials": 40,
                "analysis_seed": 7,
                "metrics": ["em", "f1"],
                "comparisons": [
                    {
                        "label": "system-vs-baseline",
                        "system": system_paths,
                        "baseline": baseline_paths,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = run_comparison_manifest(manifest)
    assert result["family_size"] == 2
    assert all(item["difference"] == 50.0 for item in result["results"])
    assert all("holm_adjusted_p" in item for item in result["results"])


def test_statistics_grid_predeclares_complete_primary_family() -> None:
    root = Path(__file__).resolve().parents[1]
    _, comparisons = _parse_manifest(root / "configs" / "statistics.example.yaml")
    assert len(comparisons) == 27
    assert comparisons[0].system[0].endswith("qwen_hotpot_amrqa_seed42.jsonl")


def test_supporting_evidence_uses_native_fact_provenance(tmp_path) -> None:
    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Question?",
                "answer": "Answer",
                "context": [["Doc", ["Supporting sentence.", "Distractor."]]],
                "supporting_facts": [["Doc", 0]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "id": "q1",
                "selected_path": {
                    "steps": [
                        {
                            "facts": [{"source_unit_ids": ['["Doc",0]']}],
                            "evidence": [],
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = evaluate_supporting_evidence(predictions, data)
    assert result["precision"] == 100.0
    assert result["recall"] == 100.0
    assert result["f1"] == 100.0


def test_efficiency_summary_uses_per_question_trace_metadata(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "metadata": {
                        "elapsed_seconds": latency,
                        "reasoning_steps": 2,
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "llm_calls": 4,
                        "retrieval_calls": 2,
                        "timed_out": timeout,
                        "max_step_termination": False,
                    }
                }
            )
            for latency, timeout in ((2.0, False), (4.0, True))
        )
        + "\n",
        encoding="utf-8",
    )
    result = summarize_efficiency(path)
    assert result["latency_seconds_per_question"] == 3.0
    assert result["timeout_rate_percent"] == 50.0


def test_gold_ranking_uses_all_eligible_pairs_and_half_credit_for_ties(tmp_path) -> None:
    path = tmp_path / "gold-ranking.jsonl"
    rows = [
        {
            "state_id": "s1",
            "candidates": [
                {
                    "gold_aligned": True,
                    "amr_reference_similarity": 0.9,
                    "semantic_relevance": 0.6,
                    "joint_score": 0.8,
                },
                {
                    "gold_aligned": False,
                    "amr_reference_similarity": 0.4,
                    "semantic_relevance": 0.6,
                    "joint_score": 0.5,
                },
                {
                    "gold_aligned": False,
                    "amr_reference_similarity": 0.2,
                    "semantic_relevance": 0.3,
                    "joint_score": 0.4,
                },
            ],
        },
        {
            "state_id": "s2",
            "candidates": [
                {
                    "gold_aligned": True,
                    "amr_reference_similarity": 0.4,
                    "semantic_relevance": 0.7,
                    "joint_score": 0.6,
                },
                {
                    "gold_aligned": True,
                    "amr_reference_similarity": 0.8,
                    "semantic_relevance": 0.8,
                    "joint_score": 0.8,
                },
                {
                    "gold_aligned": False,
                    "amr_reference_similarity": 0.6,
                    "semantic_relevance": 0.5,
                    "joint_score": 0.5,
                },
            ],
        },
        {
            "state_id": "s3",
            "candidates": [
                {
                    "gold_aligned": False,
                    "amr_reference_similarity": 0.5,
                    "semantic_relevance": 0.5,
                    "joint_score": 0.5,
                },
                {
                    "gold_aligned": False,
                    "amr_reference_similarity": 0.4,
                    "semantic_relevance": 0.4,
                    "joint_score": 0.4,
                },
                {
                    "gold_aligned": False,
                    "amr_reference_similarity": 0.3,
                    "semantic_relevance": 0.3,
                    "joint_score": 0.3,
                },
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    result = evaluate_gold_ranking(path)
    assert result["state_count"] == 3
    assert result["gold_path_candidate_count"] == 2
    assert result["gold_aligned_candidate_multiplicity"] == {"1": 1, "2": 1}
    amr = result["scores"]["amr_reference_similarity"]
    semantic = result["scores"]["semantic_relevance"]
    assert amr["eligible_pair_count"] == 4
    assert amr["conditional_hit_at_1"] == 1.0
    assert semantic["pairwise_ties"] == 1
    assert semantic["conditional_pairwise_accuracy"] == 0.875
