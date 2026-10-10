"""Command-line interface for AMRQA inference and evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from .config import load_config
from .datasets import write_jsonl
from .gold_ranking import evaluate_gold_ranking
from .metrics import summarize, summarize_efficiency
from .preprocessing import prepare_jsonl
from .provenance import write_run_manifest
from .runner import run_predictions
from .statistics import run_comparison_manifest
from .subquestion_quality import evaluate_subquestion_quality
from .supporting import evaluate_supporting_evidence


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()), format="%(asctime)s | %(levelname)s | %(message)s"
    )


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set)
    if args.seed is not None:
        config.seed = args.seed
        config.validate()
    _configure_logging(config.runtime.log_level)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    pairs: list[tuple[str, str]] = []
    processed_count = 0

    def records() -> Any:
        nonlocal processed_count
        for record in run_predictions(config, args.data, args.limit):
            processed_count += 1
            gold = record.get("gold_answer")
            if gold is not None:
                pairs.append((str(record["prediction"]), str(gold)))
            yield record

    # Stream traces to disk instead of keeping a full benchmark's nested paths in memory.
    write_jsonl(output_path, records())
    metrics = summarize(pairs)
    manifest_path = write_run_manifest(
        config=config,
        config_path=args.config,
        data_path=args.data,
        output_path=output_path,
        evaluated_count=processed_count,
        limit=args.limit,
    )
    summary: dict[str, Any] = {
        "output": str(output_path),
        "manifest": str(manifest_path),
        **metrics,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    pairs: list[tuple[str, str]] = []
    with Path(args.predictions).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                prediction, gold = record["prediction"], record["gold_answer"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid prediction at line {line_number}") from exc
            if gold is not None:
                pairs.append((str(prediction), str(gold)))
    print(json.dumps(summarize(pairs), indent=2))
    return 0


def _support_evaluate(args: argparse.Namespace) -> int:
    result = evaluate_supporting_evidence(
        args.predictions,
        args.data,
        representation=args.representation,
    )
    print(json.dumps(result, indent=2))
    return 0


def _compare(args: argparse.Namespace) -> int:
    result = run_comparison_manifest(args.manifest)
    output = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output)
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


def _efficiency(args: argparse.Namespace) -> int:
    print(json.dumps(summarize_efficiency(args.predictions), indent=2))
    return 0


def _prepare(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    print(json.dumps(prepare_jsonl(args.input, output_path), indent=2))
    return 0


def _gold_ranking_evaluate(args: argparse.Namespace) -> int:
    print(json.dumps(evaluate_gold_ranking(args.annotations), indent=2))
    return 0


def _subquestion_evaluate(args: argparse.Namespace) -> int:
    result = evaluate_subquestion_quality(
        args.predictions,
        gpt2_model_name=args.gpt2_model,
        semantic_model_name=args.semantic_model,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AMR-guided multi-path reasoning for multi-hop QA")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run inference and write JSONL traces")
    run.add_argument("--config", required=True, help="YAML configuration file")
    run.add_argument("--data", required=True, help="Input JSONL dataset")
    run.add_argument("--output", required=True, help="Output JSONL trace file")
    run.add_argument("--limit", type=int, default=None, help="Optional number of examples")
    run.add_argument("--seed", type=int, default=None, help="Override the configured run seed")
    run.add_argument(
        "--set", action="append", default=[], help="Override, e.g. model.model_name=gpt-4o-mini"
    )
    run.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output")
    run.set_defaults(handler=_run)
    evaluate = commands.add_parser("evaluate", help="recompute EM and F1 from JSONL predictions")
    evaluate.add_argument("--predictions", required=True, help="Prediction JSONL file")
    evaluate.set_defaults(handler=_evaluate)
    support = commands.add_parser(
        "support-evaluate", help="evaluate selected evidence against native supporting units"
    )
    support.add_argument("--predictions", required=True, help="Prediction JSONL file")
    support.add_argument("--data", required=True, help="Original benchmark JSONL")
    support.add_argument(
        "--representation",
        choices=("facts", "passages"),
        default="facts",
        help="Map selected atomic facts or retrieved passages to native evidence units",
    )
    support.set_defaults(handler=_support_evaluate)
    compare = commands.add_parser(
        "compare", help="run paired cluster bootstrap, randomization, and Holm correction"
    )
    compare.add_argument("--manifest", required=True, help="YAML comparison-family manifest")
    compare.add_argument("--output", default=None, help="Optional JSON result file")
    compare.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    compare.set_defaults(handler=_compare)
    efficiency = commands.add_parser(
        "efficiency", help="summarize per-question latency, calls, tokens, and termination rates"
    )
    efficiency.add_argument("--predictions", required=True, help="Prediction JSONL file")
    efficiency.set_defaults(handler=_efficiency)
    prepare = commands.add_parser(
        "prepare", help="validate an official JSON/JSONL export and write runner-ready JSONL"
    )
    prepare.add_argument("--input", required=True, help="Official JSON or JSONL dataset file")
    prepare.add_argument("--output", required=True, help="Validated JSONL output file")
    prepare.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    prepare.set_defaults(handler=_prepare)
    gold_ranking = commands.add_parser(
        "gold-ranking-evaluate",
        help="evaluate candidate rankings against gold-decomposition alignment labels",
    )
    gold_ranking.add_argument(
        "--annotations", required=True, help="Reasoning-state JSONL with gold-alignment labels"
    )
    gold_ranking.set_defaults(handler=_gold_ranking_evaluate)
    subquestions = commands.add_parser(
        "subquestion-evaluate", help="evaluate explicit generated sub-question quality"
    )
    subquestions.add_argument("--predictions", required=True, help="Prediction JSONL file")
    subquestions.add_argument("--gpt2-model", default="gpt2", help="GPT-2 model identifier")
    subquestions.add_argument(
        "--semantic-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformer model identifier",
    )
    subquestions.add_argument(
        "--device", default="auto", help="Model device such as auto, cpu, or cuda"
    )
    subquestions.set_defaults(handler=_subquestion_evaluate)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2
