from pathlib import Path

from amrqa.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_offline_cli_writes_prediction_file(tmp_path) -> None:
    output = tmp_path / "predictions.jsonl"
    exit_code = main(
        [
            "run",
            "--config",
            str(ROOT / "configs" / "offline.yaml"),
            "--data",
            str(ROOT / "data" / "demo.jsonl"),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    assert '"prediction": "No-answer"' in output.read_text(encoding="utf-8")
    manifest = output.with_name(f"{output.name}.manifest.json")
    assert manifest.exists()
    text = manifest.read_text(encoding="utf-8")
    assert '"evaluated_count": 1' in text
    assert str(tmp_path) not in text
