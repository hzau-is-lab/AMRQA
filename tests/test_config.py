from pathlib import Path

import pytest

from amrqa.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_config_and_override() -> None:
    config = load_config(ROOT / "configs" / "offline.yaml", ["reasoning.max_depth=2"])
    assert config.model.provider == "dry_run"
    assert config.model.temperature == 0.0
    assert config.model.candidate_generation_temperature == 0.7
    assert config.reasoning.max_depth == 2
    assert config.embedding.model_path_env == "AMRQA_EMBEDDING_MODEL_DIR"
    assert config.reasoning.path_width == 1
    assert config.reasoning.amr_reference_weight == 0.5
    assert config.amr.reference_source == "question"
    assert config.runtime.timeout_seconds == 180.0


def test_invalid_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_config(ROOT / "configs" / "offline.yaml", ["reasoning.unknown=1"])
