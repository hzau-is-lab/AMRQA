"""Anonymous, machine-independent run manifests."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any, Optional, Union

from .config import AMRQAConfig


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def write_run_manifest(
    *,
    config: AMRQAConfig,
    config_path: Union[str, Path],
    data_path: Union[str, Path],
    output_path: Union[str, Path],
    evaluated_count: int,
    limit: Optional[int],
) -> Path:
    """Write a sidecar manifest without usernames, hostnames, endpoints, or absolute paths."""

    config_file = Path(config_path)
    data_file = Path(data_path)
    prediction_file = Path(output_path)
    expanded_config = config.to_dict()
    serialized_config = json.dumps(
        expanded_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "config": {
            "file": config_file.name,
            "source_sha256": sha256_file(config_file),
            "expanded_sha256": hashlib.sha256(serialized_config).hexdigest(),
            "expanded": expanded_config,
        },
        "input": {
            "file": data_file.name,
            "sha256": sha256_file(data_file),
            "limit": limit,
        },
        "predictions": {
            "file": prediction_file.name,
            "sha256": sha256_file(prediction_file),
            "evaluated_count": evaluated_count,
        },
        "environment": {
            "python": platform.python_version(),
            "packages": _package_versions(
                ("amrqa", "amrlib", "FlagEmbedding", "openai", "sentence-transformers")
            ),
        },
    }
    manifest_path = prediction_file.with_name(f"{prediction_file.name}.manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
