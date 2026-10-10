"""Public releases allow intended contact details but must still reject private artifacts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run_check(root: Path, script: str = "check_release.py") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), str(root)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_public_contact_is_allowed_but_anonymous_check_remains_strict(tmp_path):
    contact = "maintainer" + "@" + "example.org"
    (tmp_path / "README.md").write_text(
        "AMRQA by Wentao Deng and collaborators.\n"
        "https://github.com/hzau-is-lab/AMRQA\n" + contact,
        encoding="utf-8",
    )
    assert run_check(tmp_path).returncode == 0
    anonymous = run_check(tmp_path, "check_anonymity.py")
    assert anonymous.returncode == 1
    assert "email address: README.md" in anonymous.stdout


@pytest.mark.parametrize(
    ("content", "label"),
    [
        ("ghp_" + "A" * 25, "GitHub token"),
        ("/" + "home/researcher/private/", "POSIX home path"),
        ("https://" + "user:password@" + "example.org", "credential-bearing URL"),
    ],
)
def test_public_check_still_rejects_sensitive_content(tmp_path, content, label):
    (tmp_path / "example.txt").write_text(content, encoding="utf-8")
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert f"{label}: example.txt" in result.stdout
    assert content not in result.stdout


def test_public_check_rejects_private_env_file(tmp_path):
    (tmp_path / ".env").write_text("LOCAL_SETTING=placeholder\n", encoding="utf-8")
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert "forbidden file: .env" in result.stdout


def test_public_check_rejects_local_cache(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert "generated/private directory: __pycache__" in result.stdout


def test_public_check_rejects_nonexistent_root(tmp_path):
    result = run_check(tmp_path / "missing")
    assert result.returncode != 0
    assert "existing directory" in result.stderr
