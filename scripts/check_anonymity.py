#!/usr/bin/env python3
"""Fail when a release tree contains common identity, secret, or local-artifact leaks."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

IGNORED_DIRECTORIES = {".git"}
FORBIDDEN_DIRECTORIES = {
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "venv",
}
FORBIDDEN_FILENAMES = {
    ".bash_history",
    ".env",
    ".netrc",
    ".python_history",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
}
FORBIDDEN_SUFFIXES = {".key", ".pem", ".pyc", ".pyo"}
TEXT_PATTERNS = {
    "POSIX home path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "root home path": re.compile("/" + r"root(?:/|\b)"),
    "mounted user path": re.compile(r"/(?:data|mnt|workspace)/[^/\s]+/"),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
    "email address": re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "Anthropic secret": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{16,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "Google API key": re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "credential-bearing URL": re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@"),
    "assigned secret": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
}


def scan(root: Path, *, anonymous: bool = True) -> list[str]:
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        forbidden_parts = set(relative.parts) & FORBIDDEN_DIRECTORIES
        egg_info_parts = {part for part in relative.parts if part.endswith(".egg-info")}
        forbidden_parts.update(egg_info_parts)
        if forbidden_parts:
            if path.is_dir() and path.name in forbidden_parts:
                findings.append(
                    "generated/private directory: "
                    f"{relative} ({', '.join(sorted(forbidden_parts))})"
                )
            continue
        if path.is_symlink():
            findings.append(f"symbolic link requires manual review: {relative}")
            continue
        if not path.is_file():
            continue
        if (
            path.name in FORBIDDEN_FILENAMES
            or path.name.startswith("credentials.")
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
        ):
            findings.append(f"forbidden file: {relative}")
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            findings.append(f"unexpected binary file: {relative}")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            findings.append(f"non-UTF-8 file requires manual review: {relative}")
            continue
        for label, pattern in TEXT_PATTERNS.items():
            if label == "email address" and not anonymous:
                continue
            if pattern.search(text):
                findings.append(f"{label}: {relative}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="Repository root to scan")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    findings = scan(root)
    if findings:
        print("Anonymous-release scan failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    count = sum(1 for path in root.rglob("*") if path.is_file())
    print(f"Anonymous-release scan passed: {count} files checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
