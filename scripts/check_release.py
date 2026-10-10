#!/usr/bin/env python3
"""Check a public release for secrets and local artifacts; allow public contact details."""

from __future__ import annotations

import argparse
from pathlib import Path

from check_anonymity import scan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="Repository root to scan")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        parser.error("Repository root must be an existing directory")
    findings = scan(root, anonymous=False)
    if findings:
        print("Public-release scan failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    count = sum(
        1
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )
    print(f"Public-release scan passed: {count} files checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
