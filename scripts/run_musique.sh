#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_musique.sh path/to/musique_dev.jsonl outputs/musique.jsonl
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 DATA_JSONL OUTPUT_JSONL" >&2
  exit 2
fi

amrqa run --config configs/default.yaml --data "$1" --output "$2"
