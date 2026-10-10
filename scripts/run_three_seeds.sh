#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_three_seeds.sh CONFIG DATA_JSONL OUTPUT_PREFIX
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 CONFIG DATA_JSONL OUTPUT_PREFIX" >&2
  exit 2
fi

config=$1
data=$2
output_prefix=$3
seeds=(42 43 44)

for seed in "${seeds[@]}"; do
  amrqa run \
    --config "$config" \
    --data "$data" \
    --output "${output_prefix}_seed${seed}.jsonl" \
    --seed "$seed"
done
