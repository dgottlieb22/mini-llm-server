#!/usr/bin/env bash
# Run the standard workload set against one engine stage.
#   scripts/run_stage.sh <engine> <out-dir> [extra run.py args...]
# Example:
#   scripts/run_stage.sh paged results/stage4 --threads 32 --engine-kw kv_budget_tokens=8192 --engine-kw block_size=16
set -euo pipefail
cd "$(dirname "$0")/.."
engine="$1"; out="$2"; shift 2
for w in offline closed8 uniform hetero burst uniform_hi hetero_hi uniform_8x hetero_4x; do
  .venv/bin/python -m miniserve.bench.run --engine "$engine" --workload "$w" --out "$out" "$@"
done
