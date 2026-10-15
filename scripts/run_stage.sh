#!/usr/bin/env bash
# Run the standard workload set against one engine stage.
#   scripts/run_stage.sh <engine> <out-dir> [extra run.py args...]
# Example:
#   scripts/run_stage.sh naive results/stage0 --threads 32
set -euo pipefail
cd "$(dirname "$0")/.."
engine="$1"; out="$2"; shift 2
for w in offline closed8 uniform hetero burst; do
  .venv/bin/python -m miniserve.bench.run --engine "$engine" --workload "$w" --out "$out" "$@"
done
