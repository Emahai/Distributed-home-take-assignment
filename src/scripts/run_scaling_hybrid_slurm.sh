#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/hybrid_pipeline.yaml}"
SIZES=(2 4 8 16 32)

for P in "${SIZES[@]}"; do
  echo "=== Running HYBRID with ranks=$P ==="
  srun -n "$P" --ntasks-per-node=1 python -m src.scripts.run_hybrid_pipeline --config "$CONFIG"
done
