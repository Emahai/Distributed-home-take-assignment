#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/data_parallel.yaml}"
SIZES=(1 2 4 8 16 32)

for P in "${SIZES[@]}"; do
  echo "=== Running DP with ranks=$P ==="
  srun -n "$P" --ntasks-per-node=1 python -m src.scripts.run_dp_mpi --config "$CONFIG"
done
