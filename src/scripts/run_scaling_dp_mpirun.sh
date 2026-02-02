#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/data_parallel.yaml}"
SIZES=(1 2 4 8)

for P in "${SIZES[@]}"; do
  echo "=== Running DP with ranks=$P ==="
  mpirun -np "$P" python -m src.scripts.run_dp_mpi --config "$CONFIG"
done
