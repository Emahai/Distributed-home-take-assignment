#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/data_parallel.yaml}"

# List of MPI world sizes for scaling runs
SIZES=(1 2 4 8 16 32)

for P in "${SIZES[@]}"; do
  echo "=== Running DP with world_size=$P ==="
  mpirun -np "$P" python -m src.scripts.run_dp_mpi --config "$CONFIG"
done
