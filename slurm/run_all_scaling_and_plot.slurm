#!/bin/bash
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -t 04:00:00

source env.sh

# 1) Serial baseline (on 1 rank)
srun -n 1 python -m src.scripts.run_serial --config configs/base.yaml

# 2) DP scaling
bash scripts/run_scaling_dp_slurm.sh configs/data_parallel.yaml

# 3) Hybrid scaling
bash scripts/run_scaling_hybrid_slurm.sh configs/hybrid_pipeline.yaml

# 4) Plots + speedup table
bash scripts/make_plots.sh ./results/metrics ./results/plots
