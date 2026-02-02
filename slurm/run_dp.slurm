#!/bin/bash
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -t 02:00:00

source env.sh
srun python -m src.scripts.run_dp_mpi --config configs/data_parallel.yaml
