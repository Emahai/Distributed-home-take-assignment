#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 00:30:00

source env.sh
python -m src.scripts.run_serial --config configs/base.yaml
