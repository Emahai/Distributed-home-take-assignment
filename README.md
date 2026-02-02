# Parallel CNN Training on Hybrid Distributed–Shared Memory Architecture
## MPI + OpenMP (via Threaded CPU Kernels) Implementation

---

## 1. Project Overview

This project investigates the parallelization of **Convolutional Neural Network (CNN)** training for large-scale image classification on a **hybrid high-performance computing (HPC) architecture**. The target system consists of:

- **32 distributed processing nodes** (distributed-memory architecture)
- **64-core shared-memory CPU per node**

The main objective is to:
1. Implement a **serial CNN baseline**
2. Implement and evaluate **two parallel strategies** using a **hybrid MPI–OpenMP programming model**
3. Compare performance, scalability, and correctness across strategies
4. Analyze key challenges in parallel CNN training and propose mitigation strategies

The implementation is done in **Python**, using:
- **MPI** for distributed-memory parallelism (`mpi4py`)
- **OpenMP-style threading** via optimized CPU math libraries (PyTorch + oneDNN / MKL / OpenBLAS)

---

## 2. Parallelization Strategies

### Strategy 2 (Baseline Parallel Strategy): Data Parallelism
- Each MPI rank processes a different shard of the training data
- All ranks maintain a full replica of the CNN model
- Gradients are synchronized using `MPI_Allreduce`
- OpenMP threads exploit shared-memory parallelism within each node

This strategy is widely used in practice due to its simplicity, scalability, and minimal communication overhead.

---

### Strategy 1 (Hybrid Strategy): Data Parallelism + Model Parallelism
- CNN layers are split across MPI ranks using **pipeline model parallelism**
- Multiple pipeline replicas are trained in parallel using data parallelism
- Activations are exchanged between pipeline stages using point-to-point MPI communication
- Gradients are synchronized across replicas per pipeline stage

This strategy is more complex but demonstrates how hybrid parallelism can be applied when model size or memory constraints require it.

---

## 3. Hardware and Programming Model Mapping

### Hardware
- 32 compute nodes
- 64 CPU cores per node
- Distributed memory across nodes
- Shared memory within each node

### Programming Model
| Level | Technology | Purpose |
|------|-----------|---------|
| Distributed memory | MPI (mpi4py) | Inter-node communication |
| Shared memory | OpenMP-style threading | Intra-node parallelism |
| Compute kernels | PyTorch + oneDNN / MKL | Optimized CNN operations |

### Execution Model
- **1 MPI rank per node**
- **64 OpenMP threads per rank**

This mapping ensures full utilization of the available hardware while avoiding oversubscription.

---

## 4. Environment Configuration

Before running any experiments, configure the environment:

```bash
source env.sh

---

### Running
- serial command
python -m src.scripts.run_serial --config configs/base.yaml

- data-parallel command
mpirun -np 2 python -m src.scripts.run_dp_mpi --config configs/data_parallel.yaml

- hybrid pipeline command
mpirun -np 4 python -m src.scripts.run_hybrid_pipeline --config configs/hybrid_pipeline.yaml
mpirun -np 4 python -m src.scripts.run_hybrid_pipeline --config configs/hybrid_pipeline.yaml


### Reproducing
- fixed config YAML files
- scaling workflow (`run_all_scaling_and_plot.slurm`)
- plot regeneration (`make_plots.sh`)
- outputs/paths for metrics and plots

```markdown
## Reproducibility Checklist
To reproduce all reported results:

1) Install dependencies: `pip install -r requirements.txt`
2) Load thread settings: `source env.sh`
3) Run full scaling script:
   `sbatch slurm/run_all_scaling_and_plot.slurm`
4) Confirm outputs exist:
   - `results/metrics/speedup_table.csv`
   - `results/plots/speedup.png`
   - `results/plots/efficiency.png`

