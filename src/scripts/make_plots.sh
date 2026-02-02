#!/usr/bin/env bash
set -euo pipefail

METRICS_DIR="${1:-./results/metrics}"
PLOTS_DIR="${2:-./results/plots}"

mkdir -p "$PLOTS_DIR"

python results/plots/plot_runtime.py --metrics_dir "$METRICS_DIR" --out_dir "$PLOTS_DIR"
python results/plots/plot_speedup.py --metrics_dir "$METRICS_DIR" --out_dir "$PLOTS_DIR"
python results/plots/plot_efficiency.py --metrics_dir "$METRICS_DIR" --out_dir "$PLOTS_DIR"
python results/plots/make_speedup_table.py --metrics_dir "$METRICS_DIR" --out "./results/metrics/speedup_table.csv"

# Change dp_world / hy_world if you ran with other sizes
python results/plots/plot_learning_curves.py --metrics_dir "$METRICS_DIR" --out_dir "$PLOTS_DIR" --dp_world 32 --hy_world 32

echo "All plots saved in $PLOTS_DIR"
