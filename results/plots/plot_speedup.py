import os
import glob
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_csv(path):
    with open(path, "r") as f:
        r = csv.DictReader(f)
        rows = list(r)
    return rows

def last_epoch_time(path):
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No rows in {path}")
    # Use mean epoch time (excluding epoch 0 warmup if possible)
    times = [float(x["epoch_time_sec"]) for x in rows]
    if len(times) >= 3:
        times = times[1:]  # drop epoch 0
    return float(np.mean(times))

def collect_dp_metrics(metrics_dir):
    dp_files = glob.glob(os.path.join(metrics_dir, "dp_world_*.csv"))
    out = {}
    for f in dp_files:
        base = os.path.basename(f)
        # dp_world_32.csv
        world = int(base.split("_")[-1].replace(".csv", ""))
        out[world] = last_epoch_time(f)
    return out

def collect_hybrid_metrics(metrics_dir):
    hy_files = glob.glob(os.path.join(metrics_dir, "hybrid_world_*_stages_2.csv"))
    out = {}
    for f in hy_files:
        base = os.path.basename(f)
        # hybrid_world_32_stages_2.csv
        parts = base.replace(".csv", "").split("_")
        world = int(parts[2])
        out[world] = last_epoch_time(f)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="./results/metrics")
    ap.add_argument("--out_dir", default="./results/plots")
    ap.add_argument("--title", default="Speedup vs Nodes")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    serial_path = os.path.join(args.metrics_dir, "serial.csv")
    if not os.path.exists(serial_path):
        raise FileNotFoundError(f"Missing serial baseline: {serial_path}")

    t1 = last_epoch_time(serial_path)

    dp = collect_dp_metrics(args.metrics_dir)
    hy = collect_hybrid_metrics(args.metrics_dir)

    # Prepare series
    dp_nodes = sorted(dp.keys())
    hy_nodes = sorted(hy.keys())

    dp_speedup = [t1 / dp[n] for n in dp_nodes]
    hy_speedup = [t1 / hy[n] for n in hy_nodes]

    plt.figure()
    if dp_nodes:
        plt.plot(dp_nodes, dp_speedup, marker="o", label="Strategy 2: Data Parallel (MPI)")
    if hy_nodes:
        plt.plot(hy_nodes, hy_speedup, marker="o", label="Strategy 1: Hybrid (Pipeline+DP)")

    plt.xlabel("MPI world size (nodes if 1 rank/node)")
    plt.ylabel("Speedup S(p) = T(1)/T(p)")
    plt.title(args.title)
    plt.grid(True)
    plt.legend()
    out = os.path.join(args.out_dir, "speedup.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
