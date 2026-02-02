import os
import glob
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))

def mean_epoch_time(path):
    rows = read_csv(path)
    times = [float(x["epoch_time_sec"]) for x in rows]
    if len(times) >= 3:
        times = times[1:]  # drop warmup
    return float(np.mean(times))

def collect_dp(metrics_dir):
    out = {}
    for f in glob.glob(os.path.join(metrics_dir, "dp_world_*.csv")):
        world = int(os.path.basename(f).split("_")[-1].replace(".csv", ""))
        out[world] = mean_epoch_time(f)
    return out

def collect_hybrid(metrics_dir):
    out = {}
    for f in glob.glob(os.path.join(metrics_dir, "hybrid_world_*_stages_2.csv")):
        parts = os.path.basename(f).replace(".csv", "").split("_")
        world = int(parts[2])
        out[world] = mean_epoch_time(f)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="./results/metrics")
    ap.add_argument("--out_dir", default="./results/plots")
    ap.add_argument("--title", default="Mean Epoch Time vs Nodes")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    serial = os.path.join(args.metrics_dir, "serial.csv")
    t1 = mean_epoch_time(serial)

    dp = collect_dp(args.metrics_dir)
    hy = collect_hybrid(args.metrics_dir)

    dp_nodes = sorted(dp.keys())
    hy_nodes = sorted(hy.keys())

    plt.figure()
    plt.plot([1], [t1], marker="o", label="Serial baseline (1 rank)")

    if dp_nodes:
        plt.plot(dp_nodes, [dp[n] for n in dp_nodes], marker="o", label="Strategy 2: Data Parallel (MPI)")
    if hy_nodes:
        plt.plot(hy_nodes, [hy[n] for n in hy_nodes], marker="o", label="Strategy 1: Hybrid (Pipeline+DP)")

    plt.xlabel("MPI world size (nodes if 1 rank/node)")
    plt.ylabel("Mean epoch time (seconds)")
    plt.title(args.title)
    plt.grid(True)
    plt.legend()
    out = os.path.join(args.out_dir, "runtime.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
