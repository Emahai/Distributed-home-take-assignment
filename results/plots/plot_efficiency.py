import os
import glob
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_csv(path):
    with open(path, "r") as f:
        r = csv.DictReader(f)
        return list(r)

def mean_epoch_time(path):
    rows = read_csv(path)
    times = [float(x["epoch_time_sec"]) for x in rows]
    if len(times) >= 3:
        times = times[1:]  # drop warmup epoch
    return float(np.mean(times))

def collect(pattern, metrics_dir, world_idx):
    files = glob.glob(os.path.join(metrics_dir, pattern))
    out = {}
    for f in files:
        base = os.path.basename(f).replace(".csv", "")
        parts = base.split("_")
        world = int(parts[world_idx])
        out[world] = mean_epoch_time(f)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="./results/metrics")
    ap.add_argument("--out_dir", default="./results/plots")
    ap.add_argument("--title", default="Efficiency vs Nodes")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    serial = os.path.join(args.metrics_dir, "serial.csv")
    if not os.path.exists(serial):
        raise FileNotFoundError("serial.csv not found")

    t1 = mean_epoch_time(serial)

    # dp_world_<world>.csv -> parts = ["dp","world","32"]
    dp = collect("dp_world_*.csv", args.metrics_dir, world_idx=2)
    # hybrid_world_<world>_stages_2.csv -> parts = ["hybrid","world","32","stages","2"]
    hy = collect("hybrid_world_*_stages_2.csv", args.metrics_dir, world_idx=2)

    dp_nodes = sorted(dp.keys())
    hy_nodes = sorted(hy.keys())

    dp_eff = [(t1 / dp[n]) / n for n in dp_nodes]
    hy_eff = [(t1 / hy[n]) / n for n in hy_nodes]

    plt.figure()
    if dp_nodes:
        plt.plot(dp_nodes, dp_eff, marker="o", label="Strategy 2: Data Parallel (MPI)")
    if hy_nodes:
        plt.plot(hy_nodes, hy_eff, marker="o", label="Strategy 1: Hybrid (Pipeline+DP)")

    plt.xlabel("MPI world size (nodes if 1 rank/node)")
    plt.ylabel("Efficiency E(p) = S(p)/p")
    plt.title(args.title)
    plt.grid(True)
    plt.legend()
    out = os.path.join(args.out_dir, "efficiency.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
