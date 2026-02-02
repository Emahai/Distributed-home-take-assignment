import os
import glob
import csv
import argparse
import numpy as np

def read_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))

def mean_epoch_time(path):
    rows = read_csv(path)
    times = [float(x["epoch_time_sec"]) for x in rows]
    if len(times) >= 3:
        times = times[1:]
    return float(np.mean(times))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="./results/metrics")
    ap.add_argument("--out", default="./results/metrics/speedup_table.csv")
    args = ap.parse_args()

    serial = os.path.join(args.metrics_dir, "serial.csv")
    t1 = mean_epoch_time(serial)

    rows = []
    # Strategy 2
    for f in sorted(glob.glob(os.path.join(args.metrics_dir, "dp_world_*.csv"))):
        world = int(os.path.basename(f).split("_")[-1].replace(".csv", ""))
        tp = mean_epoch_time(f)
        sp = t1 / tp
        ep = sp / world
        rows.append(["dp", world, tp, sp, ep])

    # Strategy 1
    for f in sorted(glob.glob(os.path.join(args.metrics_dir, "hybrid_world_*_stages_2.csv"))):
        parts = os.path.basename(f).replace(".csv", "").split("_")
        world = int(parts[2])
        tp = mean_epoch_time(f)
        sp = t1 / tp
        ep = sp / world
        rows.append(["hybrid", world, tp, sp, ep])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["strategy", "world_size", "mean_epoch_time_sec", "speedup", "efficiency"])
        for r in rows:
            w.writerow(r)

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
