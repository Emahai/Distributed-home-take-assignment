import os
import csv
import argparse
import matplotlib.pyplot as plt

def read_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))

def series(rows, key):
    return [float(r[key]) for r in rows]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="./results/metrics")
    ap.add_argument("--out_dir", default="./results/plots")
    ap.add_argument("--dp_world", type=int, default=32)
    ap.add_argument("--hy_world", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    serial = os.path.join(args.metrics_dir, "serial.csv")
    dp = os.path.join(args.metrics_dir, f"dp_world_{args.dp_world}.csv")
    hy = os.path.join(args.metrics_dir, f"hybrid_world_{args.hy_world}_stages_2.csv")

    rows_s = read_csv(serial)
    rows_dp = read_csv(dp) if os.path.exists(dp) else []
    rows_hy = read_csv(hy) if os.path.exists(hy) else []

    # Loss curves
    plt.figure()
    plt.plot(series(rows_s, "epoch"), series(rows_s, "train_loss"), marker="o", label="Serial train loss")
    if rows_dp:
        plt.plot(series(rows_dp, "epoch"), series(rows_dp, "train_loss"), marker="o", label="DP train loss")
    if rows_hy:
        plt.plot(series(rows_hy, "epoch"), series(rows_hy, "train_loss"), marker="o", label="Hybrid train loss")

    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Learning curve: Train loss")
    plt.grid(True)
    plt.legend()
    out1 = os.path.join(args.out_dir, "learning_loss.png")
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    print(f"Wrote {out1}")

    # Accuracy curves (serial & dp include train_acc; hybrid skeleton logs train_acc but no test_acc)
    plt.figure()
    plt.plot(series(rows_s, "epoch"), series(rows_s, "train_acc"), marker="o", label="Serial train acc")
    if rows_dp:
        plt.plot(series(rows_dp, "epoch"), series(rows_dp, "train_acc"), marker="o", label="DP train acc")
    if rows_hy and "train_acc" in rows_hy[0]:
        plt.plot(series(rows_hy, "epoch"), series(rows_hy, "train_acc"), marker="o", label="Hybrid train acc")

    plt.xlabel("Epoch")
    plt.ylabel("Train accuracy")
    plt.title("Learning curve: Train accuracy")
    plt.grid(True)
    plt.legend()
    out2 = os.path.join(args.out_dir, "learning_acc.png")
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"Wrote {out2}")

    # Test accuracy (only if available)
    if "test_acc" in rows_s[0]:
        plt.figure()
        plt.plot(series(rows_s, "epoch"), series(rows_s, "test_acc"), marker="o", label="Serial test acc")
        if rows_dp and "test_acc" in rows_dp[0]:
            plt.plot(series(rows_dp, "epoch"), series(rows_dp, "test_acc"), marker="o", label="DP test acc")
        plt.xlabel("Epoch")
        plt.ylabel("Test accuracy")
        plt.title("Correctness check: Test accuracy")
        plt.grid(True)
        plt.legend()
        out3 = os.path.join(args.out_dir, "test_acc.png")
        plt.savefig(out3, dpi=200, bbox_inches="tight")
        print(f"Wrote {out3}")

if __name__ == "__main__":
    main()
