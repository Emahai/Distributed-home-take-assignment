import argparse
from src.scripts.config import load_config
from src.train.dp_mpi_train import run_dp_mpi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dp_mpi(cfg)

if __name__ == "__main__":
    main()
