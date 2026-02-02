import argparse
from src.scripts.config import load_config
from src.train.serial_train import run_serial

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_serial(cfg)

if __name__ == "__main__":
    main()
