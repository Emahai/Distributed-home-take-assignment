import argparse
from src.scripts.config import load_config
from src.train.hybrid_pipeline_train import run_hybrid_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_hybrid_pipeline(cfg)

if __name__ == "__main__":
    main()
