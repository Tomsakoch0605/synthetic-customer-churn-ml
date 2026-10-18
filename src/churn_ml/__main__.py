"""CLI: python -m churn_ml generate|analyze|train|all."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .data import generate
from .eda import describe
from .model import run

def main():
    parser = argparse.ArgumentParser(description="Independent synthetic churn ML demonstration")
    parser.add_argument("command", choices=["generate", "analyze", "train", "all"])
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_subscriptions.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.command in ("generate", "all"):
        args.data.parent.mkdir(parents=True, exist_ok=True)
        generate(args.rows, args.seed).to_csv(args.data, index=False)
        print(f"Generated {args.rows} fictional records: {args.data}")
    if args.command in ("analyze", "train", "all"):
        df = pd.read_csv(args.data)
        if args.command in ("analyze", "all"):
            describe(df, args.output)
            print("EDA written to", args.output)
        if args.command in ("train", "all"):
            metrics = run(df, args.output, args.seed)
            print(json.dumps({"selected_model": metrics["selected_model"], "test": metrics["test"]}, indent=2))
if __name__ == "__main__":
    main()
