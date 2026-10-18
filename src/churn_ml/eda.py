"""Descriptive EDA of synthetic training-style dataset; no identifiers in graphics."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .data import TARGET, validate

def describe(df: pd.DataFrame, output: Path) -> None:
    validate(df)
    output.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["customer_id"]).describe(include="all").to_csv(output / "eda_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df[TARGET].value_counts().sort_index().plot.bar(ax=axes[0])
    axes[0].set_title("Synthetic churn distribution")
    axes[0].set_xlabel("Churn (0=no, 1=yes)")
    df.groupby("plan")[TARGET].mean().sort_index().plot.bar(ax=axes[1])
    axes[1].set_title("Synthetic churn rate by plan")
    axes[1].set_ylabel("Rate")
    fig.tight_layout()
    fig.savefig(output / "eda.png", dpi=140)
    plt.close(fig)
