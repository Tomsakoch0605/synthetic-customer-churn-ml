"""Synthetic, invented subscription dataset; no corporate records."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = ["tenure_months", "monthly_fee", "support_tickets_90d", "logins_30d", "payment_failures_90d", "plan", "auto_renew"]
NUMERIC = ["tenure_months", "monthly_fee", "support_tickets_90d", "logins_30d", "payment_failures_90d"]
CATEGORICAL = ["plan", "auto_renew"]
TARGET = "churn"

def generate(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible fictional snapshot; associations are explicitly invented."""
    if n < 100:
        raise ValueError("n must be >= 100")
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 61, n)
    plan = rng.choice(["basic", "plus", "premium"], n, p=[0.48, 0.34, 0.18])
    auto = rng.choice(["yes", "no"], n, p=[0.66, 0.34])
    fee = np.clip(rng.normal(24, 6, n) + np.select([plan == "plus", plan == "premium"], [18, 42], default=0), 8, 105)
    tickets = rng.poisson(0.85, n)
    logins = rng.poisson(12, n)
    fails = rng.binomial(3, 0.09, n)
    score = (-0.85 - 0.032 * tenure + 0.018 * fee + 0.55 * tickets - 0.075 * logins + 0.8 * fails + 0.85 * (auto == "no") + 0.27 * (plan == "basic"))
    probability = 1 / (1 + np.exp(-score))
    churn = rng.binomial(1, probability)
    return pd.DataFrame({"customer_id": [f"DEMO-{i:06d}" for i in range(n)], "tenure_months": tenure, "monthly_fee": np.round(fee, 2), "support_tickets_90d": tickets, "logins_30d": logins, "payment_failures_90d": fails, "plan": plan, "auto_renew": auto, "churn": churn})

def validate(df: pd.DataFrame) -> None:
    required = set(FEATURES + [TARGET, "customer_id"])
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")
    if df.customer_id.duplicated().any():
        raise ValueError("Duplicated customer_id")
    if not set(df[TARGET].dropna().unique()).issubset({0, 1}) or df[TARGET].isna().any():
        raise ValueError("Target must be 0/1 and complete")
    if len(df) < 100:
        raise ValueError("Too few rows")
    if (df[NUMERIC].isna().any().any() or (df[NUMERIC] < 0).any().any() or df[CATEGORICAL].isna().any().any()):
        raise ValueError("Missing or negative feature values")
