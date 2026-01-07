from __future__ import annotations
import numpy as np
import pandas as pd

def make_synthetic_churn(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic churn dataset (so the repo runs without external data).
    Columns loosely mimic common churn datasets.
    """
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(0, 73, size=n)
    monthly_charges = rng.normal(75, 25, size=n).clip(10, 200)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.60, 0.25, 0.15])
    internet_service = rng.choice(["DSL", "Fiber", "None"], size=n, p=[0.35, 0.50, 0.15])
    has_support = rng.choice([0, 1], size=n, p=[0.55, 0.45])
    paperless_billing = rng.choice([0, 1], size=n, p=[0.40, 0.60])

    # A simple churn probability function (nonlinear-ish)
    base = 0.20
    p = (
        base
        + 0.25 * (contract == "Month-to-month")
        - 0.10 * (contract == "Two year")
        + 0.15 * (internet_service == "Fiber")
        - 0.08 * has_support
        + 0.10 * paperless_billing
        + 0.003 * (monthly_charges - 70)
        - 0.004 * tenure_months
    )
    p = np.clip(p, 0.02, 0.85)
    churn = rng.binomial(1, p, size=n)

    total_charges = (tenure_months * monthly_charges + rng.normal(0, 50, size=n)).clip(0)

    df = pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges.round(2),
            "contract": contract,
            "internet_service": internet_service,
            "has_support": has_support,
            "paperless_billing": paperless_billing,
            "churn": churn,
        }
    )
    return df

def load_data(path: str | None) -> pd.DataFrame:
    """
    If path is None: generate synthetic dataset.
    If path provided: read CSV with a required 'churn' target column.
    """
    if path is None:
        return make_synthetic_churn()
    df = pd.read_csv(path)
    if "churn" not in df.columns:
        raise ValueError("CSV must contain a 'churn' column as the target (0/1).")
    return df
