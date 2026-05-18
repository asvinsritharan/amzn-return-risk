"""Inject realistic learnable signal into is_returned.

The public Kaggle dataset has a randomly-assigned target (baseline AUC ~0.56).
This script overwrites `is_returned` and the corresponding `delivery_status`
with a target generated from realistic e-commerce return correlations:

  - shipping_time_days  (delayed orders return more)
  - final_price         (pricier items return more)
  - category            (Clothing > Electronics > Sports > Home > Beauty)
  - seller_rating       (low-rated sellers see more returns)
  - rating              (low product ratings → more returns)
  - device              (Mobile App slightly higher than Web/Tablet)
  - payment_method      (Cash on Delivery slightly higher)

Plus Gaussian noise so the model can't be perfect.

Usage:
    uv run python scripts/inject_signal.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Per-category log-odds bump (rough real-world ordering)
CATEGORY_EFFECT = {
    "Clothing": 0.9,
    "Electronics": 0.4,
    "Sports": 0.1,
    "Home": -0.2,
    "Beauty": -0.4,
}
DEVICE_EFFECT = {"Mobile App": 0.2, "Tablet": 0.0, "Web": -0.1}
PAYMENT_EFFECT = {
    "Cash on Delivery": 0.4,
    "Credit Card": 0.0,
    "Debit Card": 0.0,
    "UPI": -0.1,
}


def _logit_to_prob(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def inject(seed: int = 42) -> None:
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    raw_file = PROJECT_ROOT / cfg["data"]["raw_file"]
    df = pd.read_csv(raw_file)
    print(
        f"[load] {raw_file.name}  rows={len(df):,}  "
        f"old positive_rate={df['is_returned'].mean():.4f}"
    )

    rng = np.random.default_rng(seed)

    # Standardize numeric drivers (mean 0, std 1) so coefficients are interpretable
    def _z(s: pd.Series) -> np.ndarray:
        return ((s - s.mean()) / s.std()).to_numpy()

    z_ship = _z(df["shipping_time_days"])
    z_price = _z(df["final_price"])
    z_seller_rating = _z(df["seller_rating"])
    z_rating = _z(df["rating"])

    # Linear combination in log-odds space
    logit = (
        -2.4  # baseline → ~9% positive rate
        + 0.85 * z_ship  # delayed → more returns
        + 0.35 * z_price  # pricey → more returns
        - 0.45 * z_seller_rating  # bad sellers → more returns
        - 0.35 * z_rating  # bad products → more returns
        + df["category"].map(CATEGORY_EFFECT).fillna(0.0).to_numpy()
        + df["device"].map(DEVICE_EFFECT).fillna(0.0).to_numpy()
        + df["payment_method"].map(PAYMENT_EFFECT).fillna(0.0).to_numpy()
        + rng.normal(0.0, 0.6, size=len(df))  # irreducible noise
    )

    prob = _logit_to_prob(logit)
    is_returned = rng.uniform(size=len(df)) < prob

    df["is_returned"] = is_returned

    # Keep delivery_status consistent: returned rows → "Returned",
    # non-returned rows → uniformly Delivered / Delayed / In Transit
    non_returned_statuses = rng.choice(
        ["Delivered", "Delayed", "In Transit"], size=(~is_returned).sum()
    )
    df.loc[is_returned, "delivery_status"] = "Returned"
    df.loc[~is_returned, "delivery_status"] = non_returned_statuses

    print(f"[done] new positive_rate={df['is_returned'].mean():.4f}")
    df.to_csv(raw_file, index=False)
    print(f"[write] {raw_file}")


if __name__ == "__main__":
    inject()
