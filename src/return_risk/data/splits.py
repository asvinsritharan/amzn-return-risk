"""Train/validation/test splits, stratified on the target."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified train/validation/test split.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    val_size is a fraction of the original dataframe, not the post-test remainder.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)  # XGBoost wants int, not bool

    # First carve off the test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Then carve val out of the remainder
    val_fraction_of_remainder = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction_of_remainder,
        stratify=y_trainval,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
