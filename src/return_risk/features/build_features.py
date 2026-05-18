"""Feature engineering pipeline.

Returns a single sklearn Pipeline (preprocessing + estimator) so the API
loads one artifact and keeps train and serve behavior in sync.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract day-of-week, month, and days-since-epoch from date columns.

    Expects a DataFrame with the configured datetime columns. The reference
    date is fixed so training and serving produce the same numeric offsets.
    """

    REFERENCE_DATE = pd.Timestamp("2024-01-01")

    def __init__(self, date_columns: list[str]):
        self.date_columns = date_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        X = X.copy()
        out_frames = []
        for col in self.date_columns:
            ts = pd.to_datetime(X[col], format="%Y-%m-%d", errors="coerce")
            frame = pd.DataFrame(
                {
                    f"{col}__dayofweek": ts.dt.dayofweek.fillna(-1).astype(int),
                    f"{col}__month": ts.dt.month.fillna(-1).astype(int),
                    f"{col}__days_since_ref": (
                        (ts - self.REFERENCE_DATE).dt.days.fillna(0).astype(int)
                    ),
                }
            )
            out_frames.append(frame)
        return pd.concat(out_frames, axis=1).to_numpy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = []
        for col in self.date_columns:
            names += [
                f"{col}__dayofweek",
                f"{col}__month",
                f"{col}__days_since_ref",
            ]
        return np.array(names)


def build_preprocessor(
    numeric_cols: list[str],
    low_card_cols: list[str],
    high_card_cols: list[str],
    datetime_cols: list[str],
) -> ColumnTransformer:
    """Build the preprocessing ColumnTransformer."""
    transformers = []

    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))

    if low_card_cols:
        transformers.append(
            (
                "low_card",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                low_card_cols,
            )
        )

    if high_card_cols:
        transformers.append(
            (
                "high_card",
                TargetEncoder(smoothing=10.0, handle_unknown="value"),
                high_card_cols,
            )
        )

    if datetime_cols:
        transformers.append(
            ("date", DateFeatureExtractor(date_columns=datetime_cols), datetime_cols)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_full_pipeline(
    numeric_cols: list[str],
    low_card_cols: list[str],
    high_card_cols: list[str],
    datetime_cols: list[str],
    estimator,
) -> Pipeline:
    """Combine the preprocessor and estimator into one sklearn Pipeline."""
    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        low_card_cols=low_card_cols,
        high_card_cols=high_card_cols,
        datetime_cols=datetime_cols,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ]
    )
