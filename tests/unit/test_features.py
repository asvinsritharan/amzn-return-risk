"""Unit tests for the feature pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from return_risk.features.build_features import (
    DateFeatureExtractor,
    build_full_pipeline,
)


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [100.0, 200.0, 50.0, 75.0],
            "discount": [10.0, 5.0, 20.0, 0.0],
            "category": ["Electronics", "Sports", "Electronics", "Home"],
            "device": ["Mobile App", "Web", "Tablet", "Web"],
            "seller_id": ["S1", "S2", "S1", "S3"],
            "purchase_date": ["2025-01-01", "2025-02-15", "2025-03-10", "2025-04-20"],
        }
    )


def test_date_feature_extractor_shapes(sample_frame):
    ext = DateFeatureExtractor(date_columns=["purchase_date"])
    out = ext.fit_transform(sample_frame[["purchase_date"]])
    assert out.shape == (4, 3)


def test_date_feature_extractor_handles_bad_dates(sample_frame):
    sample_frame.loc[0, "purchase_date"] = "garbage"
    ext = DateFeatureExtractor(date_columns=["purchase_date"])
    out = ext.fit_transform(sample_frame[["purchase_date"]])
    # bad date row should not blow up; coerced to -1/0 sentinels
    assert out.shape == (4, 3)
    assert not np.isnan(out).any()


def test_full_pipeline_fits_and_predicts(sample_frame):
    y = pd.Series([0, 1, 0, 1])
    pipeline = build_full_pipeline(
        numeric_cols=["price", "discount"],
        low_card_cols=["category", "device"],
        high_card_cols=["seller_id"],
        datetime_cols=["purchase_date"],
        estimator=LogisticRegression(max_iter=100),
    )
    pipeline.fit(sample_frame, y)
    proba = pipeline.predict_proba(sample_frame)
    assert proba.shape == (4, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_handles_unseen_low_card_value(sample_frame):
    """OneHotEncoder is configured with handle_unknown='ignore' — a new
    category at serving time should not crash inference."""
    y = pd.Series([0, 1, 0, 1])
    pipeline = build_full_pipeline(
        numeric_cols=["price", "discount"],
        low_card_cols=["category", "device"],
        high_card_cols=["seller_id"],
        datetime_cols=["purchase_date"],
        estimator=LogisticRegression(max_iter=100),
    )
    pipeline.fit(sample_frame, y)
    new = sample_frame.copy()
    new["category"] = "BrandNewCategory"
    proba = pipeline.predict_proba(new)
    assert proba.shape == (4, 2)
