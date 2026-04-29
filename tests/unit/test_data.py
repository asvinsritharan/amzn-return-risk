"""Unit tests for data loading + schema validation."""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from return_risk.data.loader import drop_leakage_and_ids
from return_risk.data.schema import raw_schema


@pytest.fixture
def valid_row() -> dict:
    """A single realistic row that should pass the schema."""
    return {
        "user_id": "U1",
        "product_id": "P1",
        "category": "Electronics",
        "subcategory": "Mobile",
        "brand": "Samsung",
        "price": 100.0,
        "discount": 10.0,
        "final_price": 90.0,
        "rating": 4.5,
        "review_count": 10,
        "stock": 5,
        "seller_id": "S1",
        "seller_rating": 4.0,
        "purchase_date": "2025-01-01",
        "shipping_time_days": 3,
        "location": "Bangalore",
        "device": "Mobile App",
        "payment_method": "UPI",
        "is_returned": False,
        "delivery_status": "Delivered",
    }


def test_schema_accepts_valid_row(valid_row):
    df = pd.DataFrame([valid_row])
    raw_schema.validate(df)  # should not raise


def test_schema_rejects_bad_rating(valid_row):
    valid_row["rating"] = 9.9  # out of [0, 5]
    df = pd.DataFrame([valid_row])
    with pytest.raises(pandera.errors.SchemaErrors):
        raw_schema.validate(df, lazy=True)


def test_schema_rejects_negative_price(valid_row):
    valid_row["price"] = -1.0
    df = pd.DataFrame([valid_row])
    with pytest.raises(pandera.errors.SchemaErrors):
        raw_schema.validate(df, lazy=True)


def test_schema_rejects_unknown_device(valid_row):
    valid_row["device"] = "Smart Fridge"
    df = pd.DataFrame([valid_row])
    with pytest.raises(pandera.errors.SchemaErrors):
        raw_schema.validate(df, lazy=True)


def test_drop_removes_expected_columns(valid_row):
    df = pd.DataFrame([valid_row])
    out = drop_leakage_and_ids(df)
    for col in ["user_id", "product_id", "delivery_status"]:
        assert col not in out.columns
    assert "is_returned" in out.columns
