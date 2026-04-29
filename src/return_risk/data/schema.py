"""
Pandera schema for the raw Amazon e-commerce dataset.
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

_ALLOWED_CATEGORIES = {"Electronics", "Sports", "Clothing", "Home", "Beauty"}
_ALLOWED_DEVICES = {"Mobile App", "Tablet", "Web"}
_ALLOWED_PAYMENTS = {"Cash on Delivery", "Credit Card", "Debit Card", "UPI"}
_ALLOWED_DELIVERY_STATUSES = {"Delivered", "Delayed", "In Transit", "Returned"}


raw_schema = DataFrameSchema(
    columns={
        "user_id": Column(str, nullable=False),
        "product_id": Column(str, nullable=False),
        "category": Column(str, Check.isin(_ALLOWED_CATEGORIES), nullable=False),
        "subcategory": Column(str, nullable=False),
        "brand": Column(str, nullable=False),
        "price": Column(float, Check.ge(0), nullable=False),
        "discount": Column(float, Check.in_range(0, 100), nullable=False),
        "final_price": Column(float, Check.ge(0), nullable=False),
        "rating": Column(float, Check.in_range(0, 5), nullable=False),
        "review_count": Column(int, Check.ge(0), nullable=False),
        "stock": Column(int, Check.ge(0), nullable=False),
        "seller_id": Column(str, nullable=False),
        "seller_rating": Column(float, Check.in_range(0, 5), nullable=False),
        "purchase_date": Column(str, nullable=False),
        "shipping_time_days": Column(int, Check.ge(0), nullable=False),
        "location": Column(str, nullable=False),
        "device": Column(str, Check.isin(_ALLOWED_DEVICES), nullable=False),
        "payment_method": Column(str, Check.isin(_ALLOWED_PAYMENTS), nullable=False),
        "is_returned": Column(bool, nullable=False),
        "delivery_status": Column(str, Check.isin(_ALLOWED_DELIVERY_STATUSES), nullable=False),
    },
    strict=True,
    coerce=False,
    ordered=False,
)
