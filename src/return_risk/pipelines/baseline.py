"""Logistic regression baseline — anchors what 'good' means before XGBoost runs.

Run from project root:
    uv run python -m return_risk.pipelines.baseline
"""

from __future__ import annotations

import warnings

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from return_risk.data.loader import drop_leakage_and_ids, load_config, load_raw
from return_risk.data.splits import make_splits
from return_risk.features.build_features import build_full_pipeline
from return_risk.pipelines.train import _compute_metrics, _set_mlflow


def baseline() -> str:
    cfg = load_config()
    _set_mlflow(cfg)

    # SAGA on 1M rows is slow to converge; baseline is intentionally simple,
    # not heavily tuned. Document this rather than chase iterations.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    df = load_raw(validate=True)
    df = drop_leakage_and_ids(df)
    target_col = cfg["data"]["target_column"]

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(
        df,
        target_col=target_col,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["validation_size"],
        random_state=cfg["project"]["random_seed"],
    )

    pipeline = build_full_pipeline(
        numeric_cols=cfg["features"]["numeric_cols"],
        low_card_cols=cfg["features"]["low_cardinality_cat_cols"],
        high_card_cols=cfg["features"]["high_cardinality_cat_cols"],
        datetime_cols=cfg["features"]["datetime_cols"],
        estimator=LogisticRegression(
            max_iter=2000,  # ← was 500
            class_weight="balanced",
            solver="saga",  # ← was default lbfgs
            random_state=cfg["project"]["random_seed"],
        ),
    )

    with mlflow.start_run(run_name="baseline_logreg") as run:
        mlflow.set_tag("model_family", "logistic_regression")
        pipeline.fit(X_train, y_train)
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        for k, v in _compute_metrics(y_val, val_proba).items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in _compute_metrics(y_test, test_proba).items():
            mlflow.log_metric(f"test_{k}", v)

        # Cast int cols to float in the signature sample so MLflow's schema
        # tolerates missing values at inference time. Does NOT affect training.
        sig_sample = X_train.head(50).copy()
        int_cols = sig_sample.select_dtypes(include=["int", "int64", "int32"]).columns
        sig_sample[int_cols] = sig_sample[int_cols].astype("float64")
        signature = infer_signature(sig_sample, pipeline.predict_proba(sig_sample))

        mlflow.sklearn.log_model(
            pipeline, name="model", signature=signature
        )  # ← name=, not artifact_path=
        return run.info.run_id


if __name__ == "__main__":
    baseline()
