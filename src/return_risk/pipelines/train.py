"""End-to-end training pipeline with MLflow tracking and model registry.

Run from project root:
    uv run python -m return_risk.pipelines.train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from return_risk.data.loader import drop_leakage_and_ids, load_config, load_raw
from return_risk.data.splits import make_splits
from return_risk.features.build_features import build_full_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _set_mlflow(cfg: dict) -> None:
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    if tracking_uri.startswith("sqlite:///"):
        # Resolve relative SQLite paths against the project root
        rel = tracking_uri.removeprefix("sqlite:///")
        absolute = PROJECT_ROOT / rel
        tracking_uri = f"sqlite:///{absolute}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])


def _compute_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision_at_0.5": precision_score(y_true, preds, zero_division=0),
        "recall_at_0.5": recall_score(y_true, preds, zero_division=0),
        "f1_at_0.5": f1_score(y_true, preds, zero_division=0),
    }


def _operating_point_recall_at_precision(
    y_true: pd.Series, proba: np.ndarray, target_precision: float = 0.5
) -> dict:
    """Find recall achievable at a given precision floor (a real business KPI)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    valid = precisions[:-1] >= target_precision
    if not valid.any():
        return {"recall_at_precision_floor": 0.0, "threshold_at_precision_floor": 1.0}
    idx = np.argmax(recalls[:-1] * valid)
    return {
        "recall_at_precision_floor": float(recalls[idx]),
        "threshold_at_precision_floor": float(thresholds[idx]),
    }


def train(run_name: str | None = None) -> str:
    cfg = load_config()
    _set_mlflow(cfg)

    print("[1/5] Loading and validating data…")
    df = load_raw(validate=True)
    df = drop_leakage_and_ids(df)

    target_col = cfg["data"]["target_column"]
    print(f"      rows={len(df):,}  positive_rate={df[target_col].mean():.4f}")

    print("[2/5] Splitting…")
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(
        df,
        target_col=target_col,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["validation_size"],
        random_state=cfg["project"]["random_seed"],
    )
    print(f"      train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    print("[3/5] Building pipeline…")
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)
    print(f"      scale_pos_weight={scale_pos_weight:.3f}")

    estimator = XGBClassifier(
        **cfg["model"]["params"],
        scale_pos_weight=scale_pos_weight,
        random_state=cfg["project"]["random_seed"],
        n_jobs=-1,
        tree_method="hist",
    )

    pipeline = build_full_pipeline(
        numeric_cols=cfg["features"]["numeric_cols"],
        low_card_cols=cfg["features"]["low_cardinality_cat_cols"],
        high_card_cols=cfg["features"]["high_cardinality_cat_cols"],
        datetime_cols=cfg["features"]["datetime_cols"],
        estimator=estimator,
    )

    print("[4/5] Training + logging to MLflow…")
    with mlflow.start_run(run_name=run_name) as run:
        # Log params
        mlflow.log_params(cfg["model"]["params"])
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("positive_rate_train", round(float(y_train.mean()), 4))

        # Fit
        pipeline.fit(X_train, y_train)

        # Evaluate
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        val_metrics = _compute_metrics(y_val, val_proba)
        test_metrics = _compute_metrics(y_test, test_proba)
        op_point = _operating_point_recall_at_precision(y_test, test_proba, 0.5)

        # Log metrics with prefixes so MLflow groups them
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        for k, v in op_point.items():
            mlflow.log_metric(f"test_{k}", v)

        print(f"      val ROC-AUC={val_metrics['roc_auc']:.4f}  PR-AUC={val_metrics['pr_auc']:.4f}")
        print(
            f"      test ROC-AUC={test_metrics['roc_auc']:.4f}  PR-AUC={test_metrics['pr_auc']:.4f}"
        )

        # Log model with signature so the API knows the schema.
        # Cast int cols to float so the signature tolerates missing values at inference.
        sig_sample = X_train.head(50).copy()
        int_cols = sig_sample.select_dtypes(include=["int", "int64", "int32"]).columns
        sig_sample[int_cols] = sig_sample[int_cols].astype("float64")
        signature = infer_signature(sig_sample, pipeline.predict_proba(sig_sample))
        registered_name = cfg["mlflow"]["registered_model_name"]

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            registered_model_name=registered_name,
        )

        run_id = run.info.run_id
        print(f"[5/5] Logged + registered. run_id={run_id}")

    # Auto-promote to Staging if thresholds are met. Never auto-promote to Production.
    _maybe_promote_to_staging(cfg, registered_name, test_metrics)
    return run_id


def _maybe_promote_to_staging(cfg: dict, model_name: str, test_metrics: dict) -> None:
    auc_floor = cfg["monitoring"]["performance_threshold_auc"]
    pr_floor = cfg["monitoring"]["performance_threshold_pr_auc"]

    passes = test_metrics["roc_auc"] >= auc_floor and test_metrics["pr_auc"] >= pr_floor

    client = mlflow.tracking.MlflowClient()
    latest_versions = client.search_model_versions(f"name='{model_name}'")
    latest = max(latest_versions, key=lambda mv: int(mv.version))

    if passes:
        client.set_model_version_tag(model_name, latest.version, "stage", "Staging")
        client.set_model_version_tag(model_name, latest.version, "auto_promoted", "true")
        print(
            f"[promote] v{latest.version} → Staging "
            f"(roc_auc={test_metrics['roc_auc']:.4f} ≥ {auc_floor}, "
            f"pr_auc={test_metrics['pr_auc']:.4f} ≥ {pr_floor})"
        )
    else:
        client.set_model_version_tag(model_name, latest.version, "stage", "Rejected")
        print(
            f"[promote] v{latest.version} REJECTED — did not meet thresholds "
            f"(roc_auc={test_metrics['roc_auc']:.4f} vs {auc_floor}, "
            f"pr_auc={test_metrics['pr_auc']:.4f} vs {pr_floor})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None, help="Optional MLflow run name")
    args = parser.parse_args()
    train(run_name=args.run_name)
