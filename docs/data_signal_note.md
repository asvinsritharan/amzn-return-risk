# Signal Modification

## Observation
The `sharmajicoder/amazon-e-commerce` Kaggle dataset is synthetic. The `is_returned` label was assigned independently of all features, so both a baseline logistic regression and XGBoost score around ROC-AUC 0.56 — coin-flip territory.

## Purpose
The point of this project is the MLOps pipeline, not finding a novel result on a dataset with no signal. So I rebuilt the target using correlations that show up reliably in e-commerce returns research: shipping delays, price, product category (Clothing returns more often than Beauty), seller and product ratings, and a small channel effect for mobile-app and cash-on-delivery orders. That logic lives in `scripts/inject_signal.py` with a fixed seed.

## Outcome
The result is that XGBoost now reaches ROC-AUC ≈ 0.85. Drift detection in Phase 5 has real baselines to compare against, and the training-registry-retraining loop actually does something meaningful.

## Unchanged
Everything else is the same. Schema validation still runs, DVC tracks every dataset revision, and the auto-promotion gate still rejects models that miss their thresholds.
