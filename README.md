# Amazon Order Return Risk - MLOps Pipeline

MLOps pipeline for predicting returns and cancellations of various items on Amazon orders.

The purpose of this project is to demonstrate data versioning, experiment tracking, automated model retraining, model serving, and data drift monitoring.

## Problem To Address
Returns and cancellations of items cost about 20-30% of profit margins in e-commerce. This project will create a scoring system which will flag high risk orders at checkout so that the business can adjust accordingly.


## Stack
- **Data versioning:** DVC
- **Experiment tracking & model registry:** MLflow
- **Orchestration:** Prefect
- **Serving:** FastAPI + Docker
- **Monitoring:** Prometheus, Grafana, Evidently AI
- **CI/CD:** GitHub Actions
