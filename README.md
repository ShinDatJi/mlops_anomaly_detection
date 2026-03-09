# MLOps MVTec Anomaly Detection

MVTec dataset: <https://www.mvtec.com/company/research/datasets/mvtec-ad>

## Project Overview

![project overview](/demo/figures/architecture.png)

* Data: public MVTec AD dataset (industrial anomaly detection)
* Modeling: CNN for image classification with patching
* Prediction: FastAPI service for real-time inference and monitoring

## Architecture

![global architecture](/demo/figures/introduction.png)

## Prediction API

The prediction service is maintained only under `apps/prediction`.
Root-level API duplicates were removed.

Quick start from repository root:

```bash
make init-prediction
make build-prediction
make run-prediction
```

API endpoints:

- `GET http://localhost:8000/status`
- `POST http://localhost:8000/predict/{category}` with multipart file field `image`

Monitoring modes:
- `minimal` (default): core operational metrics only (`anomaly_rate`, `output_drift_score`, drift, data quality issues, API traffic/performance)
- `full`: detailed data quality/error metrics and label-based metrics

For app-specific details, see `apps/prediction/README.md`.

## Project Organization

```text
    ├── LICENSE
    ├── README.md
    ├── Makefile
    ├── apps
    │   ├── data
    │   │   ├── default.env
    │   │   ├── docker-compose.yml
    │   │   ├── Dockerfile
    │   │   ├── pyproject.toml
    │   │   └── src
    │   ├── modeling
    │   │   ├── default.env
    │   │   ├── docker-compose.yml
    │   │   ├── Dockerfile
    │   │   ├── pyproject.toml
    │   │   └── src
    │   └── prediction
    │       ├── default.env
    │       ├── docker-compose.yml
    │       ├── Dockerfile
    │       ├── pyproject.toml
    │       ├── uv.lock
    │       └── src
    ├── data
    │   ├── processed
    │   └── raw
    ├── models
    ├── references
    └── reports
```
