# Monitoring App

Monitoring stack skeleton for:
- API traffic and performance
- model outputs
- input data behavior
- model performance

Components:
- Prometheus for metrics collection
- Grafana for dashboards and alerts
- Evidently batch job/exporter for data and model monitoring
- Prediction API `/metrics` + inference events feed this stack

## Setup

```bash
cp apps/monitoring/default.env apps/monitoring/.env
```

## Start

From repository root:

```bash
docker compose \
  -f apps/monitoring/docker-compose.yml \
  --env-file .env \
  --env-file apps/monitoring/.env \
  --project-directory ./ \
  up -d --build
```

## Notes

- Evidently reads:
  - reference data: `reports/monitoring/evidently/reference.parquet`
  - current data: `reports/monitoring/inference_events/events.jsonl`
- Reports are written to:
  - `reports/monitoring/evidently/html/latest_report.html`
  - `reports/monitoring/evidently/json/latest_report.json`
- Metrics labels include model metadata fields (`model_name`, `model_version`, `run_id`) for MLflow alignment.

## Modes

- `MONITORING_MODE=minimal`
  - keeps: `evidently_anomaly_rate`, `evidently_output_drift_score`, `evidently_data_quality_issues_rate`, `evidently_data_drift_score`
  - keeps API traffic metrics from the prediction service
  - provisioned dashboards: `grafana/dashboards/minimal/` (`API Traffic & Performance`, `Model Performance`)
  - designed as default operational mode with low noise
- `MONITORING_MODE=full`
  - enables detailed data quality/error metrics and optional label-based metrics
  - additional dashboards can be kept under `grafana/dashboards/full/`

Optional flags:
- `ENABLE_DETAILED_DATA_QUALITY_METRICS=true|false`
- `ENABLE_LABEL_BASED_METRICS=true|false`
