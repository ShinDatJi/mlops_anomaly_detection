# Evidently Monitoring Job

This service is a skeleton for batch monitoring and Prometheus metric export.

Current behavior:
- starts a metrics endpoint (default `:9108/metrics`)
- loads reference/current datasets from configured paths
- runs Evidently `DataDriftPreset`, `DataQualityPreset`, and `ClassificationPreset` (when labels are available)
- writes:
  - `reports/monitoring/evidently/html/latest_report.html`
  - `reports/monitoring/evidently/json/latest_report.json`
  - `reports/monitoring/evidently/json/latest_summary.json`
- exports Prometheus gauges with model metadata labels

Planned next step:
- source model metadata (`model_name`, `model_version`, `run_id`) directly from deployed model registry
