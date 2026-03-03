import os
from typing import Mapping

from prometheus_client import Gauge


def _is_enabled(var_name: str, default: bool) -> bool:
    return os.getenv(var_name, str(default).lower()).strip().lower() in {"1", "true", "yes", "on"}


MONITORING_MODE = os.getenv("MONITORING_MODE", "full").strip().lower()
ENABLE_DETAILED_METRICS = _is_enabled(
    "ENABLE_DETAILED_DATA_QUALITY_METRICS",
    default=(MONITORING_MODE != "minimal"),
)
ENABLE_LABEL_METRICS = _is_enabled(
    "ENABLE_LABEL_BASED_METRICS",
    default=(MONITORING_MODE != "minimal"),
)

DATA_DRIFT = Gauge(
    "evidently_data_drift_score",
    "Evidently data drift score",
    ["category", "model_name", "model_version", "run_id"],
)
MISSING_SHARE = Gauge(
    "evidently_missing_values_share",
    "Evidently missing values share",
    ["category", "model_name", "model_version", "run_id"],
)
DATA_QUALITY_ISSUES_RATE = Gauge(
    "evidently_data_quality_issues_rate",
    "Share of records with data quality issues",
    ["category", "model_name", "model_version", "run_id"],
)
ANOMALY_RATE = Gauge(
    "evidently_anomaly_rate",
    "Anomaly rate (anomalies / total predictions)",
    ["category", "model_name", "model_version", "run_id"],
)
OUTPUT_DRIFT_SCORE = Gauge(
    "evidently_output_drift_score",
    "Absolute output drift based on anomaly rate shift",
    ["category", "model_name", "model_version", "run_id"],
)
PRED_POS_RATE = Gauge(
    "evidently_prediction_positive_rate",
    "Current positive prediction rate",
    ["category", "model_name", "model_version", "run_id"],
)
REF_POS_RATE = Gauge(
    "evidently_reference_positive_rate",
    "Reference positive prediction rate",
    ["category", "model_name", "model_version", "run_id"],
)
F1_SCORE = Gauge(
    "evidently_model_f1_score",
    "Observed model F1 score",
    ["category", "model_name", "model_version", "run_id"],
)

if ENABLE_DETAILED_METRICS:
    MISSING_IMAGE_FILE_RATE = Gauge(
        "evidently_missing_image_file_rate",
        "Share of requests with missing/empty image file",
        ["category", "model_name", "model_version", "run_id"],
    )
    INCORRECT_IMAGE_FILE_RATE = Gauge(
        "evidently_incorrect_image_file_rate",
        "Share of requests with incorrect image files",
        ["category", "model_name", "model_version", "run_id"],
    )
    INVALID_CATEGORIES_RATE = Gauge(
        "evidently_invalid_categories_rate",
        "Share of requests with invalid categories",
        ["category", "model_name", "model_version", "run_id"],
    )
    MISSING_CATEGORIES_RATE = Gauge(
        "evidently_missing_categories_rate",
        "Share of requests with missing categories",
        ["category", "model_name", "model_version", "run_id"],
    )
    OUTLIERS_RATE = Gauge(
        "evidently_outliers_rate",
        "Share of outliers in numeric image metadata",
        ["category", "model_name", "model_version", "run_id"],
    )


def publish_summary(labels: Mapping[str, str], summary: Mapping[str, float]) -> None:
    DATA_DRIFT.labels(**labels).set(summary.get("data_drift_score", 0.0))
    MISSING_SHARE.labels(**labels).set(summary.get("missing_values_share", 0.0))
    DATA_QUALITY_ISSUES_RATE.labels(**labels).set(summary.get("data_quality_issues_rate", 0.0))
    ANOMALY_RATE.labels(**labels).set(summary.get("anomaly_rate", 0.0))
    OUTPUT_DRIFT_SCORE.labels(**labels).set(summary.get("output_drift_score", 0.0))
    PRED_POS_RATE.labels(**labels).set(summary.get("prediction_positive_rate", 0.0))
    REF_POS_RATE.labels(**labels).set(summary.get("reference_positive_rate", 0.0))
    if ENABLE_LABEL_METRICS:
        F1_SCORE.labels(**labels).set(summary.get("model_f1_score", 0.0))
    if ENABLE_DETAILED_METRICS:
        MISSING_IMAGE_FILE_RATE.labels(**labels).set(summary.get("missing_image_file_rate", 0.0))
        INCORRECT_IMAGE_FILE_RATE.labels(**labels).set(summary.get("incorrect_image_file_rate", 0.0))
        INVALID_CATEGORIES_RATE.labels(**labels).set(summary.get("invalid_categories_rate", 0.0))
        MISSING_CATEGORIES_RATE.labels(**labels).set(summary.get("missing_categories_rate", 0.0))
        OUTLIERS_RATE.labels(**labels).set(summary.get("outliers_rate", 0.0))
