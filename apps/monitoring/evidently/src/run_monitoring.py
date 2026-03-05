import json
import logging
import os
from pathlib import Path
from time import sleep

from prometheus_client import start_http_server

from src.datasets import load_dataset, prepare_prediction_dataframe
from src.export_prom_metrics import publish_summary
from src.reports import run_evidently_reports

logger = logging.getLogger("evidently_monitor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
logger.propagate = False


def _is_enabled(var_name: str, default: bool) -> bool:
    return os.getenv(var_name, str(default).lower()).strip().lower() in {"1", "true", "yes", "on"}


def _safe_mean_prediction(df) -> float:
    if "prediction" not in df.columns:
        return 0.0
    values = df["prediction"].dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def _row_counts(reference_df, current_df) -> dict[str, int]:
    reference_count = int(len(reference_df))
    current_count = int(len(current_df))
    current_ok_count = int(len(current_df[current_df["status"] == "ok"])) if "status" in current_df.columns else current_count
    return {
        "reference": reference_count,
        "current": current_count,
        "current_ok": current_ok_count,
    }


def _emit_cycle_log(row_counts: dict[str, int], summary: dict, report_payload: dict) -> None:
    reason = report_payload.get("reason")
    metrics = {
        "anomaly_rate": summary.get("anomaly_rate", 0.0),
        "output_drift_score": summary.get("output_drift_score", 0.0),
        "data_quality_issues_rate": summary.get("data_quality_issues_rate", 0.0),
        "data_drift_score": summary.get("data_drift_score", 0.0),
    }
    payload = {
        "event": "monitoring_cycle_completed",
        "rows": row_counts,
        "metrics": metrics,
    }
    if reason:
        payload["reason"] = reason
    logger.info(json.dumps(payload, sort_keys=True))


def run_once() -> None:
    reports_root = Path("/app/reports/monitoring")
    reference_root = Path("/app/references/monitoring")
    reference_path = Path(os.getenv("EVIDENTLY_REFERENCE_PATH", str(reference_root / "evidently" / "reference.parquet")))
    current_path = Path(
        os.getenv("EVIDENTLY_CURRENT_PATH", str(reports_root / "inference_events" / "events.jsonl"))
    )
    html_output = reports_root / "evidently" / "html" / "latest_report.html"
    json_output = reports_root / "evidently" / "json" / "latest_report.json"

    reference_df = prepare_prediction_dataframe(load_dataset(reference_path))
    current_df = prepare_prediction_dataframe(load_dataset(current_path))
    row_counts = _row_counts(reference_df, current_df)
    monitoring_mode = os.getenv("MONITORING_MODE", "full").strip().lower()
    enable_detailed_metrics = _is_enabled(
        "ENABLE_DETAILED_DATA_QUALITY_METRICS",
        default=(monitoring_mode != "minimal"),
    )
    enable_label_metrics = _is_enabled(
        "ENABLE_LABEL_BASED_METRICS",
        default=(monitoring_mode != "minimal"),
    )
    summary, report_payload = run_evidently_reports(
        reference_df,
        current_df,
        html_output,
        json_output,
        monitoring_mode=monitoring_mode,
        enable_detailed_metrics=enable_detailed_metrics,
        enable_label_metrics=enable_label_metrics,
    )

    base_labels = {
        "model_name": os.getenv("MLFLOW_MODEL_NAME", "unknown"),
        "model_version": os.getenv("MLFLOW_MODEL_VERSION", "unknown"),
        "run_id": os.getenv("MLFLOW_RUN_ID", "unknown"),
    }
    publish_summary({"category": "all", **base_labels}, summary)

    categories: list[str] = []
    if "category" in current_df.columns:
        categories = sorted(
            {str(c).strip() for c in current_df["category"].dropna().tolist() if str(c).strip()}
        )

    for category in categories:
        current_slice = current_df[current_df["category"] == category]
        reference_slice = reference_df[reference_df["category"] == category] if "category" in reference_df.columns else reference_df

        current_ok = current_slice[current_slice["status"] == "ok"] if "status" in current_slice.columns else current_slice
        reference_ok = reference_slice[reference_slice["status"] == "ok"] if "status" in reference_slice.columns else reference_slice

        category_summary = dict(summary)
        category_summary["anomaly_rate"] = _safe_mean_prediction(current_ok)
        category_summary["prediction_positive_rate"] = category_summary["anomaly_rate"]
        category_summary["reference_positive_rate"] = _safe_mean_prediction(reference_ok)
        category_summary["output_drift_score"] = abs(
            category_summary["anomaly_rate"] - category_summary["reference_positive_rate"]
        )
        publish_summary({"category": category, **base_labels}, category_summary)

    out_dir = reports_root / "evidently" / "json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "latest_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": {"category": "all", **base_labels}, "summary": summary, "report": report_payload}, f, indent=2)

    _emit_cycle_log(row_counts=row_counts, summary=summary, report_payload=report_payload)


def main() -> None:
    port = int(os.getenv("PROM_EXPORT_PORT", "9108"))
    interval = int(os.getenv("EVIDENTLY_INTERVAL_SECONDS", "300"))

    start_http_server(port)
    while True:
        run_once()
        sleep(interval)


if __name__ == "__main__":
    main()
