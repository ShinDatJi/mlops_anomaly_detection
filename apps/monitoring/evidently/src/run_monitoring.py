import json
import os
from pathlib import Path
from time import sleep

from prometheus_client import start_http_server

from src.datasets import load_dataset, prepare_prediction_dataframe
from src.export_prom_metrics import publish_summary
from src.reports import run_evidently_reports


def run_once() -> None:
    reports_root = Path("/app/reports/monitoring")
    reference_path = Path(os.getenv("EVIDENTLY_REFERENCE_PATH", str(reports_root / "evidently" / "reference.parquet")))
    current_path = Path(
        os.getenv("EVIDENTLY_CURRENT_PATH", str(reports_root / "inference_events" / "events.jsonl"))
    )
    html_output = reports_root / "evidently" / "html" / "latest_report.html"
    json_output = reports_root / "evidently" / "json" / "latest_report.json"

    reference_df = prepare_prediction_dataframe(load_dataset(reference_path))
    current_df = prepare_prediction_dataframe(load_dataset(current_path))
    summary, report_payload = run_evidently_reports(reference_df, current_df, html_output, json_output)

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
        category_html = reports_root / "evidently" / "html" / f"{category}_report.html"
        category_json = reports_root / "evidently" / "json" / f"{category}_report.json"
        category_summary, _ = run_evidently_reports(reference_slice, current_slice, category_html, category_json)
        publish_summary({"category": category, **base_labels}, category_summary)

    out_dir = reports_root / "evidently" / "json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "latest_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": {"category": "all", **base_labels}, "summary": summary, "report": report_payload}, f, indent=2)


def main() -> None:
    port = int(os.getenv("PROM_EXPORT_PORT", "9108"))
    interval = int(os.getenv("EVIDENTLY_INTERVAL_SECONDS", "300"))

    start_http_server(port)
    while True:
        run_once()
        sleep(interval)


if __name__ == "__main__":
    main()
