import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

try:
    from evidently import Report
except ImportError:
    from evidently.report import Report

try:
    from evidently import ColumnMapping
except ImportError:
    try:
        from evidently.legacy.pipeline.column_mapping import ColumnMapping
    except ImportError:
        from evidently.pipeline.column_mapping import ColumnMapping

try:
    from evidently.metric_preset import ClassificationPreset, DataDriftPreset, DataQualityPreset
except ImportError:
    from evidently.presets import ClassificationPreset, DataDriftPreset
    try:
        from evidently.presets import DataQualityPreset
    except ImportError:
        from evidently.presets import DataSummaryPreset as DataQualityPreset

@dataclass
class MonitoringSummary:
    data_drift_score: float = 0.0
    missing_values_share: float = 0.0
    data_quality_issues_rate: float = 0.0
    anomaly_rate: float = 0.0
    output_drift_score: float = 0.0
    prediction_positive_rate: float = 0.0
    reference_positive_rate: float = 0.0
    model_f1_score: float = 0.0
    missing_image_file_rate: float = 0.0
    incorrect_image_file_rate: float = 0.0
    invalid_categories_rate: float = 0.0
    missing_categories_rate: float = 0.0
    outliers_rate: float = 0.0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _find_number_by_keys(payload: Any, keys: set[str]) -> float | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys and _is_number(value):
                return float(value)
            found = _find_number_by_keys(value, keys)
            if found is not None:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _find_number_by_keys(item, keys)
            if found is not None:
                return found
    return None


def _binary_f1_score(df: pd.DataFrame) -> float:
    if "prediction" not in df.columns or "target" not in df.columns:
        return 0.0
    eval_df = df.dropna(subset=["prediction", "target"]).copy()
    if eval_df.empty:
        return 0.0

    prediction = eval_df["prediction"].astype(int)
    target = eval_df["target"].astype(int)

    tp = int(((prediction == 1) & (target == 1)).sum())
    fp = int(((prediction == 1) & (target == 0)).sum())
    fn = int(((prediction == 0) & (target == 1)).sum())
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def _safe_rate(mask_count: int, total_count: int) -> float:
    if total_count <= 0:
        return 0.0
    return mask_count / total_count


def _outlier_rate_iqr(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0
    numeric = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(numeric) < 4:
        return 0.0
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    outliers = ((numeric < lower) | (numeric > upper)).sum()
    return float(outliers / len(numeric))


def _data_quality_issues_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0

    issue_mask = pd.Series(False, index=df.index)
    if "status" in df.columns:
        issue_mask = issue_mask | (df["status"].fillna("error") != "ok")
    if "category" in df.columns:
        issue_mask = issue_mask | (df["category"].isna()) | (df["category"].astype(str).str.strip() == "")
    if "filename" in df.columns:
        issue_mask = issue_mask | (df["filename"].isna()) | (df["filename"].astype(str).str.strip() == "")
    if "file_size_bytes" in df.columns:
        size = pd.to_numeric(df["file_size_bytes"], errors="coerce").fillna(0)
        issue_mask = issue_mask | (size <= 0)

    return float(issue_mask.mean())


def run_evidently_reports(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_html: Path,
    output_json: Path,
    monitoring_mode: str = "full",
    enable_detailed_metrics: bool = True,
    enable_label_metrics: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Run Evidently presets and return summary values plus raw report dict."""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if reference_df.empty or current_df.empty:
        summary = MonitoringSummary().__dict__
        payload: Dict[str, Any] = {"summary": summary, "reason": "missing_reference_or_current_data"}
        output_html.write_text("<html><body><h3>No data available for Evidently run.</h3></body></html>", encoding="utf-8")
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return summary, payload

    metrics = [DataDriftPreset(), DataQualityPreset()]
    current_success_df = current_df.copy()
    if "status" in current_success_df.columns:
        current_success_df = current_success_df[current_success_df["status"] == "ok"]

    reference_success_df = reference_df.copy()
    if "status" in reference_success_df.columns:
        reference_success_df = reference_success_df[reference_success_df["status"] == "ok"]

    if current_success_df.empty or reference_success_df.empty:
        summary = MonitoringSummary().__dict__
        payload = {"summary": summary, "reason": "missing_successful_reference_or_current_data"}
        output_html.write_text("<html><body><h3>No successful data available for Evidently run.</h3></body></html>", encoding="utf-8")
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return summary, payload

    if (
        enable_label_metrics
        and {"prediction", "target"}.issubset(reference_success_df.columns)
        and {"prediction", "target"}.issubset(current_success_df.columns)
    ):
        metrics.append(ClassificationPreset())

    column_mapping = ColumnMapping(prediction="prediction", target="target")
    report = Report(metrics=metrics)
    report.run(reference_data=reference_success_df, current_data=current_success_df, column_mapping=column_mapping)
    report.save_html(str(output_html))

    report_payload: Dict[str, Any] = report.as_dict()
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    data_drift_score = _find_number_by_keys(
        report_payload,
        {"share_of_drifted_columns", "drift_share", "dataset_drift_score"},
    )
    missing_values_share = _find_number_by_keys(
        report_payload,
        {"current_share_of_missing_values", "share_of_missing_values", "missing_values_share"},
    )
    summary = MonitoringSummary(
        data_drift_score=data_drift_score if data_drift_score is not None else 0.0,
        missing_values_share=(
            missing_values_share
            if missing_values_share is not None
            else float(current_success_df.isna().mean().mean())
        ),
        data_quality_issues_rate=_data_quality_issues_rate(current_df),
        anomaly_rate=(float(current_success_df["prediction"].mean()) if "prediction" in current_success_df.columns else 0.0),
        prediction_positive_rate=(float(current_success_df["prediction"].mean()) if "prediction" in current_success_df.columns else 0.0),
        reference_positive_rate=(float(reference_success_df["prediction"].mean()) if "prediction" in reference_success_df.columns else 0.0),
        model_f1_score=(_binary_f1_score(current_success_df) if enable_label_metrics else 0.0),
    )

    if enable_detailed_metrics:
        summary.missing_image_file_rate = _safe_rate(
            int((current_df.get("error_type") == "missing_image_file").sum()) if "error_type" in current_df.columns else 0,
            len(current_df),
        )
        summary.incorrect_image_file_rate = _safe_rate(
            int((current_df.get("error_type") == "incorrect_image_file").sum()) if "error_type" in current_df.columns else 0,
            len(current_df),
        )
        summary.invalid_categories_rate = _safe_rate(
            int((current_df.get("error_type") == "invalid_category").sum()) if "error_type" in current_df.columns else 0,
            len(current_df),
        )
        summary.missing_categories_rate = _safe_rate(
            int((current_df.get("error_type") == "missing_category").sum()) if "error_type" in current_df.columns else 0,
            len(current_df),
        )
        summary.outliers_rate = _outlier_rate_iqr(current_success_df, "file_size_bytes")

    summary.output_drift_score = abs(summary.anomaly_rate - summary.reference_positive_rate)
    return summary.__dict__, report_payload
