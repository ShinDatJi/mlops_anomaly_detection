import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


@dataclass
class MonitoringSummary:
    data_drift_score: float = 0.0
    missing_values_share: float = 0.0
    data_quality_issues_rate: float = 0.0
    data_quality_issues_count: float = 0.0
    corrupted_images_count: float = 0.0
    unexpected_categories_count: float = 0.0
    no_category_count: float = 0.0
    out_of_range_or_invalid_image_count: float = 0.0
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


def _safe_text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype="object")
    return df[column]


def _safe_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _data_quality_issue_counts(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict[str, int]:
    if current_df.empty:
        return {
            "data_quality_issues_count": 0,
            "corrupted_images_count": 0,
            "unexpected_categories_count": 0,
            "no_category_count": 0,
            "out_of_range_or_invalid_image_count": 0,
        }

    error_type = _safe_text_series(current_df, "error_type")
    category = _safe_text_series(current_df, "category")
    filename = _safe_text_series(current_df, "filename")
    file_size = _safe_numeric_series(current_df, "file_size_bytes")
    prediction = _safe_numeric_series(current_df, "prediction")

    normalized_category = category.astype("string").str.strip().str.lower()
    missing_category_mask = normalized_category.isna() | (normalized_category == "") | (error_type == "missing_category")

    expected_categories: set[str] = set()
    if "category" in reference_df.columns:
        expected_categories = {
            str(value).strip().lower()
            for value in reference_df["category"].dropna().tolist()
            if str(value).strip()
        }
    if expected_categories:
        unexpected_category_mask = (~missing_category_mask) & (~normalized_category.isin(expected_categories))
    else:
        unexpected_category_mask = pd.Series(False, index=current_df.index)
    unexpected_category_mask = unexpected_category_mask | (error_type == "invalid_category")

    corrupted_mask = error_type == "incorrect_image_file"

    normalized_filename = filename.astype("string").str.strip()
    missing_or_empty_filename = normalized_filename.isna() | (normalized_filename == "")
    invalid_prediction_mask = prediction.notna() & (~prediction.isin([0, 1]))
    out_of_range_or_invalid_image_mask = (
        (error_type == "missing_image_file")
        | (error_type == "prediction_error")
        | ((file_size <= 0) & file_size.notna())
        | missing_or_empty_filename
        | invalid_prediction_mask
    )

    any_issue_mask = (
        corrupted_mask
        | unexpected_category_mask
        | missing_category_mask
        | out_of_range_or_invalid_image_mask
    )

    return {
        "data_quality_issues_count": int(any_issue_mask.sum()),
        "corrupted_images_count": int(corrupted_mask.sum()),
        "unexpected_categories_count": int(unexpected_category_mask.sum()),
        "no_category_count": int(missing_category_mask.sum()),
        "out_of_range_or_invalid_image_count": int(out_of_range_or_invalid_image_mask.sum()),
    }


def _success_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "status" not in df.columns:
        return df.copy()
    return df[df["status"] == "ok"].copy()


def _mean_prediction(df: pd.DataFrame) -> float:
    if "prediction" not in df.columns:
        return 0.0
    values = pd.to_numeric(df["prediction"], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def _simple_data_drift_score(reference_success_df: pd.DataFrame, current_success_df: pd.DataFrame) -> float:
    """
    Robust, deterministic drift proxy without Evidently runtime dependencies.
    Uses max of output drift and normalized file-size mean drift (when available).
    """
    ref_rate = _mean_prediction(reference_success_df)
    cur_rate = _mean_prediction(current_success_df)
    output_drift = abs(cur_rate - ref_rate)

    if "file_size_bytes" not in reference_success_df.columns or "file_size_bytes" not in current_success_df.columns:
        return output_drift

    ref_sizes = pd.to_numeric(reference_success_df["file_size_bytes"], errors="coerce").dropna()
    cur_sizes = pd.to_numeric(current_success_df["file_size_bytes"], errors="coerce").dropna()
    if ref_sizes.empty or cur_sizes.empty:
        return output_drift

    ref_mean = float(ref_sizes.mean())
    cur_mean = float(cur_sizes.mean())
    denom = max(abs(ref_mean), 1.0)
    size_drift = min(1.0, abs(cur_mean - ref_mean) / denom)

    return max(output_drift, size_drift)


def run_evidently_reports(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_html: Path,
    output_json: Path,
    monitoring_mode: str = "full",
    enable_detailed_metrics: bool = True,
    enable_label_metrics: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute monitoring summary in a stable way using pandas only.
    Advanced metrics remain available via flags.
    """
    del monitoring_mode  # Reserved for future behavior split.

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if reference_df.empty or current_df.empty:
        summary = MonitoringSummary().__dict__
        payload: Dict[str, Any] = {"summary": summary, "reason": "missing_reference_or_current_data"}
        output_html.write_text("<html><body><h3>No data available for monitoring run.</h3></body></html>", encoding="utf-8")
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return summary, payload

    current_success_df = _success_df(current_df)
    reference_success_df = _success_df(reference_df)

    if current_success_df.empty or reference_success_df.empty:
        summary = MonitoringSummary().__dict__
        payload = {"summary": summary, "reason": "missing_successful_reference_or_current_data"}
        output_html.write_text("<html><body><h3>No successful data available for monitoring run.</h3></body></html>", encoding="utf-8")
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return summary, payload

    anomaly_rate = _mean_prediction(current_success_df)
    reference_positive_rate = _mean_prediction(reference_success_df)
    output_drift_score = abs(anomaly_rate - reference_positive_rate)
    dq_counts = _data_quality_issue_counts(reference_df=reference_df, current_df=current_df)
    dq_total_count = dq_counts["data_quality_issues_count"]
    dq_total_rate = _safe_rate(dq_total_count, len(current_df))

    summary = MonitoringSummary(
        data_drift_score=_simple_data_drift_score(reference_success_df, current_success_df),
        missing_values_share=float(current_success_df.isna().mean().mean()),
        data_quality_issues_rate=dq_total_rate,
        data_quality_issues_count=float(dq_total_count),
        corrupted_images_count=float(dq_counts["corrupted_images_count"]),
        unexpected_categories_count=float(dq_counts["unexpected_categories_count"]),
        no_category_count=float(dq_counts["no_category_count"]),
        out_of_range_or_invalid_image_count=float(dq_counts["out_of_range_or_invalid_image_count"]),
        anomaly_rate=anomaly_rate,
        output_drift_score=output_drift_score,
        prediction_positive_rate=anomaly_rate,
        reference_positive_rate=reference_positive_rate,
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

    issue_type_counts = {
        "corrupted_images": dq_counts["corrupted_images_count"],
        "unexpected_categories": dq_counts["unexpected_categories_count"],
        "no_category_sent": dq_counts["no_category_count"],
        "out_of_range_or_invalid_image": dq_counts["out_of_range_or_invalid_image_count"],
    }
    detected_issue_types = [issue for issue, count in issue_type_counts.items() if count > 0]

    report_payload: Dict[str, Any] = {
        "summary": summary.__dict__,
        "data_quality_issues": {
            "count": dq_counts["data_quality_issues_count"],
            "types_detected": detected_issue_types,
            "type_counts": issue_type_counts,
        },
        "rows": {
            "reference": int(len(reference_df)),
            "reference_success": int(len(reference_success_df)),
            "current": int(len(current_df)),
            "current_success": int(len(current_success_df)),
        },
    }

    output_html.write_text(
        "<html><body><h3>Monitoring summary generated successfully.</h3></body></html>",
        encoding="utf-8",
    )
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    return summary.__dict__, report_payload
