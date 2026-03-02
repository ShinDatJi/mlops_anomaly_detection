from pathlib import Path

import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    """Load parquet or jsonl dataset."""
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.DataFrame()


def prepare_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize event columns for Evidently runs."""
    if df.empty:
        return df

    normalized = df.copy()
    if "defective" in normalized.columns and "prediction" not in normalized.columns:
        normalized["prediction"] = pd.to_numeric(normalized["defective"], errors="coerce")
    if "prediction" in normalized.columns:
        normalized["prediction"] = pd.to_numeric(normalized["prediction"], errors="coerce")
    if "target" in normalized.columns:
        normalized["target"] = pd.to_numeric(normalized["target"], errors="coerce")
    if "category" in normalized.columns:
        normalized["category"] = normalized["category"].astype("string")
    return normalized
