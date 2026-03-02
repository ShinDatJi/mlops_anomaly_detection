""" This script is a one time script to generate the database
for the evidently monitoring reference.
This snippet looks at the MVtec database and will take only the train/good images
of each category in order to fill the database.
The columns of the database will be appended as described in the class InferenceEvent
in apps/prediction/events.py
  """

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

mvtec_root = Path("/mnt/d/Bootcamp/Project/mvtec_anomaly_detection")
if not mvtec_root.exists():
    raise FileNotFoundError(f"Path not found: {mvtec_root}")

out = Path("reports/monitoring/evidently/reference.parquet")

rows = []
ts = datetime.now(timezone.utc).isoformat()

for category_dir in sorted([p for p in mvtec_root.iterdir() if p.is_dir()]):
    for img in (category_dir / "train" / "good").glob("*"):
        if img.is_file():
            rows.append({
                "timestamp": ts,
                "request_id": f"ref-{category_dir.name}-{img.stem}",
                "category": category_dir.name,
                "filename": img.name,
                "file_size_bytes": img.stat().st_size,
                "status": "ok",
                "error_type": None,
                "defective": 0,      # for your normalization -> prediction
                "prediction": 0,     # explicit is fine
                "target": 0,         # label
                "model_name": "reference-mvtec",
                "model_version": "dataset",
                "run_id": "reference",
            })

df = pd.DataFrame(rows)
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)
print(f"Wrote {len(df)} rows to {out}")
