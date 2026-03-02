from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json


@dataclass
class InferenceEvent:
    timestamp: str
    request_id: str
    category: str | None
    filename: str
    file_size_bytes: int
    defective: int | None = None
    status: str = "ok"
    error_type: str | None = None
    model_name: str = "unknown"
    model_version: str = "unknown"
    run_id: str = "unknown"
    target: int | None = None


def build_event(**kwargs) -> InferenceEvent:
    return InferenceEvent(timestamp=datetime.now(timezone.utc).isoformat(), **kwargs)


def append_event_jsonl(event: InferenceEvent, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(event)) + "\n")
