from dataclasses import dataclass


@dataclass
class LabelEvent:
    request_id: str
    target: int


# Placeholder for delayed-label ingestion and merge into inference events.
def ingest_label(_event: LabelEvent) -> None:
    return None
