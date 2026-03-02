from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "prediction_requests_total",
    "Total prediction API requests",
    ["method", "path", "status"],
)
REQUEST_DURATION_SECONDS = Histogram(
    "prediction_request_duration_seconds",
    "Prediction API request latency",
    ["method", "path"],
)
PREDICTIONS_TOTAL = Counter(
    "prediction_inference_total",
    "Total model predictions",
    ["category", "defective", "model_name", "model_version", "run_id"],
)
