import logging
import os
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from prometheus_client import make_asgi_app

from src.monitoring.events import append_event_jsonl, build_event
from src.monitoring.metrics import PREDICTIONS_TOTAL, REQUEST_DURATION_SECONDS, REQUESTS_TOTAL
from src.predict import predict


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("prediction_api")
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


logger = configure_logger()
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
MONITORING_EVENTS_FILE = Path(
    os.getenv("MONITORING_EVENTS_FILE", "./reports/monitoring/inference_events/events.jsonl")
)
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "unknown")
MLFLOW_MODEL_VERSION = os.getenv("MLFLOW_MODEL_VERSION", "unknown")
MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID", "unknown")


class StatusResponse(BaseModel):
    status: str


class PredictResponse(BaseModel):
    defective: bool


app = FastAPI(
    title="Prediction API",
    description="FastAPI service for anomaly prediction by category.",
    version="0.1.0",
)
app.mount("/metrics", make_asgi_app())


def normalize_metrics_path(path: str) -> str:
    if path.startswith("/predict/"):
        return "/predict/{category}"
    return path


def write_inference_event(
    request_id: str,
    category: str | None,
    filename: str | None,
    file_size_bytes: int,
    defective: int | None,
    status: str,
    error_type: str | None = None,
) -> None:
    try:
        event = build_event(
            request_id=request_id,
            category=category,
            filename=filename or "unknown",
            file_size_bytes=file_size_bytes,
            defective=defective,
            status=status,
            error_type=error_type,
            model_name=MLFLOW_MODEL_NAME,
            model_version=MLFLOW_MODEL_VERSION,
            run_id=MLFLOW_RUN_ID,
        )
        append_event_jsonl(event, MONITORING_EVENTS_FILE)
    except Exception:
        logger.warning("event_logging_failed request_id=%s", request_id, exc_info=True)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    metrics_path = normalize_metrics_path(request.url.path)
    client = request.client.host if request.client else "unknown"
    logger.info("request_started method=%s path=%s client=%s", request.method, request.url.path, client)
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request_failed method=%s path=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise
    finally:
        duration_seconds = time.perf_counter() - start
        REQUEST_DURATION_SECONDS.labels(method=request.method, path=metrics_path).observe(duration_seconds)
        REQUESTS_TOTAL.labels(method=request.method, path=metrics_path, status=str(status_code)).inc()

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request_completed method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/status", response_model=StatusResponse, tags=["system"], summary="Health check")
def status() -> StatusResponse:
    return StatusResponse(status="ok")


@app.post(
    "/predict/{category}",
    response_model=PredictResponse,
    tags=["prediction"],
    summary="Predict if an image is defective",
)
async def predict_category(category: str, request: Request, image: UploadFile = File(...)) -> PredictResponse:
    request_id = request.headers.get("x-request-id", str(uuid4()))
    category = (category or "").strip()
    filename = image.filename if image.filename else "unknown"

    if not category:
        logger.warning("missing_category filename=%s", filename)
        write_inference_event(
            request_id=request_id,
            category=None,
            filename=filename,
            file_size_bytes=0,
            defective=None,
            status="error",
            error_type="missing_category",
        )
        raise HTTPException(status_code=400, detail="Missing category")

    category_dir = MODELS_DIR / category
    if not category_dir.exists() or not category_dir.is_dir():
        logger.warning("unknown_category category=%s", category)
        write_inference_event(
            request_id=request_id,
            category=category,
            filename=filename,
            file_size_bytes=0,
            defective=None,
            status="error",
            error_type="invalid_category",
        )
        raise HTTPException(status_code=404, detail="Unknown category")

    image_bytes = await image.read()
    if not image_bytes:
        logger.warning("empty_upload category=%s filename=%s", category, image.filename)
        write_inference_event(
            request_id=request_id,
            category=category,
            filename=filename,
            file_size_bytes=0,
            defective=None,
            status="error",
            error_type="missing_image_file",
        )
        raise HTTPException(status_code=400, detail="Empty image upload")

    logger.info(
        "prediction_started category=%s filename=%s bytes=%s",
        category,
        image.filename,
        len(image_bytes),
    )

    try:
        pred = predict(category, image_bytes)
    except ValueError as exc:
        logger.warning("bad_image category=%s filename=%s error=%s", category, image.filename, str(exc))
        write_inference_event(
            request_id=request_id,
            category=category,
            filename=filename,
            file_size_bytes=len(image_bytes),
            defective=None,
            status="error",
            error_type="incorrect_image_file",
        )
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("prediction_failed category=%s filename=%s", category, image.filename)
        write_inference_event(
            request_id=request_id,
            category=category,
            filename=filename,
            file_size_bytes=len(image_bytes),
            defective=None,
            status="error",
            error_type="prediction_error",
        )
        raise HTTPException(status_code=500, detail="Prediction failed")

    PREDICTIONS_TOTAL.labels(
        category=category,
        defective=str(bool(pred)).lower(),
        model_name=MLFLOW_MODEL_NAME,
        model_version=MLFLOW_MODEL_VERSION,
        run_id=MLFLOW_RUN_ID,
    ).inc()

    try:
        write_inference_event(
            request_id=request_id,
            category=category,
            filename=filename,
            file_size_bytes=len(image_bytes),
            defective=int(bool(pred)),
            status="ok",
        )
    except Exception:
        logger.warning("event_logging_failed category=%s filename=%s", category, image.filename, exc_info=True)

    logger.info("prediction_completed category=%s defective=%s", category, bool(pred))
    return PredictResponse(defective=bool(pred))
