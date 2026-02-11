import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

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


class StatusResponse(BaseModel):
    status: str


class PredictResponse(BaseModel):
    defective: bool


app = FastAPI(
    title="Prediction API",
    description="FastAPI service for anomaly prediction by category.",
    version="0.1.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    client = request.client.host if request.client else "unknown"
    logger.info("request_started method=%s path=%s client=%s", request.method, request.url.path, client)

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request_failed method=%s path=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

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
async def predict_category(category: str, image: UploadFile = File(...)) -> PredictResponse:
    category_dir = MODELS_DIR / category
    if not category_dir.exists() or not category_dir.is_dir():
        logger.warning("unknown_category category=%s", category)
        raise HTTPException(status_code=404, detail="Unknown category")

    image_bytes = await image.read()
    if not image_bytes:
        logger.warning("empty_upload category=%s filename=%s", category, image.filename)
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
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("prediction_failed category=%s filename=%s", category, image.filename)
        raise HTTPException(status_code=500, detail="Prediction failed")

    logger.info("prediction_completed category=%s defective=%s", category, bool(pred))
    return PredictResponse(defective=bool(pred))
