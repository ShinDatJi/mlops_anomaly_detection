from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.prediction import predict as predict_module

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
predict_module.models_dir = str(MODELS_DIR) + "/"


class StatusResponse(BaseModel):
    status: str


class PredictResponse(BaseModel):
    defective: bool


app = FastAPI(title="MLOps Prediction API")


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(status="ok")


@app.post("/predict/{category}", response_model=PredictResponse)
async def predict_category(category: str, image: UploadFile = File(...)) -> PredictResponse:
    category_dir = MODELS_DIR / category
    if not category_dir.exists() or not category_dir.is_dir():
        raise HTTPException(status_code=404, detail="Unknown category")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    pred = predict_module.predict(category, image_bytes)
    return PredictResponse(defective=bool(pred))
