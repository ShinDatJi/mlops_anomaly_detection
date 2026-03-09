from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from predict import predict

class StatusResponse(BaseModel):
    status: str

class ParamsModel(BaseModel):
    patches: int
    overlap: float
    height_cropping: int
    width_cropping: int
    threshold: float

class PredictResponse(BaseModel):
    defective: bool
    params: ParamsModel
    pred_probas: list[float]

app = FastAPI(
    title="Prediction API",
    description="FastAPI service for anomaly prediction by category.",
    version="0.1.0",
)

@app.get("/status", response_model=StatusResponse, tags=["system"], summary="Health check")
def status() -> StatusResponse:
    return StatusResponse(status="ok")

@app.post(
    "/predict/{category}/{version}",
    response_model=PredictResponse,
    tags=["prediction"],
    summary="Predict if an image is defective",
)
async def predict_category(category: str, version: str, request: Request, image: UploadFile = File(...)) -> PredictResponse:
    category = (category or "").strip()
    version = (version or "").strip()

    if not category:
        raise HTTPException(status_code=400, detail="Missing category")
    if not version:
        raise HTTPException(status_code=400, detail="Missing version")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    try:
        pred, pred_probas, params = predict(category, version, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")

    return PredictResponse(defective=bool(pred), pred_probas=pred_probas, params=ParamsModel(**params))