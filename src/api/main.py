import json
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException

from .schemas import PredictRequest, PredictResponse
from .security import require_api_key

APP_ROOT = Path(__file__).resolve().parents[2]  # project_root/src/api/main.py -> project_root
MODEL_PATH = APP_ROOT / "artifacts" / "models" / "heart_disease_pipeline.joblib"
META_PATH = APP_ROOT / "artifacts" / "models" / "metadata.json"

app = FastAPI(
    title="Heart Disease Classification API",
    version=os.getenv("APP_VERSION", "0.1.0"),
)

MODEL = None
METADATA: Dict[str, Any] = {}

@app.on_event("startup")
def load_artifacts() -> None:
    global MODEL, METADATA

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")

    MODEL = joblib.load(MODEL_PATH)

    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            METADATA = json.load(f)
    else:
        METADATA = {}

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_path": str(MODEL_PATH),
        "has_metadata": bool(METADATA),
        "version": app.version,
    }

@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest) -> PredictResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        
        x = pd.DataFrame([req.features])

        
        proba = MODEL.predict_proba(x)
        p1 = float(proba[0][1])
        pred = int(p1 >= 0.5)

        model_version = None
        if isinstance(METADATA, dict):
            model_version = METADATA.get("model_version") or METADATA.get("run_id")

        return PredictResponse(
            prediction=pred,
            probability_class_1=p1,
            model_version=model_version,
            notes="Threshold is 0.5 by default",
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {str(e)}")
