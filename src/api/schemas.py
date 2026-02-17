from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature values as key:value pairs")

class PredictResponse(BaseModel):
    prediction: int
    probability_class_1: float
    model_version: Optional[str] = None
    notes: Optional[str] = None
