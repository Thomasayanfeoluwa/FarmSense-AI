# src/crop_monitor/core/schemas.py
from pydantic import BaseModel
from typing import Dict, Any, Optional

class Location(BaseModel):
    lat: float
    lon: float

class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    treatment: str
    soil: Dict[str, Any]
    satellite: Dict[str, Any]
    weather: Dict[str, Any]