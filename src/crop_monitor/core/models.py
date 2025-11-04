# src/crop_monitor/core/models.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class SoilData(BaseModel):
    lat: float
    lon: float
    ph_level: float
    organic_matter: float
    nitrogen: float
    phosphorus: float
    potassium: float
    soil_type: str
    health_score: float
    timestamp: datetime

class WeatherData(BaseModel):
    lat: float
    lon: float
    temperature: float
    humidity: float
    conditions: str
    rain_last_hour: float
    wind_speed: float
    timestamp: datetime

class SatelliteData(BaseModel):
    lat: float
    lon: float
    ndvi: float
    health_status: str
    recommendations: str
    timestamp: datetime

class PredictionResult(BaseModel):
    image_filename: str
    disease: str
    confidence: float
    treatment: str
    soil_data: Dict[str, Any]
    satellite_data: Dict[str, Any]
    weather_data: Dict[str, Any]
    timestamp: datetime