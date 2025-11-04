# src/crop_monitor/services/satellite_service.py
import csv
import requests
import logging
from cachetools import TTLCache
from datetime import datetime
import random

from src.crop_monitor.config.settings import settings

logger = logging.getLogger(__name__)

class SatelliteService:
    def __init__(self):
        self.instance_id = settings.SENTINELHUB_INSTANCE_ID  # configured
        self.cache = TTLCache(maxsize=50, ttl=86400)  # 24h cache

    def get_vegetation_data(self, lat: float, lon: float):
        """Get vegetation health data using instance cache."""
        cache_key = f"satellite_{lat}_{lon}"
        
        try:
            # Return from cache if available
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Fetch NDVI
            if self.instance_id:
                ndvi_value = self.get_ndvi_from_sentinelhub(lat, lon)
            else:
                ndvi_value = self.calculate_ndvi(lat, lon)  # mock fallback

            vegetation_data = {
                "ndvi": ndvi_value,
                "health_status": self.get_health_status(ndvi_value),
                "recommendations": self.get_vegetation_recommendations(ndvi_value),
                "timestamp": datetime.now().isoformat(),
                "source": "sentinelhub" if self.instance_id else "mock"
            }

            self.cache[cache_key] = vegetation_data
            return vegetation_data

        except Exception as e:
            logger.error(f"Satellite service error: {e}")
            return self.get_fallback_vegetation_data()

    def get_ndvi_from_sentinelhub(self, lat: float, lon: float) -> float:
        """Fetch NDVI from SentinelHub API (placeholder)."""
        logger.info(f"Fetching NDVI from SentinelHub for lat={lat}, lon={lon}")
        return round(random.uniform(0.1, 0.9), 2)

    def calculate_ndvi(self, lat: float, lon: float) -> float:
        """Mock NDVI calculation."""
        return round(random.uniform(0.1, 0.9), 2)

    def get_health_status(self, ndvi: float) -> str:
        if ndvi > 0.6:
            return "excellent"
        elif ndvi > 0.4:
            return "good"
        elif ndvi > 0.2:
            return "moderate"
        else:
            return "poor"

    def get_vegetation_recommendations(self, ndvi: float) -> str:
        if ndvi < 0.3:
            return "Consider soil nutrition assessment and potential irrigation needs."
        elif ndvi > 0.7:
            return "Healthy vegetation detected. Maintain current practices."
        else:
            return "Moderate vegetation health. Monitor for changes."

    def get_fallback_vegetation_data(self):
        return {
            "ndvi": 0.5,
            "health_status": "unknown",
            "recommendations": "Satellite data temporarily unavailable.",
            "timestamp": datetime.now().isoformat(),
            "source": "fallback",
            "warning": "Vegetation data may be estimated"
        }