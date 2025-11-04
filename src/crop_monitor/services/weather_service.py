# src/crop_monitor/services/weather_service.py
import requests
import logging
from cachetools import cached, TTLCache
from datetime import datetime
import os
import csv

#from src.crop_monitor.config import settings

from src.crop_monitor.config.settings import settings 



logger = logging.getLogger(__name__)

class WeatherService:
    def __init__(self):
        self.api_key = settings.OPENWEATHER_API_KEY  # now works
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.cache = TTLCache(maxsize=100, ttl=3600)

    @cached(cache=TTLCache(maxsize=100, ttl=300))  # 5 minute cache
    def get_weather_data(self, lat: float, lon: float):
        """Get weather data with caching and fallback"""
        cache_key = f"weather_{lat}_{lon}"
        
        try:
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            url = f"{self.base_url}/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                weather_data = {
                    "temperature": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "conditions": data["weather"][0]["description"],
                    "rain_last_hour": data.get("rain", {}).get("1h", 0),
                    "wind_speed": data["wind"]["speed"],
                    "timestamp": datetime.now().isoformat(),
                    "source": "openweathermap"
                }
                
                self.cache[cache_key] = weather_data
                return weather_data
            else:
                logger.warning(f"Weather API returned {response.status_code}")
                return self.get_fallback_weather_data()
                
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return self.get_fallback_weather_data()

    def get_fallback_weather_data(self):
        """Fallback weather data when API fails"""
        return {
            "temperature": 25.0,
            "humidity": 70.0,
            "conditions": "unknown",
            "rain_last_hour": 0,
            "wind_speed": 2.0,
            "timestamp": datetime.now().isoformat(),
            "source": "fallback",
            "warning": "Weather data may be inaccurate"
        }
    