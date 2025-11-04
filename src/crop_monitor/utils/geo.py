# Geospatial utilities
import numpy as np
import pandas as pd
import geopandas as gpd
from geopy.distance import geodesic
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Nigeria bounding box
NIGERIA_BOUNDS = {
    'lat_min': 4.0, 'lat_max': 14.0,
    'lon_min': 2.7, 'lon_max': 14.7
}

def is_in_nigeria(lat: float, lon: float) -> bool:
    """Check if coordinates are within Nigeria"""
    return (NIGERIA_BOUNDS['lat_min'] <= lat <= NIGERIA_BOUNDS['lat_max'] and 
            NIGERIA_BOUNDS['lon_min'] <= lon <= NIGERIA_BOUNDS['lon_max'])

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers"""
    return geodesic((lat1, lon1), (lat2, lon2)).km

def get_utm_zone(lon: float) -> str:
    """Get UTM zone for a longitude"""
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:326{zone:02d}" if zone <= 60 else f"EPSG:327{zone-60:02d}"

def validate_coordinates(df: pd.DataFrame, lat_col: str = 'Latitude', lon_col: str = 'Longitude') -> pd.DataFrame:
    """Validate coordinates are within reasonable bounds"""
    valid_mask = (
        df[lat_col].between(-90, 90) & 
        df[lon_col].between(-180, 180) &
        df[lat_col].notna() & 
        df[lon_col].notna()
    )
    return df[valid_mask]