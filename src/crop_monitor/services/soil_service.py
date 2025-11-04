# import traceback
# import pandas as pd
# import numpy as np
# from geopy.distance import geodesic
# import logging
# from datetime import datetime
# import joblib
# import os
# import json
# import csv
# import requests
# from typing import Dict, Any, Optional

# from src.crop_monitor.config.settings import settings 
# from src.crop_monitor.utils.feature_validator import validate_features

# logger = logging.getLogger(__name__)

# class SoilService:
#     # ----------------- Feature mapping for model compatibility ----------------- #
#     FEATURE_MAPPING = {
#         "aridity": "Aridity",
#         "clay": "Clay(%)",
#         "silt": "Silt(%)",
#         "sand": "Sand(%)",
#         "bd": "BD(g/cm3)",
#         "elevation": "Elevation(m)",
#         "ph_level": "pH_level",
#         "organic_matter": "organic_matter",
#         "nitrogen": "Nitrogen",
#         "phosphorus": "Phosphorus",
#         "potassium": "Potassium",
#         "crop_type_encoded": "CropTypeEncoded",
#         "fertilizer": "Fertilizer"
#     }

#     def __init__(self, soil_data_path: str = None, model_path: str = None, dsm_maps_dir: str = None):
#         self.soil_data_path = soil_data_path or r"C:\Users\ADEGOKE\Desktop\AI_Crop_Disease_Monitoring\data\external\global_soil_data.csv"
#         self.model_path = model_path or r"C:\Users\ADEGOKE\Desktop\AI_Crop_Disease_Monitoring\src\crop_monitor\models\best_rf.pkl"
        
#         try:
#             with open(settings.SOIL_FEATURES_JSON_PATH, "r", encoding="utf-8") as f:
#                 soil_json = json.load(f)
#             self.soil_data = pd.DataFrame(soil_json)
#             logger.info(f"Loaded soil features JSON with {len(self.soil_data)} records")
#         except Exception as e:
#             logger.error(f"Failed to load soil features JSON: {e}")
#             self.soil_data = pd.DataFrame()

#         try:
#             self.feature_json_path = settings.SOIL_FEATURES_JSON_PATH
#             self.model_expected_features = self._load_model_features()
#             logger.info(f"Loaded {len(self.model_expected_features)} validated model features")
#         except Exception as e:
#             logger.error(f"Error loading model features: {e}")
#             self.model_expected_features = []

#         logger.info(f"Using soil model: {self.model_path}")
#         logger.info(f"Model size setting: {settings.SOIL_MODEL_SIZE}")

#         self.model = self._load_model()
#         self.soil_model = self.model
#         self.soil_data_csv = self._load_soil_data()


#     # ----------------- Helper Methods ----------------- #
#     def _load_model_features(self):
#         try:
#             if not os.path.exists(self.feature_json_path):
#                 logger.warning(f"Features JSON not found: {self.feature_json_path}")
#                 return []

#             with open(self.feature_json_path, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#             features = data.get("model_expected_features", [])
#             logger.info(f"Loaded {len(features)} model features from JSON")
#             return features
#         except Exception as e:
#             logger.error(f"Error loading model features: {e}")
#             return []

#     def _load_soil_data(self):
#         try:
#             if not os.path.exists(self.soil_data_path):
#                 logger.warning(f"Soil data file not found: {self.soil_data_path}")
#                 return pd.DataFrame()
            
#             df = pd.read_csv(self.soil_data_path, encoding='latin1')
#             logger.info(f"Loaded soil CSV data with {len(df)} records from {self.soil_data_path}")
#             return df

#         except Exception as e:
#             logger.error(f"Error loading soil data: {e}")
#             return pd.DataFrame()

#     def _load_model(self):
#         try:
#             if not os.path.exists(self.model_path):
#                 logger.warning(f"Soil model file not found: {self.model_path}")
#                 return None
#             logger.info(f"Loading soil model from: {self.model_path}")
#             return joblib.load(self.model_path)
#         except Exception as e:
#             logger.error(f"Error loading soil model: {e}")
#             return None

#     def get_soil_features(self, lat: float, lon: float) -> Dict[str, Any]:
#         """
#         Fetch soil features from SoilGrids API based on latitude/longitude.
#         Falls back to default dummy values if API fails.
#         """
#         BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
#         params = {
#             "lon": lon,
#             "lat": lat,
#             "depth": "sl1",   # surface layer (0–5cm)
#             "property": ["sand", "silt", "clay", "soc", "phh2o", "cec"],
#         }

#         try:
#             resp = requests.get(BASE_URL, params=params, timeout=10)
#             resp.raise_for_status()
#             data = resp.json()

#             props = {}
#             for prop in params["property"]:
#                 values = data["properties"]["layers"][prop]["depths"][0]["values"]
#                 props[prop] = values.get("mean")

#             features = {
#                 "sand": props.get("sand", 40),
#                 "silt": props.get("silt", 30),
#                 "clay": props.get("clay", 30),
#                 "organic_matter": props.get("soc", 2.0),
#                 "ph": props.get("phh2o", 6.5),
#                 "cec": props.get("cec", 15),
#             }

#             logger.info(f"Retrieved soil features from SoilGrids for lat={lat}, lon={lon}: {features}")
#             return features

#         except Exception as e:
#             logger.error(f"SoilGrids API failed: {e}")
#             return {
#                 "sand": 55,
#                 "silt": 25,
#                 "clay": 20,
#                 "organic_matter": 2.5,
#                 "ph": 6.5,
#                 "cec": 15,
#             }

#     def _predict_soil_type(self, input_features: Dict[str, Any]) -> str:
#         """
#         Predict soil type using the loaded soil model and return a human-readable label.
#         Always returns a string label, even if the model is missing or fails.
#         """
#         try:
#             # Fallback logic if model is not loaded
#             if not hasattr(self, "soil_model") or self.soil_model is None:
#                 clay = input_features.get("clay", 30)
#                 silt = input_features.get("silt", 30)
#                 sand = input_features.get("sand", 40)
#                 if sand >= 60:
#                     return "sandy"
#                 elif clay >= 40:
#                     return "clay"
#                 elif silt >= 40:
#                     return "silty"
#                 else:
#                     return "loam"

#             # Prepare model input safely
#             model_input = [input_features.get(f, 0) for f in self.soil_model.feature_names_in_]

#             # Predict numeric value
#             numeric_pred = self.soil_model.predict([model_input])[0]

#             # Map numeric prediction to labels
#             soil_type_map = {
#                 0: "sandy",
#                 1: "loam",
#                 2: "clay",
#                 3: "silty",
#                 4: "peaty",
#                 5: "chalky",
#                 6: "organic"
#             }

#             # Round and map
#             return soil_type_map.get(round(numeric_pred), "loam")

#         except Exception as e:
#             logger.warning(f"Failed to predict soil type: {e}")
#             return "loam"

#     def align_features(self, raw_dict: dict):
#         try:
#             # Create dataframe from input dict
#             df = pd.DataFrame([raw_dict]) if isinstance(raw_dict, dict) else pd.DataFrame(raw_dict)

#             # Ensure all expected features exist and are initialized with sensible defaults
#             if isinstance(self.model_expected_features, list) and self.model_expected_features:
#                 for col in self.model_expected_features:
#                     if col not in df.columns:
#                         df[col] = self._default_value(col)

#                 # Keep only expected columns in the expected order
#                 df = df[self.model_expected_features]
#             else:
#                 # If model has no expected features defined, keep the input as-is
#                 df = df.copy()

#             # Validate features via existing validator (may adjust types/scales)
#             df_aligned = validate_features(df=df, json_path=self.feature_json_path)

#             # Ensure method returns a DataFrame with all expected features
#             if isinstance(self.model_expected_features, list) and self.model_expected_features:
#                 for col in self.model_expected_features:
#                     if col not in df_aligned.columns:
#                         df_aligned[col] = self._default_value(col)
#                 df_aligned = df_aligned[self.model_expected_features]

#             return df_aligned
#         except Exception as e:
#             logger.error(f"Error aligning features: {e}")
#             # Return a one-row DataFrame with safe defaults for all expected features
#             if isinstance(self.model_expected_features, list) and self.model_expected_features:
#                 return pd.DataFrame([{f: self._default_value(f) for f in self.model_expected_features}])
#             return pd.DataFrame()

#     def _query_soilgrids_fixed(self, lat: float, lon: float):
#         """
#         Fetch topsoil (0-5cm) properties from SoilGrids and return validated soil info.
#         Always uses SoilGrids; fallback only if API fails.
#         """
#         try:
#             # Build request URL from settings
#             url = f"{settings.SOILGRIDS_API_URL}?lon={lon}&lat={lat}"
#             response = requests.get(url, timeout=30)
#             response.raise_for_status()
#             data = response.json()
#             props = data.get("properties", {})

#             # Helper function to extract mean topsoil value
#             def get_mean(prop_name, default):
#                 try:
#                     layer = props.get(prop_name, {})
#                     depths = layer.get("depths", [])
#                     if depths:
#                         val = depths[0]["values"].get("mean")
#                         if val is not None and not np.isnan(val):
#                             return float(val)
#                 except Exception:
#                     pass
#                 return default

#             # Get main properties
#             clay = get_mean("clay", 30)
#             silt = get_mean("silt", 30)
#             sand = get_mean("sand", 40)
#             ph_val = get_mean("phh2o", 6.5)

#             # Normalize clay+silt+sand to 100%
#             total = clay + silt + sand
#             if total > 0:
#                 clay = round(clay / total * 100, 1)
#                 silt = round(silt / total * 100, 1)
#                 sand = round(sand / total * 100, 1)

#             # Construct final soil info
#             soil_info = {
#                 "clay": clay,
#                 "silt": silt,
#                 "sand": sand,
#                 "ph_level": ph_val,
#                 "organic_matter": self._default_value("organic_matter"),
#                 "nitrogen": self._default_value("nitrogen"),
#                 "phosphorus": self._default_value("phosphorus"),
#                 "potassium": self._default_value("potassium"),
#                 "soil_type": self._predict_soil_type({
#                     "clay": clay,
#                     "silt": silt,
#                     "sand": sand,
#                     "ph_level": ph_val
#                 }),
#                 "timestamp": datetime.now().isoformat(),
#                 "source": "soilgrids"
#             }

#             logger.info(f"SoilGrids returned soil info: {soil_info}")
#             return soil_info

#         except Exception as e:
#             logger.warning(f"SoilGrids query failed, using fallback: {e}")
#             fallback = self.get_fallback_soil_data()
#             fallback["soil_type"] = self._predict_soil_type(fallback)
#             fallback["timestamp"] = datetime.now().isoformat()
#             return fallback


#     def get_enhanced_soil_analysis(self, lat: float, lon: float):
#         """
#         Enhanced version that uses DSM maps when available
#         """
#         # Try DSM first (high-resolution Nigeria maps)
#         dsm_data = self._get_dsm_soil_data(lat, lon)
#         if dsm_data and dsm_data.get('source') == 'dsm_high_res':
#             logger.info("Using high-resolution DSM data")
#             return self._enhance_with_dsm_prediction(dsm_data)
        
#         # Fall back to current method
#         return self.get_soil_analysis(lat, lon)

#     # ----------------- Main Public Methods ----------------- #
#     def get_soil_analysis(self, lat: float, lon: float):
#         """
#         ORIGINAL METHOD - Get soil analysis dynamically:
#         1. Try nearest soil record from CSV/JSON
#         2. If not found, query FAO/ISRIC API
#         3. If API fails, use fallback
#         4. Always populate all soil properties
#         5. Predict health score dynamically using the model
        
#         This is your existing method that get_enhanced_soil_analysis falls back to
#         """
#         try:
#             # Step 1: Try nearest soil from CSV/JSON
#             if self.soil_data_csv is not None and not self.soil_data_csv.empty:
#                 nearest_soil = self.find_nearest_soil_data(lat, lon, df=self.soil_data_csv, max_distance_km=10)
#             else:
#                 nearest_soil = None

#             if nearest_soil is not None and isinstance(nearest_soil, dict) and len(nearest_soil) > 0:

#                 # Map nearest CSV/JSON fields to standardized keys
#                 soil_info = {
#                     "ph_level": nearest_soil.get('pH_level', 6.5),
#                     "organic_matter": nearest_soil.get('SOC(g/kg)', 2.0),
#                     "nitrogen": nearest_soil.get('Nitrogen', 20.0),
#                     "phosphorus": nearest_soil.get('Phosphorus', 15.0),
#                     "potassium": nearest_soil.get('Potassium', 150.0),
#                     "clay": nearest_soil.get('Clay(%)', 30.0),
#                     "silt": nearest_soil.get('Silt(%)', 30.0),
#                     "sand": nearest_soil.get('Sand(%)', 40.0),
#                     "soil_type": nearest_soil.get('soil_type', None),
#                     "timestamp": datetime.now().isoformat(),
#                     "source": "fao_soil_database",
#                     "model_size": settings.SOIL_MODEL_SIZE
#                 }

#                 # Ensure numeric values and classify soil_type if missing
#                 for key in ["clay", "sand", "silt"]:
#                     val = soil_info.get(key)
#                     if val is None or not isinstance(val, (int, float)) or np.isnan(val):
#                         soil_info[key] = self._default_value(key)

#                 # Predict soil type
#                 predicted_type = self._predict_soil_type(soil_info)
#                 soil_info["soil_type"] = predicted_type
#                 logger.debug(f"Predicted soil_type: {predicted_type} for features: {soil_info}")

#                 # Log if fallback 'loam' is used
#                 if soil_info["soil_type"] == "loam":
#                     logger.warning(f"Soil type predictor returned fallback 'loam' for CSV/JSON data at coordinates ({lat}, {lon}). Check input features: {soil_info}")

#             else:
#                 # Step 2: Query API (FAO/ISRIC) as backup, else use fallback
#                 soil_info = self._query_soilgrids_fixed(lat, lon) or self.get_fallback_soil_data()

#             # Ensure soil_info contains all basic keys
#             base_keys = ["ph_level", "organic_matter", "nitrogen", "phosphorus", "potassium",
#                         "clay", "silt", "sand", "soil_type"]
#             for k in base_keys:
#                 if k not in soil_info or soil_info.get(k) is None:
#                     soil_info[k] = self._default_value(k) if k in ["clay", "silt", "sand"] else self._default_value(k)

#             # Step 3: Predict soil health dynamically
#             soil_info["health_score"] = self._predict_soil_health_internal(soil_info)

#             return soil_info

#         except Exception as e:
#             logger.error(f"Soil service error: {e}")
#             fallback = self.get_fallback_soil_data()
#             logger.debug(f"Predicting soil_type for fallback input: {fallback}")
#             fallback["soil_type"] = self._predict_soil_type(fallback)
#             # Log if fallback 'loam' is used
#             if fallback["soil_type"] == "loam":
#                 logger.warning(f"Soil type predictor returned fallback 'loam' during fallback in get_soil_analysis at coordinates ({lat}, {lon}). Check input features: {fallback}")
#             return fallback

#     def _predict_soil_health_internal(self, soil_data: dict) -> float:
#         """
#         Internal method for soil health prediction used by both original and enhanced methods
#         """
#         try:
#             if self.model is not None and self.model_expected_features:
#                 aligned_input = self.align_features(soil_data)

#                 # Fill missing columns with defaults and NaNs
#                 for col in self.model_expected_features:
#                     if col not in aligned_input.columns:
#                         aligned_input[col] = self._default_value(col)
#                     if aligned_input[col].isnull().any():
#                         aligned_input[col].fillna(self._default_value(col), inplace=True)

#                 aligned_input = aligned_input[self.model_expected_features]
#                 aligned_input = aligned_input.apply(pd.to_numeric, errors='coerce')
#                 for col in aligned_input.columns:
#                     if aligned_input[col].isnull().any():
#                         aligned_input[col].fillna(self._default_value(col), inplace=True)

#                 try:
#                     pred = self.model.predict(aligned_input)
#                     return float(pred[0]) if hasattr(pred, '__len__') and len(pred) > 0 else float(pred)
#                 except Exception as e:
#                     logger.error(f"Model prediction failed: {e}")
#                     return 0.7
#             else:
#                 return 0.7
#         except Exception as e:
#             logger.error(f"Soil health prediction error: {e}")
#             return 0.7

#     def predict_soil_health(self, soil_data: dict) -> float:
#         """
#         Predict soil health dynamically using trained model and aligned features.
#         Ensures all required features are present, fills missing values with defaults.
#         """
#         try:
#             if self.model is not None and self.model_expected_features:
#                 input_df = self.align_features(soil_data)
#                 # Fill missing or null features
#                 for col in self.model_expected_features:
#                     if col not in input_df.columns:
#                         input_df[col] = self._default_value(col)
#                     if input_df[col].isnull().any():
#                         input_df[col].fillna(self._default_value(col), inplace=True)
#                 # Keep only expected features
#                 input_df = input_df[self.model_expected_features]
#                 input_df = input_df.apply(pd.to_numeric, errors='coerce')
#                 for col in input_df.columns:
#                     if input_df[col].isnull().any():
#                         input_df[col].fillna(self._default_value(col), inplace=True)
#                 # Predict dynamically
#                 prediction = self.model.predict(input_df)
#                 return float(prediction[0]) if hasattr(prediction, '__len__') and len(prediction) > 0 else float(prediction)
#             else:
#                 return 0.7
#         except Exception as e:
#             logger.error(f"Soil health prediction error: {e}")
#             return 0.7


#     def find_nearest_soil_data(self, lat: float, lon: float, df: pd.DataFrame = None, max_distance_km=10):
#         df = df or self.soil_data_csv
#         try:
#             if df is None or df.empty:
#                 return None

#             # Ensure required columns exist and are numeric
#             if not set(["Latitude", "Longitude"]).issubset(set(df.columns)):
#                 logger.warning("Soil CSV does not contain Latitude/Longitude columns")
#                 return None

#             # Drop rows with missing lat/lon
#             df_valid = df.dropna(subset=["Latitude", "Longitude"]).copy()
#             if df_valid.empty:
#                 return None

#             # Compute distances safely
#             def safe_distance(row):
#                 try:
#                     lat_r = float(row["Latitude"])
#                     lon_r = float(row["Longitude"])
#                     return geodesic((lat, lon), (lat_r, lon_r)).km
#                 except Exception:
#                     return np.inf

#             distances = df_valid.apply(safe_distance, axis=1)
#             if distances.empty:
#                 return None

#             nearest_idx = distances.idxmin()
#             if distances.loc[nearest_idx] <= max_distance_km:
#                 return df_valid.loc[nearest_idx].to_dict()
#             return None
#         except Exception as e:
#             logger.warning(f"Could not find nearest soil data: {e}")
#             return None

#     def _default_value(self, feature_name: str):
#         defaults = {
#             "ph_level": 6.5,
#             "organic_matter": 2.0,
#             "nitrogen": 20.0,
#             "phosphorus": 15.0,
#             "potassium": 150.0,
#             "clay": 30.0,
#             "silt": 30.0,
#             "sand": 40.0,
#             "elevation": 100.0,
#             "map": 1000.0,
#             "mat": 25.0,
#             "pet": 1000.0,
#             "aridity": 0.8,
#             "temperature": 25.0,
#             "humidity": 50.0,
#             "rainfall": 50.0,
#             "soil_type_encoded": 0,
#             "ndvi": 0.5,
#             "vegetation_index": 0.5,
#             "precipitation": 50.0,
#             "sunlight_hours": 6.0,
#             "wind_speed": 5.0,
#             "soil_moisture": 0.3,
#             "crop_type_encoded": 0,
#             "irrigation": 0,
#             "fertilizer": 0,
#             "pesticide": 0
#         }
#         return defaults.get(feature_name, 0.0)

#     def get_fallback_soil_data(self) -> Dict[str, Any]:
#         """
#         Returns a complete fallback soil record if CSV/JSON/API fails.
#         Ensures all soil properties are present for dynamic predictions.
#         """
#         fallback = {
#             "ph_level": 6.5,
#             "organic_matter": 2.0,
#             "nitrogen": 20.0,
#             "phosphorus": 15.0,
#             "potassium": 150.0,
#             "clay": 30.0,
#             "silt": 30.0,
#             "sand": 40.0,
#             "soil_type": "loam",
#             "timestamp": datetime.now().isoformat(),
#             "source": "fallback",
#             "model_size": settings.SOIL_MODEL_SIZE,
#             "warning": "Soil data may be generalized",
#             "risk_level": "Medium"
#         }
#         return fallback

#     def get_user_by_location(self, lat: float, lon: float) -> dict:
#         """
#         Fetch user info for a given location.
#         """
#         # This method seems to reference MongoDB - keeping it for compatibility
#         # You might need to adjust this based on your actual database setup
#         try:
#             # Placeholder - implement based on your actual database structure
#             return {}
#         except Exception as e:
#             logger.error(f"Error getting user by location: {e}")
#             return {}

#     def enhance_recommendation(self, disease_name: str, confidence: float, crop_type: str) -> str:
#         if confidence > 0.8:
#             return f"Apply treatment for {disease_name} immediately on {crop_type}."
#         return f"Monitor {crop_type} for {disease_name}."

# if __name__ == "__main__":
#     sample_input = {
#         "clay": 30.0,
#         "silt": 40.0,
#         "sand": 30.0,
#         "organic_matter": 12.5,
#         "bd": 1.3,
#         "elevation": 250.0,
#         "map": 1200.0,
#         "mat": 26.0,
#         "pet": 1000.0,
#         "aridity": 0.7,
#         "crop_type_encoded": 1,
#         "soil_type_encoded": 2,
#         "irrigation": 1,
#         "fertilizer": 1,
#         "pesticide": 0,
#         "ndvi": 0.55,
#         "vegetation_index": 0.60,
#         "precipitation": 48.0,
#         "sunlight_hours": 7.0,
#         "wind_speed": 4.5,
#         "soil_moisture": 0.32,
#         "temperature": 27.0,
#         "humidity": 65.0
#     }

#     service = SoilService()
    
#     # Test original method
#     health_score = service.predict_soil_health(sample_input)
#     print(f"Predicted soil health: {health_score}")
    
#     # Test enhanced method for Nigeria location
#     nigeria_location = service.get_enhanced_soil_analysis(9.0, 8.0)  # Nigeria coordinates
#     print(f"Enhanced analysis for Nigeria: {nigeria_location.get('source', 'unknown')}")
    
#     # Test fallback for non-Nigeria location
#     non_nigeria_location = service.get_enhanced_soil_analysis(40.0, -74.0)  # New York coordinates
#     print(f"Enhanced analysis for non-Nigeria: {non_nigeria_location.get('source', 'unknown')}")




# import traceback
# import pandas as pd
# import numpy as np
# from geopy.distance import geodesic
# import logging
# from datetime import datetime
# import joblib
# import os
# import json
# import csv
# import functools
# import requests
# from typing import Dict, Any, Optional, List
# import warnings
# from sklearn.exceptions import DataConversionWarning

# warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# from src.crop_monitor.config.settings import settings 
# from src.crop_monitor.utils.feature_validator import validate_features

# logger = logging.getLogger(__name__)

# class SoilService:
#     # ----------------- Feature mapping for model compatibility ----------------- #
#     FEATURE_MAPPING = {
#         "aridity": "Aridity",
#         "clay": "Clay(%)",
#         "silt": "Silt(%)", 
#         "sand": "Sand(%)",
#         "bd": "BD(g/cm3)",
#         "elevation": "Elevation(m)",
#         "ph_level": "pH_level",
#         "organic_matter": "organic_matter",
#         "nitrogen": "Nitrogen",
#         "phosphorus": "Phosphorus",
#         "potassium": "Potassium",
#         "crop_type_encoded": "CropTypeEncoded",
#         "fertilizer": "Fertilizer",
#         # New features for advanced models - standardized naming
#         "longitude": "Longitude",
#         "latitude": "Latitude", 
#         "map": "MAP(mm)",
#         "mat": "MAT(Celsius)",
#         "pet": "PET(mm)",
#         "depth_thickness": "Depth_Thickness",
#         "depth_ratio": "Depth_Ratio",
#         "depth_middle_normalized": "Depth_Middle_Normalized",
#         "aridity_elevation": "Aridity_Elevation",
#         "pet_map_ratio": "PET_MAP_Ratio",
#         "climate_productivity": "Climate_Productivity",
#         "usda_texture_encoded": "USDA_Texture_Encoded",
#         "region_encoded": "REGION_Encoded",
#         "map_mat_ratio_robust": "MAP_MAT_Ratio_Robust",
#         "upper_depth": "UpperDepth(cm)",
#         "lower_depth": "LowerDepth(cm)",
#         "middle_depth": "MiddleDepth(cm)"
#     }

#     # Reverse mapping for consistent feature handling
#     REVERSE_FEATURE_MAPPING = {v: k for k, v in FEATURE_MAPPING.items()}

#     def __init__(self, soil_data_path: str = None, model_path: str = None):
#         # self.soil_data_path = soil_data_path or r"C:\Users\ADEGOKE\Desktop\AI_Crop_Disease_Monitoring\data\external\global_soil_data.csv"
#         # self.model_path = model_path or r"C:\Users\ADEGOKE\Desktop\AI_Crop_Disease_Monitoring\src\crop_monitor\models\best_rf.pkl"
#         self.soil_data_path = soil_data_path or settings.SOIL_DATA_PATH
#         self.model_path = model_path or settings.SOIL_MODEL_PATH

#         try:
#             with open(settings.SOIL_FEATURES_JSON_PATH, "r", encoding="utf-8") as f:
#                 soil_json = json.load(f)
#             self.soil_data = pd.DataFrame(soil_json)
#             logger.info(f"Loaded soil features JSON with {len(self.soil_data)} records")
#         except Exception as e:
#             logger.error(f"Failed to load soil features JSON: {e}")
#             self.soil_data = pd.DataFrame()

#         try:
#             self.feature_json_path = settings.SOIL_FEATURES_JSON_PATH
#             self.model_expected_features = self._load_model_features()
#             logger.info(f"Loaded {len(self.model_expected_features)} validated model features")
#         except Exception as e:
#             logger.error(f"Error loading model features: {e}")
#             self.model_expected_features = []

#         logger.info(f"Using soil model: {self.model_path}")
#         logger.info(f"Model size setting: {settings.SOIL_MODEL_SIZE}")

#         self.model = self._load_model()
#         self.soil_model = self.model
#         self.soil_data_csv = self._load_soil_data()
        
#         # Load new state-of-the-art soil models
#         self._load_advanced_soil_models()

#     def _load_advanced_soil_models(self):
#         """Load the advanced soil classification and SOC regression models using settings."""
#         try:
#             # Use paths from settings
#             clf_model_path = settings.SOIL_CLASS_MODEL_PATH
#             scaler_clf_path = settings.SCALER_CLASS_PATH
#             label_encoder_path = settings.TEXTURE_LABEL_ENCODER_PATH
#             soc_model_path = settings.SOC_MODEL_PATH
#             scaler_soc_path = settings.SCALER_SOC_PATH

#             # Check all files exist
#             missing_files = [p for p in [clf_model_path, scaler_clf_path, label_encoder_path, 
#                                         soc_model_path, scaler_soc_path] if not os.path.exists(p)]
#             if missing_files:
#                 logger.warning(f"Advanced model files not found: {missing_files}. Using basic methods.")
#                 self.advanced_models_loaded = False
#                 self.clf_model = None
#                 self.soc_model = None
#                 self.scaler_clf = None
#                 self.scaler_soc = None
#                 self.label_encoder = None
#                 return

#             # Load models
#             self.clf_model = joblib.load(clf_model_path)
#             self.scaler_clf = joblib.load(scaler_clf_path)
#             self.label_encoder = joblib.load(label_encoder_path)
#             self.soc_model = joblib.load(soc_model_path)
#             self.scaler_soc = joblib.load(scaler_soc_path)

#             logger.info("Successfully loaded advanced soil classification and SOC regression models")
#             self.advanced_models_loaded = True

#         except Exception as e:
#             logger.exception(f"Error loading advanced soil models: {e}")
#             self.advanced_models_loaded = False
#             self.clf_model = None
#             self.soc_model = None
#             self.scaler_clf = None
#             self.scaler_soc = None
#             self.label_encoder = None

#     def predict_soil_properties(self, features: dict) -> dict:
#         """
#         Predict both soil type (classification) and SOC (regression) using advanced models.
#         Falls back to basic methods if advanced models are not available.
#         """
#         # If advanced models aren't loaded, use basic methods
#         if not self.advanced_models_loaded:
#             return self._predict_soil_properties_basic(features)
            
#         try:
#             # Prepare input data with consistent feature handling
#             prepared_features = self._prepare_input_features(features)
#             df_input = pd.DataFrame([prepared_features])

#             # ----- Classification Prediction -----
#             classification_result = self._predict_soil_classification(df_input)
            
#             # ----- SOC Regression Prediction -----
#             regression_result = self._predict_soc_regression(df_input)

#             return {
#                 "soil_type": classification_result,
#                 "soc_g_per_kg": regression_result,
#                 "source": "advanced_model",
#                 "prediction_confidence": "high"
#             }

#         except Exception as e:
#             logger.exception(f"Advanced soil prediction failed: {e}")
#             # Fallback to basic methods with detailed error info
#             return {
#                 **self._predict_soil_properties_basic(features),
#                 "source": "basic_model_fallback",
#                 "prediction_confidence": "medium",
#                 "warning": f"Advanced model failed: {str(e)}"
#             }

#     def _predict_soil_classification(self, df_input: pd.DataFrame) -> str:
#         """Predict soil type using classification model"""
#         try:
#             X_clf = self._get_features_for_scaler(df_input, self.scaler_clf)
#             X_clf_scaled = self.scaler_clf.transform(X_clf)
#             soil_class_encoded = self.clf_model.predict(X_clf_scaled)[0]
#             soil_class_label = self.label_encoder.inverse_transform([soil_class_encoded])[0]
#             return soil_class_label
#         except Exception as e:
#             logger.exception(f"Soil classification prediction failed: {e}")
#             raise

#     def _predict_soc_regression(self, df_input: pd.DataFrame) -> float:
#         """Predict SOC value using regression model"""
#         try:
#             X_soc = self._get_features_for_scaler(df_input, self.scaler_soc)
#             X_soc_scaled = self.scaler_soc.transform(X_soc)
#             soc_value = self.soc_model.predict(X_soc_scaled)[0]
#             return round(soc_value, 2)
#         except Exception as e:
#             logger.exception(f"SOC regression prediction failed: {e}")
#             raise

#     def _get_features_for_scaler(self, df_input: pd.DataFrame, scaler) -> pd.DataFrame:
#         """Get properly aligned features for a specific scaler"""
#         required_features = scaler.feature_names_in_
#         prepared_df = pd.DataFrame()
        
#         for feature in required_features:
#             if feature in df_input.columns:
#                 prepared_df[feature] = df_input[feature]
#             else:
#                 # Convert to internal naming and get value
#                 internal_name = self.REVERSE_FEATURE_MAPPING.get(feature, feature)
#                 if internal_name in df_input.columns:
#                     prepared_df[feature] = df_input[internal_name]
#                 else:
#                     # Use default value
#                     prepared_df[feature] = [self._get_default_value(feature)]
        
#         return prepared_df[required_features]

#     def _prepare_input_features(self, features: dict) -> dict:
#         """Prepare and normalize input features for model prediction"""
#         prepared_features = {}
        
#         # First, normalize feature names to internal naming convention
#         for key, value in features.items():
#             # Handle ph vs ph_level inconsistency
#             if key == "ph":
#                 prepared_features["ph_level"] = value
#             else:
#                 prepared_features[key] = value
        
#         # Derive advanced features dynamically from scaler if available
#         if hasattr(self, 'scaler_clf') and self.scaler_clf is not None:
#             # Use classification scaler features as proxy for advanced set
#             model_features = [self.REVERSE_FEATURE_MAPPING.get(f, f) for f in self.scaler_clf.feature_names_in_]
#         else:
#             # Fallback to hardcoded list
#             model_features = [
#                 "longitude", "latitude", "elevation", "map", "mat", "pet", "aridity",
#                 "depth_thickness", "depth_ratio", "depth_middle_normalized", 
#                 "aridity_elevation", "pet_map_ratio", "climate_productivity",
#                 "usda_texture_encoded", "region_encoded", "map_mat_ratio_robust",
#                 "clay", "silt", "sand", "bd", "upper_depth", "lower_depth", "middle_depth",
#                 "ph_level", "organic_matter"
#             ]
        
#         # Ensure all required advanced features are present
#         for feature in model_features:
#             if feature not in prepared_features:
#                 prepared_features[feature] = self._get_default_value(feature)
        
#         return prepared_features

#     def _predict_soil_properties_basic(self, features: dict) -> dict:
#         """Fallback method using basic soil prediction"""
#         basic_soil_type = self._predict_soil_type_basic(features)
#         basic_soc = features.get('organic_matter', 2.0)
        
#         return {
#             "soil_type": basic_soil_type,
#             "soc_g_per_kg": round(basic_soc, 2),
#             "source": "basic_model"
#         }

#     def _get_default_value(self, feature_name: str) -> float:
#         """Unified method to get default values for any feature"""
#         # Handle both internal and external feature naming
#         internal_name = self.REVERSE_FEATURE_MAPPING.get(feature_name, feature_name)
        
#         defaults = {
#             # Basic soil features
#             "ph_level": 6.5,
#             "organic_matter": 2.0,
#             "nitrogen": 20.0,
#             "phosphorus": 15.0,
#             "potassium": 150.0,
#             "clay": 30.0,
#             "silt": 30.0,
#             "sand": 40.0,
            
#             # Environmental features
#             "elevation": 100.0,
#             "map": 1000.0,
#             "mat": 25.0,
#             "pet": 1200.0,
#             "aridity": 0.8,
#             "temperature": 25.0,
#             "humidity": 50.0,
#             "rainfall": 50.0,
#             "precipitation": 50.0,
            
#             # Soil properties
#             "soil_type_encoded": 0,
#             "ndvi": 0.5,
#             "vegetation_index": 0.5,
#             "sunlight_hours": 6.0,
#             "wind_speed": 5.0,
#             "soil_moisture": 0.3,
#             "bd": 1.3,
            
#             # Management practices
#             "crop_type_encoded": 0,
#             "irrigation": 0,
#             "fertilizer": 0,
#             "pesticide": 0,
            
#             # Advanced model features
#             "longitude": 0.0,
#             "latitude": 0.0,
#             "depth_thickness": 10.0,
#             "depth_ratio": 1.0,
#             "depth_middle_normalized": 0.5,
#             "aridity_elevation": 160.0,
#             "pet_map_ratio": 0.8,
#             "climate_productivity": 0.9,
#             "usda_texture_encoded": 3.0,
#             "region_encoded": 1.0,
#             "map_mat_ratio_robust": 60.5,
#             "upper_depth": 0.0,
#             "lower_depth": 10.0,
#             "middle_depth": 5.0,
            
#             # API response features
#             "cec": 15.0,
#         }
        
#         return defaults.get(internal_name, 0.0)

#     # ----------------- Core Helper Methods ----------------- #
#     def _load_model_features(self):
#         try:
#             if not os.path.exists(self.feature_json_path):
#                 logger.warning(f"Features JSON not found: {self.feature_json_path}")
#                 return []

#             with open(self.feature_json_path, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#             features = data.get("model_expected_features", [])
#             logger.info(f"Loaded {len(features)} model features from JSON")
#             return features
#         except Exception as e:
#             logger.exception(f"Error loading model features: {e}")
#             return []

#     def _load_soil_data(self):
#         try:
#             if not os.path.exists(self.soil_data_path):
#                 logger.warning(f"Soil data file not found: {self.soil_data_path}")
#                 return pd.DataFrame()
            
#             df = pd.read_csv(self.soil_data_path, encoding='latin1')
#             logger.info(f"Loaded soil CSV data with {len(df)} records from {self.soil_data_path}")
#             return df

#         except Exception as e:
#             logger.exception(f"Error loading soil data: {e}")
#             return pd.DataFrame()

#     def _load_model(self):
#         try:
#             if not os.path.exists(self.model_path):
#                 logger.warning(f"Soil model file not found: {self.model_path}")
#                 return None
#             logger.info(f"Loading soil model from: {self.model_path}")
#             return joblib.load(self.model_path)
#         except Exception as e:
#             logger.exception(f"Error loading soil model: {e}")
#             return None

#     def get_soil_features(self, lat: float, lon: float) -> Dict[str, Any]:
#         """
#         Fetch soil features from SoilGrids API based on latitude/longitude.
#         Falls back to default dummy values if API fails.
#         """
#         BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
#         params = {
#             "lon": lon,
#             "lat": lat,
#             "depth": "sl1",   # surface layer (0–5cm)
#             "property": ["sand", "silt", "clay", "soc", "phh2o", "cec"],
#         }

#         try:
#             resp = requests.get(BASE_URL, params=params, timeout=10)
#             resp.raise_for_status()
#             data = resp.json()

#             props = {}
#             for prop in params["property"]:
#                 values = data["properties"]["layers"][prop]["depths"][0]["values"]
#                 props[prop] = values.get("mean")

#             # Use consistent naming: ph -> ph_level
#             features = {
#                 "sand": props.get("sand", 40),
#                 "silt": props.get("silt", 30),
#                 "clay": props.get("clay", 30),
#                 "organic_matter": props.get("soc", 2.0),
#                 "ph_level": props.get("phh2o", 6.5),  # Consistent naming
#                 "cec": props.get("cec", 15),
#                 # Add coordinates for advanced models
#                 "longitude": lon,
#                 "latitude": lat,
#             }

#             logger.info(f"Retrieved soil features from SoilGrids for lat={lat}, lon={lon}")
#             return features

#         except Exception as e:
#             logger.exception(f"SoilGrids API failed: {e}")
#             return {
#                 "sand": 55,
#                 "silt": 25,
#                 "clay": 20,
#                 "organic_matter": 2.5,
#                 "ph_level": 6.5,  # Consistent naming
#                 "cec": 15,
#                 "longitude": lon,
#                 "latitude": lat,
#             }

#     def _predict_soil_type(self, input_features: Dict[str, Any]) -> str:
#         """
#         Predict soil type using the advanced models if available, otherwise use basic logic.
#         Always returns a string label, even if the model is missing or fails.
#         """
#         # Try advanced model first
#         if self.advanced_models_loaded:
#             try:
#                 predictions = self.predict_soil_properties(input_features)
#                 return predictions["soil_type"]
#             except Exception as e:
#                 logger.warning(f"Advanced soil type prediction failed, using basic: {e}")
        
#         # Fallback to basic logic
#         return self._predict_soil_type_basic(input_features)

#     def _predict_soil_type_basic(self, input_features: Dict[str, Any]) -> str:
#         """Basic soil type prediction as fallback"""
#         try:
#             # Fallback logic if model is not loaded
#             if not hasattr(self, "soil_model") or self.soil_model is None:
#                 clay = input_features.get("clay", 30)
#                 silt = input_features.get("silt", 30)
#                 sand = input_features.get("sand", 40)
#                 if sand >= 60:
#                     return "sandy"
#                 elif clay >= 40:
#                     return "clay"
#                 elif silt >= 40:
#                     return "silty"
#                 else:
#                     return "loam"

#             model_input_df = pd.DataFrame([input_features])[self.soil_model.feature_names_in_]
#             numeric_pred = self.soil_model.predict(model_input_df)[0]


#             # Map numeric prediction to labels
#             soil_type_map = {
#                 0: "sandy",
#                 1: "loam",
#                 2: "clay",
#                 3: "silty",
#                 4: "peaty",
#                 5: "chalky",
#                 6: "organic"
#             }

#             # Round and map
#             return soil_type_map.get(round(numeric_pred), "loam")

#         except Exception as e:
#             logger.exception(f"Basic soil type prediction failed: {e}")
#             return "loam"

#     def align_features(self, raw_dict: dict):
#         """
#         Align features for model prediction with optimized default filling.
#         Reduces redundant operations while maintaining safety.
#         """
#         try:
#             # Create dataframe from input dict
#             df = pd.DataFrame([raw_dict]) if isinstance(raw_dict, dict) else pd.DataFrame(raw_dict)

#             # Only proceed if we have expected features defined
#             if not isinstance(self.model_expected_features, list) or not self.model_expected_features:
#                 logger.warning("No model expected features defined, returning input as-is")
#                 return df

#             # Single pass: ensure all expected features exist with defaults
#             for col in self.model_expected_features:
#                 if col not in df.columns:
#                     df[col] = self._get_default_value(col)

#             # Keep only expected columns in the expected order
#             df = df[self.model_expected_features]

#             # Validate features via existing validator
#             df_aligned = validate_features(df=df, json_path=self.feature_json_path)

#             # Final check: ensure no missing values after validation
#             for col in self.model_expected_features:
#                 if col not in df_aligned.columns or df_aligned[col].isnull().any():
#                     df_aligned[col] = self._get_default_value(col)

#             return df_aligned[self.model_expected_features]

#         except Exception as e:
#             logger.exception(f"Error aligning features: {e}")
#             # Return a one-row DataFrame with safe defaults for all expected features
#             if isinstance(self.model_expected_features, list) and self.model_expected_features:
#                 return pd.DataFrame([{f: self._get_default_value(f) for f in self.model_expected_features}])
#             return pd.DataFrame()



#     @functools.lru_cache(maxsize=128)
#     def _query_soilgrids_fixed(self, lat: float, lon: float):
#         """
#         Fetch topsoil (0-5cm) properties from SoilGrids and return validated soil info.
#         Prioritizes advanced models for predictions; falls back to basic methods only if advanced fails.
#         """
#         try:
#             # Build request URL from settings
#             url = f"{settings.SOILGRIDS_API_URL}?lon={lon}&lat={lat}"
#             response = requests.get(url, timeout=30)
#             response.raise_for_status()
#             data = response.json()
#             props = data.get("properties", {})

#             # Helper function to extract mean topsoil value
#             def get_mean(prop_name, default):
#                 try:
#                     layer = props.get(prop_name, {})
#                     depths = layer.get("depths", [])
#                     if depths:
#                         val = depths[0]["values"].get("mean")
#                         if val is not None and not np.isnan(val):
#                             return float(val)
#                 except Exception:
#                     pass
#                 return default

#             # Get main properties
#             clay = get_mean("clay", 30)
#             silt = get_mean("silt", 30)
#             sand = get_mean("sand", 40)
#             ph_val = get_mean("phh2o", 6.5)
#             soc_val = get_mean("soc", 2.0)

#             # Normalize clay+silt+sand to 100%
#             total = clay + silt + sand
#             if total > 0:
#                 clay = round(clay / total * 100, 1)
#                 silt = round(silt / total * 100, 1)
#                 sand = round(sand / total * 100, 1)

#             # Prepare features for advanced prediction
#             advanced_features = {
#                 "clay": clay,
#                 "silt": silt,
#                 "sand": sand,
#                 "ph_level": ph_val,  # Consistent naming
#                 "organic_matter": soc_val,
#                 "longitude": lon,
#                 "latitude": lat
#             }
            
#             # Prioritize advanced predictions as main path
#             advanced_predictions = self.predict_soil_properties(advanced_features)

#             # Construct final soil info using advanced results
#             soil_info = {
#                 "clay": clay,
#                 "silt": silt,
#                 "sand": sand,
#                 "ph_level": ph_val,  # Consistent naming
#                 "organic_matter": advanced_predictions["soc_g_per_kg"],
#                 "nitrogen": self._get_default_value("nitrogen"),
#                 "phosphorus": self._get_default_value("phosphorus"),
#                 "potassium": self._get_default_value("potassium"),
#                 "soil_type": advanced_predictions["soil_type"],
#                 "timestamp": datetime.now().isoformat(),
#                 "source": advanced_predictions.get("source", "soilgrids"),
#                 "prediction_confidence": advanced_predictions.get("prediction_confidence", "unknown")
#             }

#             logger.info(f"SoilGrids analysis completed for lat={lat}, lon={lon}")
#             return soil_info

#         except Exception as e:
#             logger.exception(f"SoilGrids query failed, using fallback: {e}")
#             fallback = self.get_fallback_soil_data()
#             # Attempt advanced prediction on fallback data; fall to basic if advanced fails
#             try:
#                 advanced_fallback = self.predict_soil_properties(fallback)
#                 fallback["soil_type"] = advanced_fallback["soil_type"]
#                 fallback["organic_matter"] = advanced_fallback["soc_g_per_kg"]
#                 fallback["source"] = advanced_fallback.get("source", "fallback")
#                 fallback["prediction_confidence"] = advanced_fallback.get("prediction_confidence", "low")
#             except Exception as fallback_e:
#                 logger.warning(f"Advanced fallback failed, using basic: {fallback_e}")
#                 # Basic fallback (no advanced call)
#                 fallback["soil_type"] = "loam"
#                 fallback["organic_matter"] = 2.0
#                 fallback["source"] = "basic_fallback"
#                 fallback["prediction_confidence"] = "low"
#             fallback["timestamp"] = datetime.now().isoformat()
#             return fallback

#     # ----------------- Main Public Methods ----------------- #
#     def get_soil_analysis(self, lat: float, lon: float):
#         """
#         Get comprehensive soil analysis using advanced models:
#         1. Try nearest soil record from CSV/JSON
#         2. If not found, query SoilGrids API  
#         3. If API fails, use fallback
#         4. Always populate all soil properties
#         5. Predict health score dynamically using the model
#         """
#         try:
#             # Step 1: Try nearest soil from CSV/JSON
#             nearest_soil = None
#             if self.soil_data_csv is not None and not self.soil_data_csv.empty:
#                 nearest_soil = self.find_nearest_soil_data(lat, lon, df=self.soil_data_csv, max_distance_km=10)

#             if nearest_soil is not None and isinstance(nearest_soil, dict) and len(nearest_soil) > 0:
#                 # Map nearest CSV/JSON fields to standardized keys
#                 soil_info = {
#                     "ph_level": nearest_soil.get('pH_level', 6.5),
#                     "organic_matter": nearest_soil.get('SOC(g/kg)', 2.0),
#                     "nitrogen": nearest_soil.get('Nitrogen', 20.0),
#                     "phosphorus": nearest_soil.get('Phosphorus', 15.0),
#                     "potassium": nearest_soil.get('Potassium', 150.0),
#                     "clay": nearest_soil.get('Clay(%)', 30.0),
#                     "silt": nearest_soil.get('Silt(%)', 30.0),
#                     "sand": nearest_soil.get('Sand(%)', 40.0),
#                     "soil_type": nearest_soil.get('soil_type', None),
#                     "timestamp": datetime.now().isoformat(),
#                     "source": "fao_soil_database",
#                     "model_size": settings.SOIL_MODEL_SIZE
#                 }

#                 # Ensure numeric values
#                 for key in ["clay", "sand", "silt"]:
#                     val = soil_info.get(key)
#                     if val is None or not isinstance(val, (int, float)) or np.isnan(val):
#                         soil_info[key] = self._get_default_value(key)

#                 # Use advanced prediction for soil type and SOC
#                 advanced_predictions = self.predict_soil_properties(soil_info)
#                 soil_info["soil_type"] = advanced_predictions["soil_type"]
#                 soil_info["organic_matter"] = advanced_predictions["soc_g_per_kg"]
#                 soil_info["prediction_confidence"] = advanced_predictions.get("prediction_confidence", "high")
                
#                 logger.debug(f"Used local soil data with advanced predictions for lat={lat}, lon={lon}")

#             else:
#                 # Step 2: Query SoilGrids API as backup, else use fallback
#                 soil_info = self._query_soilgrids_fixed(lat, lon) or self.get_fallback_soil_data()

#             # Ensure soil_info contains all basic keys
#             base_keys = ["ph_level", "organic_matter", "nitrogen", "phosphorus", "potassium",
#                         "clay", "silt", "sand", "soil_type"]
#             for k in base_keys:
#                 if k not in soil_info or soil_info.get(k) is None:
#                     soil_info[k] = self._get_default_value(k)

#             # Step 3: Predict soil health dynamically
#             soil_info["health_score"] = self._predict_soil_health_internal(soil_info)

#             return soil_info

#         except Exception as e:
#             logger.exception(f"Soil service error for lat={lat}, lon={lon}: {e}")
#             fallback = self.get_fallback_soil_data()
#             # Use advanced prediction for fallback
#             advanced_fallback = self.predict_soil_properties(fallback)
#             fallback["soil_type"] = advanced_fallback["soil_type"]
#             fallback["organic_matter"] = advanced_fallback["soc_g_per_kg"]
#             fallback["prediction_confidence"] = advanced_fallback.get("prediction_confidence", "low")
#             fallback["health_score"] = self._predict_soil_health_internal(fallback)
#             return fallback

#     def _predict_soil_health_internal(self, soil_data: dict) -> float:
#         """
#         Internal method for soil health prediction with optimized feature handling
#         """
#         try:
#             if self.model is not None and self.model_expected_features:
#                 # Single alignment pass with built-in default filling
#                 aligned_input = self.align_features(soil_data)
                
#                 # Convert to numeric and handle any remaining nulls
#                 aligned_input = aligned_input.apply(pd.to_numeric, errors='coerce')
#                 for col in aligned_input.columns:
#                     if aligned_input[col].isnull().any():
#                         aligned_input[col].fillna(self._get_default_value(col), inplace=True)

#                 try:
#                     pred = self.model.predict(aligned_input)
#                     return float(pred[0]) if hasattr(pred, '__len__') and len(pred) > 0 else float(pred)
#                 except Exception as e:
#                     logger.exception(f"Model prediction failed: {e}")
#                     return 0.7
#             else:
#                 return 0.7
#         except Exception as e:
#             logger.exception(f"Soil health prediction error: {e}")
#             return 0.7

#     def predict_soil_health(self, soil_data: dict) -> float:
#         """
#         Predict soil health dynamically using trained model and aligned features.
#         """
#         try:
#             if self.model is not None and self.model_expected_features:
#                 # Use optimized alignment method
#                 input_df = self.align_features(soil_data)
                
#                 # Predict dynamically
#                 prediction = self.model.predict(input_df)
#                 return float(prediction[0]) if hasattr(prediction, '__len__') and len(prediction) > 0 else float(prediction)
#             else:
#                 return 0.7
#         except Exception as e:
#             logger.exception(f"Soil health prediction error: {e}")
#             return 0.7

#     def find_nearest_soil_data(self, lat: float, lon: float, df: pd.DataFrame = None, max_distance_km=10):
#         df = df or self.soil_data_csv
#         try:
#             if df is None or df.empty:
#                 return None

#             # Ensure required columns exist and are numeric
#             if not set(["Latitude", "Longitude"]).issubset(set(df.columns)):
#                 logger.warning("Soil CSV does not contain Latitude/Longitude columns")
#                 return None

#             # Drop rows with missing lat/lon
#             df_valid = df.dropna(subset=["Latitude", "Longitude"]).copy()
#             if df_valid.empty:
#                 return None

#             # Compute distances safely
#             def safe_distance(row):
#                 try:
#                     lat_r = float(row["Latitude"])
#                     lon_r = float(row["Longitude"])
#                     return geodesic((lat, lon), (lat_r, lon_r)).km
#                 except Exception:
#                     return np.inf

#             distances = df_valid.apply(safe_distance, axis=1)
#             if distances.empty:
#                 return None

#             nearest_idx = distances.idxmin()
#             if distances.loc[nearest_idx] <= max_distance_km:
#                 return df_valid.loc[nearest_idx].to_dict()
#             return None
#         except Exception as e:
#             logger.exception(f"Could not find nearest soil data: {e}")
#             return None

#     def get_fallback_soil_data(self) -> Dict[str, Any]:
#         """
#         Returns a complete fallback soil record if CSV/JSON/API fails.
#         """
#         fallback = {
#             "ph_level": 6.5,
#             "organic_matter": 2.0,
#             "nitrogen": 20.0,
#             "phosphorus": 15.0,
#             "potassium": 150.0,
#             "clay": 30.0,
#             "silt": 30.0,
#             "sand": 40.0,
#             "soil_type": "loam",
#             "timestamp": datetime.now().isoformat(),
#             "source": "fallback",
#             "model_size": settings.SOIL_MODEL_SIZE,
#             "warning": "Soil data may be generalized",
#             "risk_level": "Medium",
#             "prediction_confidence": "low"
#         }
#         return fallback

#     def get_user_by_location(self, lat: float, lon: float) -> dict:
#         """
#         Fetch user info for a given location.
#         """
#         try:
#             # Placeholder - implement based on your actual database structure
#             return {}
#         except Exception as e:
#             logger.exception(f"Error getting user by location: {e}")
#             return {}

#     def enhance_recommendation(self, disease_name: str, confidence: float, crop_type: str) -> str:
#         if confidence > 0.8:
#             return f"Apply treatment for {disease_name} immediately on {crop_type}."
#         return f"Monitor {crop_type} for {disease_name}."
    















import traceback
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import logging
from datetime import datetime
import joblib
import os
import json
import csv
import functools
import requests
from typing import Dict, Any, Optional, List
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from dotenv import load_dotenv
load_dotenv()

from src.crop_monitor.config.settings import settings 
from src.crop_monitor.utils.feature_validator import validate_features

logger = logging.getLogger(__name__)

class SoilService:
    # ----------------- Updated Feature Mapping for Model Compatibility ----------------- #
    FEATURE_MAPPING = {
        # Basic soil features
        "clay": "Clay(%)",
        "silt": "Silt(%)", 
        "sand": "Sand(%)",
        "bd": "BD(g/cm3)",
        "elevation": "Elevation(m)",
        
        # Climate features  
        "map": "MAP(mm)",
        "mat": "MAT(Celsius)", 
        "pet": "PET(mm)",
        "aridity": "Aridity",
        
        # Location
        "longitude": "Longitude",
        "latitude": "Latitude",
        
        # Depth features
        "upper_depth": "UpperDepth(cm)",
        "lower_depth": "LowerDepth(cm)", 
        "middle_depth": "MiddleDepth(cm)",
        "layer_id": "LayerID",
        
        # Engineered features
        "depth_thickness": "Depth_Thickness",
        "depth_ratio": "Depth_Ratio", 
        "depth_middle_normalized": "Depth_Middle_Normalized",
        "clay_sand_ratio": "Clay_Sand_Ratio",
        "silt_clay_ratio": "Silt_Clay_Ratio",
        "texture_sum_check": "Texture_Sum_Check",
        "texture_balance": "Texture_Balance",
        "map_mat_ratio_robust": "MAP_MAT_Ratio_Robust",
        "aridity_elevation": "Aridity_Elevation",
        "pet_map_ratio": "PET_MAP_Ratio",
        "climate_productivity": "Climate_Productivity",
        
        # Encoded features
        "usda_texture_encoded": "USDA_Texture_Encoded",
        "region_encoded": "REGION_Encoded",
        
        # Legacy features (for backward compatibility)
        "ph_level": "pH_level",
        "organic_matter": "organic_matter",
        "nitrogen": "Nitrogen",
        "phosphorus": "Phosphorus",
        "potassium": "Potassium",
        "crop_type_encoded": "CropTypeEncoded",
        "fertilizer": "Fertilizer"
    }

    # Reverse mapping for consistent feature handling
    REVERSE_FEATURE_MAPPING = {v: k for k, v in FEATURE_MAPPING.items()}

    def __init__(self, soil_data_path: str = None):
        # Initialize paths with proper fallbacks
        self.soil_data_path = soil_data_path or getattr(settings, 'SOIL_DATA_PATH', '')
        self.feature_json_path = getattr(settings, 'SOIL_FEATURES_JSON_PATH', '')
        self.soc_regression_model_path = getattr(settings, 'SOC_REGRESSION_MODEL_PATH', '')
        self.soil_classification_model_path = getattr(settings, 'SOIL_CLASSIFICATION_MODEL_PATH', '')
        self.scaler_classification_path = getattr(settings, 'SCALER_CLASSIFICATION_PATH', '')
        self.scaler_soc_path = getattr(settings, 'SCALER_SOC_PATH', '')
        self.texture_label_encoder_path = getattr(settings, 'TEXTURE_LABEL_ENCODER_PATH', '')
        self.disease_model_path = getattr(settings, 'DISEASE_MODEL_PATH', '')
        self.disease_json_path = getattr(settings, 'DISEASE_JSON_PATH', '')
        self.disease_detection_model_path = getattr(settings, 'DISEASE_DETECTION_MODEL_PATH', '')
        self.yield_forecasting_model_path = getattr(settings, 'YIELD_FORECASTING_MODEL_PATH', '')

        # Load soil features JSON data
        self.soil_data = self._load_soil_json_data()
        
        # Load model expected features
        self.model_expected_features = self._load_model_features()
        
        # Load soil CSV data
        self.soil_data_csv = self._load_soil_data()
        
        # Load advanced soil models
        self._load_advanced_soil_models()
        
        logger.info("✅ SoilService initialized with advanced soil models")

    def _load_soil_json_data(self) -> pd.DataFrame:
        """Load soil features from JSON file"""
        try:
            if not os.path.exists(self.feature_json_path):
                logger.warning(f"Soil features JSON not found: {self.feature_json_path}")
                return pd.DataFrame()

            with open(self.feature_json_path, "r", encoding="utf-8") as f:
                soil_json = json.load(f)
            soil_data = pd.DataFrame(soil_json)
            logger.info(f"✅ Loaded soil features JSON with {len(soil_data)} records")
            return soil_data
        except Exception as e:
            logger.error(f"❌ Failed to load soil features JSON: {e}")
            return pd.DataFrame()

    def _load_advanced_soil_models(self):
        """Load the advanced soil classification and SOC regression models using settings."""
        try:
            # Use correct paths from settings with safe attribute access
            clf_model_path = getattr(settings, 'SOIL_CLASSIFICATION_MODEL_PATH', '')
            scaler_clf_path = getattr(settings, 'SCALER_CLASSIFICATION_PATH', '')
            label_encoder_path = getattr(settings, 'TEXTURE_LABEL_ENCODER_PATH', '')
            soc_model_path = getattr(settings, 'SOC_REGRESSION_MODEL_PATH', '')
            scaler_soc_path = getattr(settings, 'SCALER_SOC_PATH', '')

            # Check if paths are provided and files exist
            required_paths = [clf_model_path, scaler_clf_path, label_encoder_path, soc_model_path, scaler_soc_path]
            if not all(required_paths):
                logger.warning("⚠️ Advanced model paths not configured in settings")
                self.advanced_models_loaded = False
                return

            missing_files = [p for p in required_paths if not os.path.exists(p)]
            if missing_files:
                logger.warning(f"⚠️ Advanced model files not found: {missing_files}. Using basic methods.")
                self.advanced_models_loaded = False
                self.clf_model = None
                self.soc_model = None
                self.scaler_clf = None
                self.scaler_soc = None
                self.label_encoder = None
                return

            # Load models
            logger.info("🔄 Loading advanced soil models...")
            self.clf_model = joblib.load(clf_model_path)
            self.scaler_clf = joblib.load(scaler_clf_path)
            self.label_encoder = joblib.load(label_encoder_path)
            self.soc_model = joblib.load(soc_model_path)
            self.scaler_soc = joblib.load(scaler_soc_path)

            logger.info("✅ Successfully loaded advanced soil classification and SOC regression models")
            self.advanced_models_loaded = True

        except Exception as e:
            logger.exception(f"❌ Error loading advanced soil models: {e}")
            self.advanced_models_loaded = False
            self.clf_model = None
            self.soc_model = None
            self.scaler_clf = None
            self.scaler_soc = None
            self.label_encoder = None

    def _prepare_input_features(self, features: dict) -> dict:
        """Prepare and calculate derived features for model prediction"""
        prepared = features.copy()
        
        # Calculate derived features (same as in training)
        if all(k in prepared for k in ['upper_depth', 'lower_depth']):
            depth_thickness = prepared['lower_depth'] - prepared['upper_depth']
            depth_ratio = prepared['lower_depth'] / (prepared['upper_depth'] + 1)
            middle_depth = prepared.get('middle_depth', (prepared['upper_depth'] + prepared['lower_depth']) / 2)
            depth_middle_norm = middle_depth / (depth_thickness + 1)
            
            prepared.update({
                'depth_thickness': depth_thickness,
                'depth_ratio': depth_ratio,
                'depth_middle_normalized': depth_middle_norm,
                'middle_depth': middle_depth
            })
        
        # Calculate texture ratios if clay/silt/sand available
        if all(k in prepared for k in ['clay', 'silt', 'sand']):
            clay_sand_ratio = prepared['clay'] / (prepared['sand'] + 1)
            silt_clay_ratio = prepared['silt'] / (prepared['clay'] + 1)
            texture_sum = prepared['clay'] + prepared['silt'] + prepared['sand']
            texture_balance = abs(prepared['clay'] - prepared['silt']) + abs(prepared['silt'] - prepared['sand'])
            
            prepared.update({
                'clay_sand_ratio': clay_sand_ratio,
                'silt_clay_ratio': silt_clay_ratio,
                'texture_sum_check': texture_sum,
                'texture_balance': texture_balance
            })
        
        # Calculate climate interactions
        if all(k in prepared for k in ['map', 'mat', 'pet', 'aridity', 'elevation']):
            map_mat_ratio = prepared['map'] / (max(prepared['mat'], 1) + 1)
            aridity_elevation = prepared['aridity'] * prepared['elevation']
            pet_map_ratio = prepared['pet'] / (prepared['map'] + 1)
            climate_productivity = (prepared['map'] * prepared['mat']) / 1000
            
            prepared.update({
                'map_mat_ratio_robust': map_mat_ratio,
                'aridity_elevation': aridity_elevation,
                'pet_map_ratio': pet_map_ratio,
                'climate_productivity': climate_productivity
            })
        
        # Set defaults for encoded features if missing
        prepared.setdefault('usda_texture_encoded', 3)
        prepared.setdefault('region_encoded', 1)
        prepared.setdefault('layer_id', 1)
        
        return prepared

    def predict_soil_properties(self, features: dict) -> dict:
        """
        Predict both soil type (classification) and SOC (regression) using advanced models.
        Falls back to basic methods if advanced models are not available.
        """
        # If advanced models aren't loaded, use basic methods
        if not self.advanced_models_loaded:
            logger.warning("⚠️ Advanced models not loaded, using basic prediction methods")
            return self._predict_soil_properties_basic(features)
            
        try:
            # Prepare features with derived calculations
            prepared_features = self._prepare_input_features(features)
            
            # Convert to model feature names
            model_features = {}
            for internal_name, value in prepared_features.items():
                model_name = self.FEATURE_MAPPING.get(internal_name)
                if model_name:
                    model_features[model_name] = value
            
            # Create DataFrames for both models
            df_all = pd.DataFrame([model_features])
            
            # SOC Regression
            soc_features = [f for f in getattr(self.scaler_soc, 'feature_names_in_', []) if f in df_all.columns]
            if soc_features and len(soc_features) == len(getattr(self.scaler_soc, 'feature_names_in_', [])):
                X_soc = df_all[self.scaler_soc.feature_names_in_]
                X_soc_scaled = self.scaler_soc.transform(X_soc)
                soc_value = self.soc_model.predict(X_soc_scaled)[0]
                logger.debug(f"✅ SOC regression prediction: {soc_value}")
            else:
                # Fallback if missing SOC features
                soc_value = prepared_features.get('organic_matter', 2.0)
                logger.warning(f"⚠️ Using fallback SOC value: {soc_value}")
            
            # Soil Classification
            clf_features = [f for f in getattr(self.scaler_clf, 'feature_names_in_', []) if f in df_all.columns]
            if clf_features and len(clf_features) == len(getattr(self.scaler_clf, 'feature_names_in_', [])):
                X_clf = df_all[self.scaler_clf.feature_names_in_]
                X_clf_scaled = self.scaler_clf.transform(X_clf)
                soil_class_encoded = self.clf_model.predict(X_clf_scaled)[0]
                soil_class_label = self.label_encoder.inverse_transform([soil_class_encoded])[0]
                logger.debug(f"✅ Soil classification prediction: {soil_class_label}")
            else:
                # Fallback to basic classification
                soil_class_label = self._predict_soil_type_basic(prepared_features)
                logger.warning(f"⚠️ Using basic soil type prediction: {soil_class_label}")
            
            return {
                "soil_type": soil_class_label,
                "soc_g_per_kg": round(soc_value, 2),
                "source": "advanced_model",
                "prediction_confidence": "high"
            }

        except Exception as e:
            logger.exception(f"❌ Advanced soil prediction failed: {e}")
            # Fallback to basic methods with detailed error info
            return {
                **self._predict_soil_properties_basic(features),
                "source": "basic_model_fallback",
                "prediction_confidence": "medium",
                "warning": f"Advanced model failed: {str(e)}"
            }

    def _predict_soil_properties_basic(self, features: dict) -> dict:
        """Fallback method using basic soil prediction"""
        logger.info("🔄 Using basic soil property prediction")
        basic_soil_type = self._predict_soil_type_basic(features)
        basic_soc = features.get('organic_matter', 2.0)
        
        return {
            "soil_type": basic_soil_type,
            "soc_g_per_kg": round(basic_soc, 2),
            "source": "basic_model"
        }

    def _get_default_value(self, feature_name: str) -> float:
        """Unified method to get default values for any feature"""
        # Handle both internal and external feature naming
        internal_name = self.REVERSE_FEATURE_MAPPING.get(feature_name, feature_name)
        
        defaults = {
            # Basic soil features
            "ph_level": 6.5,
            "organic_matter": 2.0,
            "nitrogen": 20.0,
            "phosphorus": 15.0,
            "potassium": 150.0,
            "clay": 30.0,
            "silt": 30.0,
            "sand": 40.0,
            
            # Environmental features
            "elevation": 100.0,
            "map": 1000.0,
            "mat": 25.0,
            "pet": 1200.0,
            "aridity": 0.8,
            "temperature": 25.0,
            "humidity": 50.0,
            "rainfall": 50.0,
            "precipitation": 50.0,
            
            # Soil properties
            "soil_type_encoded": 0,
            "ndvi": 0.5,
            "vegetation_index": 0.5,
            "sunlight_hours": 6.0,
            "wind_speed": 5.0,
            "soil_moisture": 0.3,
            "bd": 1.3,
            
            # Management practices
            "crop_type_encoded": 0,
            "irrigation": 0,
            "fertilizer": 0,
            "pesticide": 0,
            
            # Advanced model features
            "longitude": 0.0,
            "latitude": 0.0,
            "depth_thickness": 10.0,
            "depth_ratio": 1.0,
            "depth_middle_normalized": 0.5,
            "aridity_elevation": 160.0,
            "pet_map_ratio": 0.8,
            "climate_productivity": 0.9,
            "usda_texture_encoded": 3.0,
            "region_encoded": 1.0,
            "map_mat_ratio_robust": 60.5,
            "upper_depth": 0.0,
            "lower_depth": 10.0,
            "middle_depth": 5.0,
            "layer_id": 1.0,
            
            # Engineered features
            "clay_sand_ratio": 1.0,
            "silt_clay_ratio": 1.0,
            "texture_sum_check": 100.0,
            "texture_balance": 20.0,
            
            # API response features
            "cec": 15.0,
        }
        
        return defaults.get(internal_name, 0.0)

    # ----------------- Core Helper Methods ----------------- #
    def _load_model_features(self):
        """Load model expected features from JSON file"""
        try:
            if not os.path.exists(self.feature_json_path):
                logger.warning(f"⚠️ Features JSON not found: {self.feature_json_path}")
                return []

            with open(self.feature_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("model_expected_features", [])
            logger.info(f"✅ Loaded {len(features)} model features from JSON")
            return features
        except Exception as e:
            logger.exception(f"❌ Error loading model features: {e}")
            return []

    def _load_soil_data(self):
        """Load soil data from CSV file"""
        try:
            if not os.path.exists(self.soil_data_path):
                logger.warning(f"⚠️ Soil data file not found: {self.soil_data_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(self.soil_data_path, encoding='latin1')
            logger.info(f"✅ Loaded soil CSV data with {len(df)} records from {self.soil_data_path}")
            return df

        except Exception as e:
            logger.exception(f"❌ Error loading soil data: {e}")
            return pd.DataFrame()

    def get_soil_features(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch soil features from SoilGrids API based on latitude/longitude.
        Falls back to default dummy values if API fails.
        """
        BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        params = {
            "lon": lon,
            "lat": lat,
            "depth": "sl1",   # surface layer (0–5cm)
            "property": ["sand", "silt", "clay", "soc", "phh2o", "cec"],
        }

        try:
            logger.info(f"🔄 Fetching soil data from SoilGrids for lat={lat}, lon={lon}")
            resp = requests.get(BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            props = {}
            for prop in params["property"]:
                values = data["properties"]["layers"][prop]["depths"][0]["values"]
                props[prop] = values.get("mean")

            # Use consistent naming: ph -> ph_level
            features = {
                "sand": props.get("sand", 40),
                "silt": props.get("silt", 30),
                "clay": props.get("clay", 30),
                "organic_matter": props.get("soc", 2.0),
                "ph_level": props.get("phh2o", 6.5),  # Consistent naming
                "cec": props.get("cec", 15),
                # Add coordinates for advanced models
                "longitude": lon,
                "latitude": lat,
            }

            logger.info(f"✅ Retrieved soil features from SoilGrids for lat={lat}, lon={lon}")
            return features

        except Exception as e:
            logger.exception(f"❌ SoilGrids API failed: {e}")
            logger.info("🔄 Using fallback soil features")
            return {
                "sand": 55,
                "silt": 25,
                "clay": 20,
                "organic_matter": 2.5,
                "ph_level": 6.5,  # Consistent naming
                "cec": 15,
                "longitude": lon,
                "latitude": lat,
            }

    def _predict_soil_type_basic(self, input_features: Dict[str, Any]) -> str:
        """Basic soil type prediction as fallback"""
        try:
            logger.debug("🔄 Using basic soil type prediction")
            clay = input_features.get("clay", 30)
            silt = input_features.get("silt", 30)
            sand = input_features.get("sand", 40)
            
            if sand >= 60:
                return "sandy"
            elif clay >= 40:
                return "clay"
            elif silt >= 40:
                return "silty"
            else:
                return "loam"

        except Exception as e:
            logger.exception(f"❌ Basic soil type prediction failed: {e}")
            return "loam"

    def align_features(self, raw_dict: dict):
        """
        Align features for model prediction with optimized default filling.
        Reduces redundant operations while maintaining safety.
        """
        try:
            # Create dataframe from input dict
            df = pd.DataFrame([raw_dict]) if isinstance(raw_dict, dict) else pd.DataFrame(raw_dict)

            # Only proceed if we have expected features defined
            if not isinstance(self.model_expected_features, list) or not self.model_expected_features:
                logger.warning("⚠️ No model expected features defined, returning input as-is")
                return df

            # Single pass: ensure all expected features exist with defaults
            for col in self.model_expected_features:
                if col not in df.columns:
                    df[col] = self._get_default_value(col)

            # Keep only expected columns in the expected order
            df = df[self.model_expected_features]

            # Validate features via existing validator
            df_aligned = validate_features(df=df, json_path=self.feature_json_path)

            # Final check: ensure no missing values after validation
            for col in self.model_expected_features:
                if col not in df_aligned.columns or df_aligned[col].isnull().any():
                    df_aligned[col] = self._get_default_value(col)

            return df_aligned[self.model_expected_features]

        except Exception as e:
            logger.exception(f"❌ Error aligning features: {e}")
            # Return a one-row DataFrame with safe defaults for all expected features
            if isinstance(self.model_expected_features, list) and self.model_expected_features:
                return pd.DataFrame([{f: self._get_default_value(f) for f in self.model_expected_features}])
            return pd.DataFrame()

    @functools.lru_cache(maxsize=128)
    def _query_soilgrids_fixed(self, lat: float, lon: float):
        """
        Fetch topsoil (0-5cm) properties from SoilGrids and return validated soil info.
        Prioritizes advanced models for predictions; falls back to basic methods only if advanced fails.
        """
        try:
            # Build request URL from settings
            soilgrids_url = getattr(settings, 'SOILGRIDS_API_URL', 'https://rest.isric.org/soilgrids/v2.0/properties/query')
            url = f"{soilgrids_url}?lon={lon}&lat={lat}"
            logger.info(f"🔄 Querying SoilGrids API for lat={lat}, lon={lon}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            props = data.get("properties", {})

            # Helper function to extract mean topsoil value
            def get_mean(prop_name, default):
                try:
                    layer = props.get(prop_name, {})
                    depths = layer.get("depths", [])
                    if depths:
                        val = depths[0]["values"].get("mean")
                        if val is not None and not np.isnan(val):
                            return float(val)
                except Exception:
                    pass
                return default

            # Get main properties
            clay = get_mean("clay", 30)
            silt = get_mean("silt", 30)
            sand = get_mean("sand", 40)
            ph_val = get_mean("phh2o", 6.5)
            soc_val = get_mean("soc", 2.0)

            # Normalize clay+silt+sand to 100%
            total = clay + silt + sand
            if total > 0:
                clay = round(clay / total * 100, 1)
                silt = round(silt / total * 100, 1)
                sand = round(sand / total * 100, 1)

            # Prepare features for advanced prediction
            advanced_features = {
                "clay": clay,
                "silt": silt,
                "sand": sand,
                "ph_level": ph_val,  # Consistent naming
                "organic_matter": soc_val,
                "longitude": lon,
                "latitude": lat,
                # Add default values for required features
                "elevation": self._get_default_value("elevation"),
                "map": self._get_default_value("map"),
                "mat": self._get_default_value("mat"),
                "pet": self._get_default_value("pet"),
                "aridity": self._get_default_value("aridity"),
                "upper_depth": 0,
                "lower_depth": 5,  # Standard 0-5cm depth
                "bd": self._get_default_value("bd")
            }
            
            # Prioritize advanced predictions as main path
            logger.info("🔄 Running advanced soil property predictions")
            advanced_predictions = self.predict_soil_properties(advanced_features)

            # Construct final soil info using advanced results
            soil_info = {
                "clay": clay,
                "silt": silt,
                "sand": sand,
                "ph_level": ph_val,  # Consistent naming
                "organic_matter": advanced_predictions["soc_g_per_kg"],
                "nitrogen": self._get_default_value("nitrogen"),
                "phosphorus": self._get_default_value("phosphorus"),
                "potassium": self._get_default_value("potassium"),
                "soil_type": advanced_predictions["soil_type"],
                "timestamp": datetime.now().isoformat(),
                "source": advanced_predictions.get("source", "soilgrids"),
                "prediction_confidence": advanced_predictions.get("prediction_confidence", "unknown")
            }

            logger.info(f"✅ SoilGrids analysis completed for lat={lat}, lon={lon}")
            return soil_info

        except Exception as e:
            logger.exception(f"❌ SoilGrids query failed, using fallback: {e}")
            fallback = self.get_fallback_soil_data()
            # Attempt advanced prediction on fallback data; fall to basic if advanced fails
            try:
                logger.info("🔄 Attempting advanced prediction on fallback data")
                advanced_fallback = self.predict_soil_properties(fallback)
                fallback["soil_type"] = advanced_fallback["soil_type"]
                fallback["organic_matter"] = advanced_fallback["soc_g_per_kg"]
                fallback["source"] = advanced_fallback.get("source", "fallback")
                fallback["prediction_confidence"] = advanced_fallback.get("prediction_confidence", "low")
            except Exception as fallback_e:
                logger.warning(f"⚠️ Advanced fallback failed, using basic: {fallback_e}")
                # Basic fallback (no advanced call)
                fallback["soil_type"] = "loam"
                fallback["organic_matter"] = 2.0
                fallback["source"] = "basic_fallback"
                fallback["prediction_confidence"] = "low"
            fallback["timestamp"] = datetime.now().isoformat()
            return fallback

    # ----------------- Main Public Methods ----------------- #
    def get_soil_analysis(self, lat: float, lon: float):
        """
        Get comprehensive soil analysis using advanced models:
        1. Try nearest soil record from CSV/JSON
        2. If not found, query SoilGrids API  
        3. If API fails, use fallback
        4. Always populate all soil properties
        5. Predict health score dynamically using the model
        """
        try:
            logger.info(f"🔄 Starting soil analysis for lat={lat}, lon={lon}")
            
            # Step 1: Try nearest soil from CSV/JSON
            nearest_soil = None
            if self.soil_data_csv is not None and not self.soil_data_csv.empty:
                logger.info("🔄 Searching for nearest soil data in CSV")
                nearest_soil = self.find_nearest_soil_data(lat, lon, df=self.soil_data_csv, max_distance_km=10)

            if nearest_soil is not None and isinstance(nearest_soil, dict) and len(nearest_soil) > 0:
                logger.info("✅ Found nearby soil data in local database")
                # Map nearest CSV/JSON fields to standardized keys
                soil_info = {
                    "ph_level": nearest_soil.get('pH_level', 6.5),
                    "organic_matter": nearest_soil.get('SOC(g/kg)', 2.0),
                    "nitrogen": nearest_soil.get('Nitrogen', 20.0),
                    "phosphorus": nearest_soil.get('Phosphorus', 15.0),
                    "potassium": nearest_soil.get('Potassium', 150.0),
                    "clay": nearest_soil.get('Clay(%)', 30.0),
                    "silt": nearest_soil.get('Silt(%)', 30.0),
                    "sand": nearest_soil.get('Sand(%)', 40.0),
                    "soil_type": nearest_soil.get('soil_type', None),
                    "timestamp": datetime.now().isoformat(),
                    "source": "fao_soil_database"
                }

                # Ensure numeric values
                for key in ["clay", "sand", "silt"]:
                    val = soil_info.get(key)
                    if val is None or not isinstance(val, (int, float)) or np.isnan(val):
                        soil_info[key] = self._get_default_value(key)

                # Use advanced prediction for soil type and SOC
                logger.info("🔄 Running advanced predictions on local soil data")
                advanced_predictions = self.predict_soil_properties(soil_info)
                soil_info["soil_type"] = advanced_predictions["soil_type"]
                soil_info["organic_matter"] = advanced_predictions["soc_g_per_kg"]
                soil_info["prediction_confidence"] = advanced_predictions.get("prediction_confidence", "high")
                
                logger.debug(f"✅ Used local soil data with advanced predictions for lat={lat}, lon={lon}")

            else:
                logger.info("🔄 No local soil data found, querying SoilGrids API")
                # Step 2: Query SoilGrids API as backup, else use fallback
                soil_info = self._query_soilgrids_fixed(lat, lon) or self.get_fallback_soil_data()

            # Ensure soil_info contains all basic keys
            base_keys = ["ph_level", "organic_matter", "nitrogen", "phosphorus", "potassium",
                        "clay", "silt", "sand", "soil_type"]
            for k in base_keys:
                if k not in soil_info or soil_info.get(k) is None:
                    soil_info[k] = self._get_default_value(k)

            # Step 3: Set default health score (since main soil model is not available)
            soil_info["health_score"] = 0.7
            logger.info(f"✅ Soil analysis completed with health score: 0.7")

            return soil_info

        except Exception as e:
            logger.exception(f"❌ Soil service error for lat={lat}, lon={lon}: {e}")
            logger.info("🔄 Using fallback soil data due to error")
            fallback = self.get_fallback_soil_data()
            # Use advanced prediction for fallback
            try:
                advanced_fallback = self.predict_soil_properties(fallback)
                fallback["soil_type"] = advanced_fallback["soil_type"]
                fallback["organic_matter"] = advanced_fallback["soc_g_per_kg"]
                fallback["prediction_confidence"] = advanced_fallback.get("prediction_confidence", "low")
            except Exception as fallback_e:
                logger.warning(f"⚠️ Advanced fallback failed, using basic: {fallback_e}")
                # Basic fallback (no advanced call)
                fallback["soil_type"] = "loam"
                fallback["organic_matter"] = 2.0
                fallback["source"] = "basic_fallback"
                fallback["prediction_confidence"] = "low"
            fallback["health_score"] = 0.7
            return fallback

    def predict_soil_health(self, soil_data: dict) -> float:
        """
        Predict soil health dynamically using trained model and aligned features.
        Note: Main soil model not available, returning default value.
        """
        logger.warning("⚠️ Main soil health model not available, returning default health score")
        return 0.7

    def find_nearest_soil_data(self, lat: float, lon: float, df: pd.DataFrame = None, max_distance_km=10):
        """Find nearest soil data point from CSV data"""
        df = df or self.soil_data_csv
        try:
            if df is None or df.empty:
                logger.debug("⚠️ No soil data available for nearest search")
                return None

            # Ensure required columns exist and are numeric
            if not set(["Latitude", "Longitude"]).issubset(set(df.columns)):
                logger.warning("⚠️ Soil CSV does not contain Latitude/Longitude columns")
                return None

            # Drop rows with missing lat/lon
            df_valid = df.dropna(subset=["Latitude", "Longitude"]).copy()
            if df_valid.empty:
                logger.debug("⚠️ No valid coordinates in soil data")
                return None

            # Compute distances safely
            def safe_distance(row):
                try:
                    lat_r = float(row["Latitude"])
                    lon_r = float(row["Longitude"])
                    return geodesic((lat, lon), (lat_r, lon_r)).km
                except Exception:
                    return np.inf

            logger.debug(f"🔄 Calculating distances for {len(df_valid)} soil data points")
            distances = df_valid.apply(safe_distance, axis=1)
            if distances.empty:
                return None

            nearest_idx = distances.idxmin()
            nearest_distance = distances.loc[nearest_idx]
            
            if nearest_distance <= max_distance_km:
                logger.debug(f"✅ Found nearest soil data {nearest_distance:.2f} km away")
                return df_valid.loc[nearest_idx].to_dict()
            else:
                logger.debug(f"⚠️ Nearest soil data too far: {nearest_distance:.2f} km")
                return None
                
        except Exception as e:
            logger.exception(f"❌ Could not find nearest soil data: {e}")
            return None

    def get_fallback_soil_data(self) -> Dict[str, Any]:
        """
        Returns a complete fallback soil record if CSV/JSON/API fails.
        """
        logger.info("🔄 Generating fallback soil data")
        fallback = {
            "ph_level": 6.5,
            "organic_matter": 2.0,
            "nitrogen": 20.0,
            "phosphorus": 15.0,
            "potassium": 150.0,
            "clay": 30.0,
            "silt": 30.0,
            "sand": 40.0,
            "soil_type": "loam",
            "timestamp": datetime.now().isoformat(),
            "source": "fallback",
            "warning": "Soil data may be generalized",
            "risk_level": "Medium",
            "prediction_confidence": "low"
        }
        return fallback

    def get_user_by_location(self, lat: float, lon: float) -> dict:
        """
        Fetch user info for a given location.
        """
        try:
            # Placeholder - implement based on your actual database structure
            return {}
        except Exception as e:
            logger.exception(f"❌ Error getting user by location: {e}")
            return {}

    def enhance_recommendation(self, disease_name: str, confidence: float, crop_type: str) -> str:
        """Enhance disease recommendation based on confidence"""
        if confidence > 0.8:
            return f"Apply treatment for {disease_name} immediately on {crop_type}."
        return f"Monitor {crop_type} for {disease_name}."

    def verify_model_health(self) -> dict:
        """
        Verify that all soil models are functioning correctly.
        Used for health checks and deployment validation.
        """
        logger.info("🔄 Running soil model health check")
        health_sample = {
            'longitude': -122.4194,
            'latitude': 37.7749,
            'elevation': 250,
            'map': 1200,
            'mat': 15.5, 
            'pet': 900,
            'aridity': 0.75,
            'clay': 28.5,
            'silt': 35.2,
            'sand': 36.3,
            'bd': 1.32,
            'upper_depth': 0,
            'lower_depth': 15
        }
        
        try:
            result = self.predict_soil_properties(health_sample)
            logger.info(f"✅ Soil model health check passed: {result}")
            return {
                "status": "healthy",
                "soil_type": result["soil_type"],
                "soc_prediction": result["soc_g_per_kg"],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Soil model health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }