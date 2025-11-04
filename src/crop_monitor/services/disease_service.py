# # src/crop_monitor/services/disease_service.py
# import logging
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import os
# import json
# from fastapi import UploadFile
# from typing import Dict, Any, List
# import csv
# from tensorflow.keras.applications.efficientnet import preprocess_input
# import asyncio

# logger = logging.getLogger(__name__)

# from src.crop_monitor.services import satellite_service
# from src.crop_monitor.services.satellite_service import SatelliteService

# class DiseaseService:
#     def __init__(self, model_path: str = None, class_file: str = None):
#         # Use provided paths or default hardcoded paths
#         self.model_path = model_path or "src/crop_monitor/models/plantvillage_efficientnet.tflite"
#         self.class_file = class_file or "src/crop_monitor/models/plantvillage_classes.json"

#         self.interpreter = self._load_model()  # TFLite interpreter

#         # Get input/output details for TFLite
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()

#         # Input size
#         self.input_height = self.input_details[0]['shape'][1]  # 128
#         self.input_width = self.input_details[0]['shape'][2]   # 128
#         self.input_channels = self.input_details[0]['shape'][3]  # 3
#         self.confidence_threshold = 0.5

#         # Load class names from JSON
#         self.class_names = self._load_class_names(self.class_file)

#     def _load_model(self):
#         """Load the disease detection TFLite model"""
#         try:
#             if not os.path.exists(self.model_path):
#                 raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
#             logger.info(f"Loading disease TFLite model from: {self.model_path}")
#             interpreter = tf.lite.Interpreter(model_path=self.model_path)
#             interpreter.allocate_tensors()
#             logger.info("✅ TFLite model loaded successfully!")
#             return interpreter
#         except Exception as e:
#             logger.error(f"Failed to load disease TFLite model: {e}")
#             raise

#     def _load_class_names(self, class_file: str):
#         """Load class names from a JSON file with safe parsing and logging."""
#         if not os.path.exists(class_file):
#             logger.error(f"Class names JSON not found: {class_file}")
#             raise FileNotFoundError(f"Class names JSON not found: {class_file}")
        
#         try:
#             with open(class_file, 'r', encoding='utf-8') as f:
#                 try:
#                     class_names = json.load(f)
#                     if not isinstance(class_names, list):
#                         logger.warning(f"Class names JSON is not a list. Resetting to default ['Healthy']")
#                         class_names = ["Healthy"]
#                 except json.JSONDecodeError as e:
#                     logger.error(f"Malformed class names JSON at {class_file}: {e}")
#                     class_names = ["Healthy"]
#             logger.info(f"✅ Loaded {len(class_names)} class names from {class_file}")
#             return class_names
#         except Exception as e:
#             logger.error(f"Unexpected error loading class names: {e}")
#             return ["Healthy"]

#     async def predict_disease_unified(
#         self,
#         files,                             # Accept either a single UploadFile or a list of UploadFile
#         lats: list[float] = None,          # Latitude(s)
#         lons: list[float] = None,          # Longitude(s)
#         soil_service=None,
#         weather_service=None,
#         db_sqlite_conn=None,
#         mongo_collection=None,
#         output_csv_path: str = "predictions.csv"
#     ) -> list[dict]:
#         """
#         Unified disease prediction function.
#         Handles both single and batch predictions without altering existing logic.
#         Automatically fetches soil and weather data for risk_level calculation.
#         Can save to SQLite or MongoDB, and export batch results to CSV.
#         Soil, weather, satellite API calls are wrapped in asyncio.to_thread to avoid blocking.
#         """

#         # Ensure we have a list for unified processing
#         if not isinstance(files, list):
#             files = [files]

#         results = []

#         # Determine if CSV exists already
#         csv_exists = os.path.exists(output_csv_path)

#         for idx, file in enumerate(files):
#             lat, lon = (None, None)
#             if lats and lons and idx < len(lats):
#                 lat = lats[idx]
#                 lon = lons[idx]
#             elif len(files) == 1 and lats and lons:
#                 lat, lon = lats[0], lons[0]
#             try:
#                 # --- Read image ---
#                 contents = await file.read()
#                 image = Image.open(BytesIO(contents)).convert('RGB')
#                 image = image.resize((self.input_width, self.input_height))
#                 img_array = np.array(image).astype(np.float32)
#                 img_array = preprocess_input(img_array)
#                 img_array = np.expand_dims(img_array, axis=0)

#                 # --- Run inference ---
#                 self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
#                 self.interpreter.invoke()
#                 pred = self.interpreter.get_tensor(self.output_details[0]['index'])

#                 pred_index = int(np.argmax(pred))
#                 confidence = float(pred[0][pred_index])
#                 disease_name = self.class_names[pred_index] if pred_index < len(self.class_names) else "Unknown"

#                 # --- Fetch environmental data per file ---
#                 env_data = {"soil_health": 0.6, "humidity": 75}
#                 soil_info = {}
#                 satellite_info = {}
#                 weather_info = {}

#                 if lat is not None and lon is not None:
#                     try:
#                         if soil_service:
#                             soil_info = await asyncio.to_thread(soil_service.get_soil_analysis, lat, lon)
#                             env_data["soil_health"] = soil_info.get("health_score", 0.6)
#                         if weather_service:
#                             weather_info = await asyncio.to_thread(weather_service.get_weather_data, lat, lon)
#                             env_data["humidity"] = weather_info.get("humidity", 75)
#                         if satellite_service:
#                             satellite_client = SatelliteService()
#                             satellite_info = await asyncio.to_thread(satellite_client.get_vegetation_data, lat, lon)
#                     except Exception as api_e:
#                         logger.warning(f"Failed to fetch env data for {file.filename}: {api_e}")

#                 # --- Build prediction dict ---
#                 prediction = {
#                     "filename": file.filename,
#                     "disease": disease_name,
#                     "confidence": confidence,
#                     "risk_level": self.derive_risk_level(confidence, env_data),
#                     "soil": soil_info,
#                     "satellite": satellite_info,
#                     "weather": weather_info,
#                     "low_confidence": confidence < self.confidence_threshold
#                 }
#                 results.append(prediction)


                                
#                 # --- Add recommendations with confidence handling ---
#                 prediction["recommendation"] = self.enhance_recommendation(
#                     disease=disease_name,
#                     weather_data=weather_info,
#                     soil_data=soil_info,
#                     satellite_data=satellite_info,
#                     confidence=confidence
#                 )

#                 results.append(prediction)


#                 # --- Save to SQLite ---
#                 if db_sqlite_conn:
#                     try:
#                         cursor = db_sqlite_conn.cursor()
#                         cursor.execute("""
#                             CREATE TABLE IF NOT EXISTS predictions (
#                                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                                 filename TEXT,
#                                 disease TEXT,
#                                 confidence REAL,
#                                 risk_level TEXT
#                             )
#                         """)
#                         cursor.execute("""
#                             INSERT INTO predictions (filename, disease, confidence, risk_level)
#                             VALUES (?, ?, ?, ?)
#                         """, (
#                             prediction["filename"],
#                             prediction["disease"],
#                             prediction["confidence"],
#                             prediction["risk_level"]
#                         ))
#                         db_sqlite_conn.commit()
#                     except Exception as e:
#                         logger.error(f"SQLite insert error for {file.filename}: {e}")

#                 # --- Save to MongoDB ---
#                 if mongo_collection is not None:
#                     try:
#                         mongo_collection.insert_one(prediction)
#                     except Exception as e:
#                         logger.error(f"MongoDB insert error for {file.filename}: {e}")

#             except Exception as e:
#                 logger.error(f"Prediction error for {file.filename}: {e}")

#                 # --- Attempt to derive risk_level even on failure ---
#                 try:
#                     env_data = {"soil_health": 0.6, "humidity": 75}
#                     if lat is not None and lon is not None and soil_service:
#                         soil_info = await asyncio.to_thread(soil_service.get_soil_analysis, lat, lon)
#                         env_data["soil_health"] = soil_info.get("health_score", 0.5)
#                         if weather_service:
#                             env_data["humidity"] = await asyncio.to_thread(weather_service.get_humidity, lat, lon)
#                     risk_level = self.derive_risk_level(0.0, env_data)
#                 except Exception as inner_e:
#                     logger.warning(f"Failed to derive risk_level for failed prediction: {inner_e}")
#                     risk_level = "Unknown"

#                 failed_prediction = {
#                     "filename": file.filename,
#                     "disease": "Prediction failed",
#                     "confidence": 0.0,
#                     "risk_level": risk_level
#                 }
#                 results.append(failed_prediction)

#                 # --- Save failed prediction to SQLite ---
#                 if db_sqlite_conn:
#                     try:
#                         cursor = db_sqlite_conn.cursor()
#                         cursor.execute("""
#                             CREATE TABLE IF NOT EXISTS predictions (
#                                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                                 filename TEXT,
#                                 disease TEXT,
#                                 confidence REAL,
#                                 risk_level TEXT
#                             )
#                         """)
#                         cursor.execute("""
#                             INSERT INTO predictions (filename, disease, confidence, risk_level)
#                             VALUES (?, ?, ?, ?)
#                         """, (
#                             failed_prediction["filename"],
#                             failed_prediction["disease"],
#                             failed_prediction["confidence"],
#                             failed_prediction["risk_level"]
#                         ))
#                         db_sqlite_conn.commit()
#                     except Exception as e:
#                         logger.error(f"SQLite insert error for failed prediction: {e}")

#                 # --- Save failed prediction to MongoDB ---
#                 if mongo_collection is not None:
#                     try:
#                         mongo_collection.insert_one(failed_prediction)
#                     except Exception as e:
#                         logger.error(f"MongoDB insert error for failed prediction: {e}")

#         # --- Export to CSV (append mode, dynamic header handling) ---
#         csv_columns = ["filename", "disease", "confidence", "risk_level"]
#         try:
#             with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
#                 writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
#                 if not csv_exists:
#                     writer.writeheader()
#                 for data in results:
#                     try:
#                         writer.writerow({k: data.get(k, "") for k in csv_columns})
#                     except Exception as e:
#                         logger.error(f"Failed to write CSV row for {data.get('filename', 'unknown')}: {e}")
#             logger.info(f"Predictions exported to CSV: {output_csv_path}")
#         except Exception as e:
#             logger.error(f"Failed to export CSV {output_csv_path}: {e}")

#         return results

#     # =======================
#     # Convenience wrapper for FastAPI endpoint
#     # =======================
#     async def predict_disease_endpoint(
#         self,
#         file: UploadFile = None,
#         files: list[UploadFile] = None,
#         lat: float = None,
#         lon: float = None,
#         lats: list[float] = None,
#         lons: list[float] = None,
#         soil_service=None,
#         weather_service=None,
#         db_sqlite_conn=None,
#         mongo_collection=None,
#         output_csv_path: str = "predictions.csv"
#     ) -> list[dict]:
#         """
#         Convenience wrapper that lets you call the endpoint with either
#         `file=` or `files=`. Calls predict_disease_unified internally.
#         """
#         if file and not files:
#             return await self.predict_disease_unified(
#                 files=file,
#                 lats=[lat] if lat is not None else None,
#                 lons=[lon] if lon is not None else None,
#                 soil_service=soil_service,
#                 weather_service=weather_service,
#                 db_sqlite_conn=db_sqlite_conn,
#                 mongo_collection=mongo_collection,
#                 output_csv_path=output_csv_path
#             )
#         elif files:
#             return await self.predict_disease_unified(
#                 files=files,
#                 lats=lats,
#                 lons=lons,
#                 soil_service=soil_service,
#                 weather_service=weather_service,
#                 db_sqlite_conn=db_sqlite_conn,
#                 mongo_collection=mongo_collection,
#                 output_csv_path=output_csv_path
#             )
#         else:
#             raise ValueError("You must provide either `file` or `files` to predict.")

#     def _safe_json_load(self, value: str, context: str = "") -> Any:
#         """Attempt to parse JSON string; log errors and return empty dict on failure."""
#         if not value:
#             return {}
#         try:
#             return json.loads(value)
#         except json.JSONDecodeError as e:
#             logger.warning(f"Malformed JSON in {context}: {e}. Returning empty dict.")
#             # Attempt to fix single quotes
#             try:
#                 fixed = value.replace("'", '"')
#                 return json.loads(fixed)
#             except Exception:
#                 return {}

#     def derive_risk_level(self, confidence: float, env: dict) -> str:
#         """Assign risk level based on confidence + environmental factors"""
#         try:
#             risk_score = confidence
#             if env.get("soil_health") and env["soil_health"] < 0.5:
#                 risk_score += 0.1
#             if env.get("humidity") and env["humidity"] > 70:
#                 risk_score += 0.1
#             if risk_score < 0.4:
#                 return "Low"
#             elif risk_score < 0.7:
#                 return "Medium"
#             else:
#                 return "High"
#         except Exception:
#             return "Medium"  # fallback


#     # ------------------ Fully Model-driven, Multi-factor & Comprehensive Recommendation Function ------------------
#     def enhance_recommendation(
#             self,
#             disease: str,
#             weather_data: dict = None,
#             soil_data: dict = None,
#             satellite_data: dict = None,
#             disease_recommendation_map: dict = None,
#             confidence: float = 1.0
#         ) -> str:

#         # ⚠️ Low-confidence check
#         if confidence < self.confidence_threshold:
#             return f"⚠️ Low-confidence prediction for {disease} – treat recommendations cautiously."

#         """
#         Generate extremely comprehensive recommendations for a predicted disease.
        
#         This function integrates:
#         - Weather conditions: humidity, rainfall, temperature, wind
#         - Soil chemistry & health: pH, NPK, organic matter, soil health score
#         - Satellite vegetation indices: NDVI, health_status
#         - Disease-specific guidance: PlantVillage-trained classification, dynamic mapping, or model-driven
#         - Multi-level risk assessment and tailored intervention guidance
#         """

#         recommendations = [f"Treatment guidance for {disease}"]
#         disease_lower = disease.lower()

#         # =====================
#         # 1. Weather-based adjustments
#         # =====================
#         if weather_data:
#             humidity = weather_data.get("humidity", 0)
#             rain = weather_data.get("rain_last_hour", 0)
#             temp = weather_data.get("temperature", 0)
#             wind_speed = weather_data.get("wind_speed", 0)
#             leaf_wetness = weather_data.get("leaf_wetness", 0)

#             # Humidity-based risk
#             if humidity > 90:
#                 recommendations.append("Extremely high humidity - increase fungicide frequency and monitor leaf wetness.")
#             elif humidity > 75:
#                 recommendations.append("High humidity - consider adjusting spray schedule to prevent fungal outbreaks.")
#             elif humidity < 40:
#                 recommendations.append("Low humidity - monitor irrigation and reduce powdery mildew risk.")

#             # Rainfall adjustment
#             if rain > 15:
#                 recommendations.append("Heavy rainfall detected - delay chemical application, ensure leaves are dry before spraying.")
#             elif rain > 5:
#                 recommendations.append("Moderate rainfall - ensure field drainage and adjust irrigation schedule.")
#             elif rain == 0:
#                 recommendations.append("No recent rainfall - irrigation may be needed to maintain plant health.")

#             # Temperature adjustment
#             if temp > 35:
#                 recommendations.append("High temperature - schedule spraying early morning or late evening to avoid phytotoxicity.")
#             elif temp < 12:
#                 recommendations.append("Low temperature - disease progression may be slowed, adjust treatment accordingly.")

#             # Wind adjustment
#             if wind_speed > 25:
#                 recommendations.append("High wind detected - take precautions to prevent spray drift.")

#             # Leaf wetness
#             if leaf_wetness > 0.8:
#                 recommendations.append("High leaf wetness - increased risk of fungal disease, adjust spraying and drainage.")

#         # =====================
#         # 2. Soil-based adjustments
#         # =====================
#         if soil_data:
#             ph = soil_data.get("ph_level", 6.5)
#             nitrogen = soil_data.get("nitrogen", 0)
#             phosphorus = soil_data.get("phosphorus", 0)
#             potassium = soil_data.get("potassium", 0)
#             organic_matter = soil_data.get("organic_matter", 0.5)
#             health_score = soil_data.get("health_score", 0.5)
#             moisture = soil_data.get("moisture", 0.5)

#             # pH
#             if ph < 5.5:
#                 recommendations.append("Acidic soil detected - consider liming to optimize nutrient availability.")
#             elif ph > 7.5:
#                 recommendations.append("Alkaline soil detected - consider acidifying amendments for optimal nutrient uptake.")

#             # Macronutrients
#             if nitrogen < 20:
#                 recommendations.append("Nitrogen deficient - apply suitable nitrogen fertilizer.")
#             if phosphorus < 15:
#                 recommendations.append("Phosphorus deficient - apply phosphate fertilizer.")
#             if potassium < 120:
#                 recommendations.append("Potassium deficient - apply potash fertilizer.")

#             # Organic matter & health
#             if organic_matter < 3.0:
#                 recommendations.append("Low organic matter - consider compost or green manure.")
#             if health_score < 0.5:
#                 recommendations.append("Overall soil health low - integrate soil amendments and crop rotation.")

#             # Soil moisture
#             if moisture < 0.3:
#                 recommendations.append("Soil moisture low - irrigation recommended.")
#             elif moisture > 0.8:
#                 recommendations.append("Soil overly wet - improve drainage to prevent root disease.")

#         # =====================
#         # 3. Satellite-based adjustments
#         # =====================
#         if satellite_data:
#             ndvi = satellite_data.get("ndvi", 0)
#             health_status = satellite_data.get("health_status", "").lower()
#             if ndvi < 0.3 or health_status in ["poor", "stressed", "dry"]:
#                 recommendations.append("Satellite data indicates stressed vegetation - adjust irrigation, fertilization, and monitor pests.")

#             if ndvi > 0.8 and health_status in ["healthy", "good"]:
#                 recommendations.append("Satellite NDVI indicates healthy vegetation - maintain current management practices.")

#         # Define the disease_recommendation_map from the PlantVillage dataset
#         disease_recommendation_map = {
#             "apple___apple_scab": "Prune affected areas; apply fungicides.",
#             "apple___black_rot": "Remove and destroy infected leaves and fruits; apply fungicides.",
#             "apple___cedar_apple_rust": "Remove and destroy infected leaves; apply fungicides.",
#             "apple___healthy": "No treatment necessary.",
#             "blueberry___healthy": "No treatment necessary.",
#             "cherry_(including_sour)___healthy": "No treatment necessary.",
#             "cherry_(including_sour)___powdery_mildew": "Prune affected areas; apply fungicides.",
#             "corn_(maize)___cercospora_leaf_spot gray_leaf_spot": "Remove and destroy infected leaves; apply fungicides.",
#             "corn_(maize)___common_rust_": "Remove and destroy infected leaves; apply fungicides.",
#             "corn_(maize)___northern_leaf_blight": "Remove and destroy infected leaves; apply fungicides.",
#             "corn_(maize)___healthy": "No treatment necessary.",
#             "grape___black_rot": "Remove and destroy infected leaves and fruits; apply fungicides.",
#             "grape___esca_(black_measles)": "Prune affected areas; apply fungicides.",
#             "grape___leaf_blight_(isariopsis_leaf_spot)": "Remove and destroy infected leaves; apply fungicides.",
#             "grape___healthy": "No treatment necessary.",
#             "orange___haunglongbing_(citrus_greening)": "Remove and destroy infected plants; apply systemic insecticides.",
#             "peach___bacterial_spot": "Remove and destroy infected plants; apply copper-based bactericides.",
#             "peach___healthy": "No treatment necessary.",
#             "pepper,_bell___bacterial_spot": "Remove and destroy infected plants; apply copper-based bactericides.",
#             "pepper,_bell___healthy": "No treatment necessary.",
#             "potato___early_blight": "Remove and destroy infected leaves; apply fungicides.",
#             "potato___late_blight": "Remove and destroy infected plants; apply fungicides.",
#             "potato___healthy": "No treatment necessary.",
#             "raspberry___healthy": "No treatment necessary.",
#             "soybean___healthy": "No treatment necessary.",
#             "squash___powdery_mildew": "Prune affected areas; apply fungicides.",
#             "strawberry___leaf_scorch": "Remove and destroy infected leaves; apply fungicides.",
#             "strawberry___healthy": "No treatment necessary.",
#             "tomato___bacterial_spot": "Remove and destroy infected plants; apply copper-based bactericides.",
#             "tomato___early_blight": "Remove and destroy infected leaves; apply fungicides.",
#             "tomato___late_blight": "Remove and destroy infected plants; apply fungicides.",
#             "tomato___leaf_mold": "Remove and destroy infected leaves; apply fungicides.",
#             "tomato___septoria_leaf_spot": "Remove and destroy infected leaves; apply fungicides.",
#             "tomato___spider_mites two-spotted_spider_mite": "Apply miticides.",
#             "tomato___target_spot": "Remove and destroy infected leaves; apply fungicides.",
#             "tomato___tomato_mosaic_virus": "Remove and destroy infected plants; control vectors with insecticides.",
#             "tomato___tomato_yellow_leaf_curl_virus": "Remove and destroy infected plants; apply insecticides.",
#             "tomato___healthy": "No treatment necessary.",
#         }

#         # =====================
#         # 4. Disease-specific recommendations
#         # =====================
#         if disease_recommendation_map and disease_lower in disease_recommendation_map:
#             recommendations.append(disease_recommendation_map[disease_lower])
#         else:
#             # PlantVillage-inspired rules (retained as fallback for non-mapped diseases)
#             if "healthy" in disease_lower:
#                 recommendations.append("No intervention required, continue regular monitoring.")
#             elif "late blight" in disease_lower:
#                 recommendations.append("Late Blight detected - apply fungicide, remove infected tissue, and monitor neighboring plants.")
#             elif "early blight" in disease_lower:
#                 recommendations.append("Early Blight detected - apply recommended fungicide, maintain crop hygiene, rotate crops if possible.")
#             elif "powdery mildew" in disease_lower:
#                 recommendations.append("Powdery mildew detected - use sulfur-based fungicides and avoid high humidity areas.")
#             elif "rust" in disease_lower:
#                 recommendations.append("Rust detected - apply appropriate fungicide and remove infected leaves.")
#             elif "mosaic" in disease_lower:
#                 recommendations.append("Mosaic virus detected - remove infected plants, control vector insects, and implement resistant cultivars.")
#             elif "rot" in disease_lower or "mold" in disease_lower:
#                 recommendations.append("Rot or mold detected - improve drainage, remove infected tissue, and apply suitable treatment.")
#             else:
#                 recommendations.append(f"General recommendation: monitor crop, adjust environment, and consult agronomist for {disease}.")

#         # =====================
#         # 5. Multi-factor risk & safety consolidation
#         # =====================
#         # Remove duplicates, prioritize severe warnings at the start
#         severity_keywords = ["extremely", "heavy", "high", "low", "critical"]
#         recommendations_sorted = sorted(
#             list(dict.fromkeys(recommendations)),
#             key=lambda x: any(k in x.lower() for k in severity_keywords),
#             reverse=True
#         )
#         return " | ".join(recommendations_sorted)





# src/crop_monitor/services/disease_service.py
import logging
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import os
import json
from fastapi import UploadFile
from typing import Dict, Any, List
import csv
from tensorflow.keras.applications.efficientnet import preprocess_input
import asyncio
from scipy.special import softmax

logger = logging.getLogger(__name__)

from src.crop_monitor.config.settings import settings
from src.crop_monitor.services import satellite_service
from src.crop_monitor.services.satellite_service import SatelliteService

class DiseaseService:
    def __init__(self, model_path: str = None, class_file: str = None):
        # Use provided paths or default hardcoded paths
        self.model_path = model_path or settings.DISEASE_MODEL_PATH
        self.class_file = class_file or settings.DISEASE_JSON_PATH

        self.interpreter = self._load_model()  # TFLite interpreter

        # Get input/output details for TFLite
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Log input shape for confirmation
        logger.info(f"Input shape: {self.input_details[0]['shape']}")

        # Input size
        self.input_height = self.input_details[0]['shape'][1]  # 128
        self.input_width = self.input_details[0]['shape'][2]   # 128
        self.input_channels = self.input_details[0]['shape'][3]  # 3
        self.confidence_threshold = 0.5

        # Load class names from JSON
        self.class_names = self._load_class_names(self.class_file)

    def _load_model(self):
        """Load the disease detection TFLite model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading disease TFLite model from: {self.model_path}")
            interpreter = tf.lite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            logger.info("✅ TFLite model loaded successfully!")
            return interpreter
        except Exception as e:
            logger.error(f"Failed to load disease TFLite model: {e}")
            raise

    def _load_class_names(self, class_file: str):
        """Load class names from a JSON file with safe parsing and logging."""
        if not os.path.exists(class_file):
            logger.error(f"Class names JSON not found: {class_file}")
            raise FileNotFoundError(f"Class names JSON not found: {class_file}")
        
        try:
            with open(class_file, 'r', encoding='utf-8') as f:
                try:
                    class_names = json.load(f)
                    if not isinstance(class_names, list):
                        logger.warning(f"Class names JSON is not a list. Resetting to default ['Healthy']")
                        class_names = ["Healthy"]
                except json.JSONDecodeError as e:
                    logger.error(f"Malformed class names JSON at {class_file}: {e}")
                    class_names = ["Healthy"]
            logger.info(f"✅ Loaded {len(class_names)} class names from {class_file}")
            return class_names
        except Exception as e:
            logger.error(f"Unexpected error loading class names: {e}")
            return ["Healthy"]
        

    def clear_prediction_cache(self):
        """Clear any prediction caches to prevent same results"""
        if hasattr(self, '_prediction_cache'):
            self._prediction_cache.clear()
        logger.info("✅ Prediction cache cleared")   

    async def predict_disease_unified(
        self,
        files,                             # Accept either a single UploadFile or a list of UploadFile
        lats: list[float] = None,          # Latitude(s)
        lons: list[float] = None,          # Longitude(s)
        soil_service=None,
        weather_service=None,
        db_sqlite_conn=None,
        mongo_collection=None,
        output_csv_path: str = "predictions.csv"
    ) -> list[dict]:
        """
        Unified disease prediction function.
        Handles both single and batch predictions without altering existing logic.
        Automatically fetches soil and weather data for risk_level calculation.
        Can save to SQLite or MongoDB, and export batch results to CSV.
        Soil, weather, satellite API calls are wrapped in asyncio.to_thread to avoid blocking.
        """

        # Ensure we have a list for unified processing
        if not isinstance(files, list):
            files = [files]

        results = []

        # Determine if CSV exists already
        csv_exists = os.path.exists(output_csv_path)

        for idx, file in enumerate(files):
            lat, lon = (None, None)
            if lats and lons and idx < len(lats):
                lat = lats[idx]
                lon = lons[idx]
            elif len(files) == 1 and lats and lons:
                lat, lon = lats[0], lons[0]
            try:
                # --- Read image ---
                contents = await file.read()
                image = Image.open(BytesIO(contents)).convert('RGB')
                # Validate image
                if image.size[0] == 0 or image.size[1] == 0:
                    raise ValueError("Invalid image")
                image = image.resize((self.input_width, self.input_height))
                img_array = np.array(image).astype(np.float32)
                img_array = preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                # --- Run inference ---
                self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
                self.interpreter.invoke()
                pred_logits = self.interpreter.get_tensor(self.output_details[0]['index'])
                pred_probs = softmax(pred_logits[0])  # Apply softmax to first (and only) sample
                pred_index = int(np.argmax(pred_probs))
                confidence = float(pred_probs[pred_index])  # Now a true probability (0-1)
                disease_name = self.class_names[pred_index] if pred_index < len(self.class_names) else "Unknown"
                # Log prediction details
                logger.debug(f"Pred index: {pred_index}, raw max: {np.max(pred_logits)}, probs max: {confidence}")

                # --- Fetch environmental data per file ---
                env_data = {"soil_health": 0.6, "humidity": 75}
                soil_info = {}
                satellite_info = {}
                weather_info = {}

                if lat is not None and lon is not None:
                    try:
                        if soil_service:
                            soil_info = await asyncio.to_thread(soil_service.get_soil_analysis, lat, lon)
                            env_data["soil_health"] = soil_info.get("health_score", 0.6)
                        if weather_service:
                            weather_info = await asyncio.to_thread(weather_service.get_weather_data, lat, lon)
                            env_data["humidity"] = weather_info.get("humidity", 75)
                        if satellite_service:
                            satellite_client = SatelliteService()
                            satellite_info = await asyncio.to_thread(satellite_client.get_vegetation_data, lat, lon)
                    except Exception as api_e:
                        logger.warning(f"Failed to fetch env data for {file.filename}: {api_e}")

                # --- Build prediction dict ---
                prediction = {
                    "filename": file.filename,
                    "disease": disease_name,
                    "confidence": confidence,
                    "risk_level": self.derive_risk_level(confidence, env_data),
                    "soil": soil_info,
                    "satellite": satellite_info,
                    "weather": weather_info,
                    "low_confidence": confidence < self.confidence_threshold
                }

                # --- Add recommendations with confidence handling ---
                prediction["recommendation"] = self.enhance_recommendation(
                    disease=disease_name,
                    weather_data=weather_info,
                    soil_data=soil_info,
                    satellite_data=satellite_info,
                    confidence=confidence
                )

                results.append(prediction)

                # --- Save to SQLite ---
                if db_sqlite_conn:
                    try:
                        cursor = db_sqlite_conn.cursor()
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS predictions (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                filename TEXT,
                                disease TEXT,
                                confidence REAL,
                                risk_level TEXT
                            )
                        """)
                        cursor.execute("""
                            INSERT INTO predictions (filename, disease, confidence, risk_level)
                            VALUES (?, ?, ?, ?)
                        """, (
                            prediction["filename"],
                            prediction["disease"],
                            prediction["confidence"],
                            prediction["risk_level"]
                        ))
                        db_sqlite_conn.commit()
                    except Exception as e:
                        logger.error(f"SQLite insert error for {file.filename}: {e}")

                # --- Save to MongoDB ---
                if mongo_collection is not None:
                    try:
                        mongo_collection.insert_one(prediction)
                    except Exception as e:
                        logger.error(f"MongoDB insert error for {file.filename}: {e}")

            except Exception as e:
                logger.error(f"Prediction error for {file.filename}: {e}")

                # --- Attempt to derive risk_level even on failure ---
                try:
                    env_data = {"soil_health": 0.6, "humidity": 75}
                    if lat is not None and lon is not None and soil_service:
                        soil_info = await asyncio.to_thread(soil_service.get_soil_analysis, lat, lon)
                        env_data["soil_health"] = soil_info.get("health_score", 0.5)
                        if weather_service:
                            env_data["humidity"] = await asyncio.to_thread(weather_service.get_humidity, lat, lon)
                    risk_level = self.derive_risk_level(0.0, env_data)
                except Exception as inner_e:
                    logger.warning(f"Failed to derive risk_level for failed prediction: {inner_e}")
                    risk_level = "Unknown"

                failed_prediction = {
                    "filename": file.filename,
                    "disease": "Prediction failed",
                    "confidence": 0.0,
                    "risk_level": risk_level
                }
                results.append(failed_prediction)

                # --- Save failed prediction to SQLite ---
                if db_sqlite_conn:
                    try:
                        cursor = db_sqlite_conn.cursor()
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS predictions (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                filename TEXT,
                                disease TEXT,
                                confidence REAL,
                                risk_level TEXT
                            )
                        """)
                        cursor.execute("""
                            INSERT INTO predictions (filename, disease, confidence, risk_level)
                            VALUES (?, ?, ?, ?)
                        """, (
                            failed_prediction["filename"],
                            failed_prediction["disease"],
                            failed_prediction["confidence"],
                            failed_prediction["risk_level"]
                        ))
                        db_sqlite_conn.commit()
                    except Exception as e:
                        logger.error(f"SQLite insert error for failed prediction: {e}")

                # --- Save failed prediction to MongoDB ---
                if mongo_collection is not None:
                    try:
                        mongo_collection.insert_one(failed_prediction)
                    except Exception as e:
                        logger.error(f"MongoDB insert error for failed prediction: {e}")

        # --- Export to CSV (append mode, dynamic header handling) ---
        csv_columns = ["filename", "disease", "confidence", "risk_level"]
        try:
            with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                if not csv_exists:
                    writer.writeheader()
                for data in results:
                    try:
                        writer.writerow({k: data.get(k, "") for k in csv_columns})
                    except Exception as e:
                        logger.error(f"Failed to write CSV row for {data.get('filename', 'unknown')}: {e}")
            logger.info(f"Predictions exported to CSV: {output_csv_path}")
        except Exception as e:
            logger.error(f"Failed to export CSV {output_csv_path}: {e}")

        return results

    # =======================
    # Convenience wrapper for FastAPI endpoint
    # =======================
    async def predict_disease_endpoint(
        self,
        file: UploadFile = None,
        files: list[UploadFile] = None,
        lat: float = None,
        lon: float = None,
        lats: list[float] = None,
        lons: list[float] = None,
        soil_service=None,
        weather_service=None,
        db_sqlite_conn=None,
        mongo_collection=None,
        output_csv_path: str = "predictions.csv"
    ) -> list[dict]:
        """
        Convenience wrapper that lets you call the endpoint with either
        `file=` or `files=`. Calls predict_disease_unified internally.
        """
        if file and not files:
            return await self.predict_disease_unified(
                files=file,
                lats=[lat] if lat is not None else None,
                lons=[lon] if lon is not None else None,
                soil_service=soil_service,
                weather_service=weather_service,
                db_sqlite_conn=db_sqlite_conn,
                mongo_collection=mongo_collection,
                output_csv_path=output_csv_path
            )
        elif files:
            return await self.predict_disease_unified(
                files=files,
                lats=lats,
                lons=lons,
                soil_service=soil_service,
                weather_service=weather_service,
                db_sqlite_conn=db_sqlite_conn,
                mongo_collection=mongo_collection,
                output_csv_path=output_csv_path
            )
        else:
            raise ValueError("You must provide either `file` or `files` to predict.")

    def _safe_json_load(self, value: str, context: str = "") -> Any:
        """Attempt to parse JSON string; log errors and return empty dict on failure."""
        if not value:
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.warning(f"Malformed JSON in {context}: {e}. Returning empty dict.")
            # Attempt to fix single quotes
            try:
                fixed = value.replace("'", '"')
                return json.loads(fixed)
            except Exception:
                return {}

    def derive_risk_level(self, confidence: float, env: dict) -> str:
        """Assign risk level based on confidence + environmental factors"""
        try:
            risk_score = confidence
            if env.get("soil_health") and env["soil_health"] < 0.5:
                risk_score += 0.1
            if env.get("humidity") and env["humidity"] > 70:
                risk_score += 0.1
            if risk_score < 0.4:
                return "Low"
            elif risk_score < 0.7:
                return "Medium"
            else:
                return "High"
        except Exception:
            return "Medium"  # fallback


    # ------------------ Fully Model-driven, Multi-factor & Comprehensive Recommendation Function ------------------
    def enhance_recommendation(
            self,
            disease: str,
            weather_data: dict = None,
            soil_data: dict = None,
            satellite_data: dict = None,
            disease_recommendation_map: dict = None,
            confidence: float = 1.0
        ) -> str:

        # ⚠️ Low-confidence check
        if confidence < self.confidence_threshold:
            return f"⚠️ Low-confidence prediction for {disease} – treat recommendations cautiously."

        """
        Generate extremely comprehensive recommendations for a predicted disease.
        
        This function integrates:
        - Weather conditions: humidity, rainfall, temperature, wind
        - Soil chemistry & health: pH, NPK, organic matter, soil health score
        - Satellite vegetation indices: NDVI, health_status
        - Disease-specific guidance: PlantVillage-trained classification, dynamic mapping, or model-driven
        - Multi-level risk assessment and tailored intervention guidance
        """

        recommendations = [f"Treatment guidance for {disease}"]
        disease_lower = disease.lower()

        # =====================
        # 1. Weather-based adjustments
        # =====================
        if weather_data:
            humidity = weather_data.get("humidity", 0)
            rain = weather_data.get("rain_last_hour", 0)
            temp = weather_data.get("temperature", 0)
            wind_speed = weather_data.get("wind_speed", 0)
            leaf_wetness = weather_data.get("leaf_wetness", 0)

            # Humidity-based risk
            if humidity > 90:
                recommendations.append("Extremely high humidity - increase fungicide frequency and monitor leaf wetness.")
            elif humidity > 75:
                recommendations.append("High humidity - consider adjusting spray schedule to prevent fungal outbreaks.")
            elif humidity < 40:
                recommendations.append("Low humidity - monitor irrigation and reduce powdery mildew risk.")

            # Rainfall adjustment
            if rain > 15:
                recommendations.append("Heavy rainfall detected - delay chemical application, ensure leaves are dry before spraying.")
            elif rain > 5:
                recommendations.append("Moderate rainfall - ensure field drainage and adjust irrigation schedule.")
            elif rain == 0:
                recommendations.append("No recent rainfall - irrigation may be needed to maintain plant health.")

            # Temperature adjustment
            if temp > 35:
                recommendations.append("High temperature - schedule spraying early morning or late evening to avoid phytotoxicity.")
            elif temp < 12:
                recommendations.append("Low temperature - disease progression may be slowed, adjust treatment accordingly.")

            # Wind adjustment
            if wind_speed > 25:
                recommendations.append("High wind detected - take precautions to prevent spray drift.")

            # Leaf wetness
            if leaf_wetness > 0.8:
                recommendations.append("High leaf wetness - increased risk of fungal disease, adjust spraying and drainage.")

        # =====================
        # 2. Soil-based adjustments
        # =====================
        if soil_data:
            ph = soil_data.get("ph_level", 6.5)
            nitrogen = soil_data.get("nitrogen", 0)
            phosphorus = soil_data.get("phosphorus", 0)
            potassium = soil_data.get("potassium", 0)
            organic_matter = soil_data.get("organic_matter", 0.5)
            health_score = soil_data.get("health_score", 0.5)
            moisture = soil_data.get("moisture", 0.5)

            # pH
            if ph < 5.5:
                recommendations.append("Acidic soil detected - consider liming to optimize nutrient availability.")
            elif ph > 7.5:
                recommendations.append("Alkaline soil detected - consider acidifying amendments for optimal nutrient uptake.")

            # Macronutrients
            if nitrogen < 20:
                recommendations.append("Nitrogen deficient - apply suitable nitrogen fertilizer.")
            if phosphorus < 15:
                recommendations.append("Phosphorus deficient - apply phosphate fertilizer.")
            if potassium < 120:
                recommendations.append("Potassium deficient - apply potash fertilizer.")

            # Organic matter & health
            if organic_matter < 3.0:
                recommendations.append("Low organic matter - consider compost or green manure.")
            if health_score < 0.5:
                recommendations.append("Overall soil health low - integrate soil amendments and crop rotation.")

            # Soil moisture
            if moisture < 0.3:
                recommendations.append("Soil moisture low - irrigation recommended.")
            elif moisture > 0.8:
                recommendations.append("Soil overly wet - improve drainage to prevent root disease.")

        # =====================
        # 3. Satellite-based adjustments
        # =====================
        if satellite_data:
            ndvi = satellite_data.get("ndvi", 0)
            health_status = satellite_data.get("health_status", "").lower()
            if ndvi < 0.3 or health_status in ["poor", "stressed", "dry"]:
                recommendations.append("Satellite data indicates stressed vegetation - adjust irrigation, fertilization, and monitor pests.")

            if ndvi > 0.8 and health_status in ["healthy", "good"]:
                recommendations.append("Satellite NDVI indicates healthy vegetation - maintain current management practices.")

        # Define the disease_recommendation_map from the PlantVillage dataset
        disease_recommendation_map = {
            "apple___apple_scab": "Prune affected areas; apply fungicides.",
            "apple___black_rot": "Remove and destroy infected leaves and fruits; apply fungicides.",
            "apple___cedar_apple_rust": "Remove and destroy infected leaves; apply fungicides.",
            "apple___healthy": "No treatment necessary.",
            "blueberry___healthy": "No treatment necessary.",
            "cherry_(including_sour)___healthy": "No treatment necessary.",
            "cherry_(including_sour)___powdery_mildew": "Prune affected areas; apply fungicides.",
            "corn_(maize)___cercospora_leaf_spot gray_leaf_spot": "Remove and destroy infected leaves; apply fungicides.",
            "corn_(maize)___common_rust_": "Remove and destroy infected leaves; apply fungicides.",
            "corn_(maize)___northern_leaf_blight": "Remove and destroy infected leaves; apply fungicides.",
            "corn_(maize)___healthy": "No treatment necessary.",
            "grape___black_rot": "Remove and destroy infected leaves and fruits; apply fungicides.",
            "grape___esca_(black_measles)": "Prune affected areas; apply fungicides.",
            "grape___leaf_blight_(isariopsis_leaf_spot)": "Remove and destroy infected leaves; apply fungicides.",
            "grape___healthy": "No treatment necessary.",
            "orange___haunglongbing_(citrus_greening)": "Remove and destroy infected plants; apply systemic insecticides.",
            "peach___bacterial_spot": "Remove and destroy infected plants; apply copper-based bactericides.",
            "peach___healthy": "No treatment necessary.",
            "pepper,_bell___bacterial_spot": "Remove and destroy infected plants; apply copper-based bactericides.",
            "pepper,_bell___healthy": "No treatment necessary.",
            "potato___early_blight": "Remove and destroy infected leaves; apply fungicides.",
            "potato___late_blight": "Remove and destroy infected plants; apply fungicides.",
            "potato___healthy": "No treatment necessary.",
            "raspberry___healthy": "No treatment necessary.",
            "soybean___healthy": "No treatment necessary.",
            "squash___powdery_mildew": "Prune affected areas; apply fungicides.",
            "strawberry___leaf_scorch": "Remove and destroy infected leaves; apply fungicides.",
            "strawberry___healthy": "No treatment necessary.",
            "tomato___bacterial_spot": "Remove and destroy infected plants; apply copper-based bactericides.",
            "tomato___early_blight": "Remove and destroy infected leaves; apply fungicides.",
            "tomato___late_blight": "Remove and destroy infected plants; apply fungicides.",
            "tomato___leaf_mold": "Remove and destroy infected leaves; apply fungicides.",
            "tomato___septoria_leaf_spot": "Remove and destroy infected leaves; apply fungicides.",
            "tomato___spider_mites two-spotted_spider_mite": "Apply miticides.",
            "tomato___target_spot": "Remove and destroy infected leaves; apply fungicides.",
            "tomato___tomato_mosaic_virus": "Remove and destroy infected plants; control vectors with insecticides.",
            "tomato___tomato_yellow_leaf_curl_virus": "Remove and destroy infected plants; apply insecticides.",
            "tomato___healthy": "No treatment necessary.",
        }

        # =====================
        # 4. Disease-specific recommendations
        # =====================
        if disease_recommendation_map and disease_lower in disease_recommendation_map:
            recommendations.append(disease_recommendation_map[disease_lower])
        else:
            # Check for unknown/unavailable before fallback
            if disease_lower == "unknown" or disease_lower == "unavailable":
                recommendations.append("Prediction uncertain - re-upload clear leaf image for accurate diagnosis.")
            # PlantVillage-inspired rules (retained as fallback for non-mapped diseases)
            if "healthy" in disease_lower:
                recommendations.append("No intervention required, continue regular monitoring.")
            elif "late blight" in disease_lower:
                recommendations.append("Late Blight detected - apply fungicide, remove infected tissue, and monitor neighboring plants.")
            elif "early blight" in disease_lower:
                recommendations.append("Early Blight detected - apply recommended fungicide, maintain crop hygiene, rotate crops if possible.")
            elif "powdery mildew" in disease_lower:
                recommendations.append("Powdery mildew detected - use sulfur-based fungicides and avoid high humidity areas.")
            elif "rust" in disease_lower:
                recommendations.append("Rust detected - apply appropriate fungicide and remove infected leaves.")
            elif "mosaic" in disease_lower:
                recommendations.append("Mosaic virus detected - remove infected plants, control vector insects, and implement resistant cultivars.")
            elif "rot" in disease_lower or "mold" in disease_lower:
                recommendations.append("Rot or mold detected - improve drainage, remove infected tissue, and apply suitable treatment.")
            else:
                recommendations.append(f"General recommendation: monitor crop, adjust environment, and consult agronomist for {disease}.")

        # =====================
        # 5. Multi-factor risk & safety consolidation
        # =====================
        # Remove duplicates, prioritize severe warnings at the start
        severity_keywords = ["extremely", "heavy", "high", "low", "critical"]
        recommendations_sorted = sorted(
            list(dict.fromkeys(recommendations)),
            key=lambda x: any(k in x.lower() for k in severity_keywords),
            reverse=True
        )
        return " | ".join(recommendations_sorted)
    