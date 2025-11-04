# import logging
# from typing import Optional, List, Dict, Any, Union
# from datetime import datetime
# import sqlite3
# import os
# import json
# import csv
# from pymongo import MongoClient, errors
# from src.crop_monitor.config.settings import settings
# from src.crop_monitor.core.models import PredictionResult
# import asyncio
# import threading
# import joblib

# logger = logging.getLogger(__name__)

# class DatabaseService:
#     def __init__(self):
#         self.mongo_client = None
#         self.mongo_db = None
#         self.sqlite_conn = None
#         self.sqlite_lock = threading.Lock() 


#         # Load soil model first
#         self.load_soil_model(os.path.join(settings.MODEL_DIR, "best_rf.pkl"))

#         # Setup databases (MongoDB + SQLite)
#         self.setup_databases()

#         # ====== FIXED CSV path ======
#         self.csv_path = os.path.join(os.getcwd(), "data", "predictions.csv")
#         os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        

#         # Initialize CSV only if it does not exist
#         if not os.path.exists(self.csv_path):
#             with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
#                 writer = csv.DictWriter(f, fieldnames=[
#                     "image_filename", "disease", "confidence", "treatment",
#                     "soil_data", "satellite_data", "weather_data", "timestamp"
#                 ])
#                 writer.writeheader()
#             logger.info(f"✅ CSV file initialized at {self.csv_path}")
#         else:
#             logger.info(f"✅ CSV already exists at {self.csv_path}")

#         # One-time cleanup on startup to fix any malformed JSON
#         self._cleanup_malformed_json()


#     def load_soil_model(self, model_path):
#         if os.path.exists(model_path):
#             self.soil_model = joblib.load(model_path)
#             logger.info(f"✅ Soil model loaded from {model_path}")
#         else:
#             self.soil_model = None
#             logger.warning(f"Soil model not found at {model_path}. Fallbacks will be used.")


#     def _safe_load_json(self, data: str) -> dict:
#         """Safely load a JSON string into a dictionary. Returns empty dict if parsing fails."""
#         try:
#             if not data:
#                 return {}
#             if isinstance(data, dict):
#                 return data
#             return json.loads(data)
#         except Exception as e:
#             logger.warning(f"Failed to parse JSON: {e}")
#             return {}
        

#     def _predict_soil_type(self, input_features: Dict[str, Any]) -> str:
#         try:
#             if hasattr(self, "soil_model") and self.soil_model:
#                 # Dynamically get all model-required features
#                 model_input = [input_features.get(f, 0) for f in self.soil_model.feature_names_in_]
#                 pred = self.soil_model.predict([model_input])
#                 return str(pred[0])
#             else:
#                 return "unknown"
#         except Exception as e:
#             logger.warning(f"Failed to predict soil type: {e}")
#             return "unknown"


#     def setup_databases(self):
#         """Initialize both MongoDB and SQLite connections"""
#         try:
#             mongodb_url = settings.MONGODB_ATLAS_URI
#             if mongodb_url and mongodb_url != "mongodb+srv://username:password@cluster.mongodb.net/":
#                 try:
#                     self.mongo_client = MongoClient(mongodb_url, serverSelectionTimeoutMS=5000)
#                     self.mongo_client.admin.command('ping')

#                     # ✅ Ensure mongo_db is Database object
#                     self.mongo_db = self.mongo_client.get_database("crop_monitor")

#                     # ✅ Ensure `predictions` is a Collection object
#                     self.predictions_collection = self.mongo_db.get_collection("predictions")

#                     logger.info("✅ MongoDB Atlas connection established")
#                     self._ensure_mongo_indexes()
#                 except errors.ServerSelectionTimeoutError:
#                     logger.warning("MongoDB Atlas connection timeout - proceeding in offline mode")
#                     self.mongo_db = None
#                     self.predictions_collection = None
#                 except Exception as e:
#                     logger.warning(f"MongoDB Atlas connection failed: {e} - proceeding in offline mode")
#                     self.mongo_db = None
#                     self.predictions_collection = None
#             else:
#                 logger.warning("MongoDB Atlas URI not configured - proceeding in offline mode")
#                 self.mongo_db = None
#                 self.predictions_collection = None

#             # SQLite setup
#             sqlite_path = settings.SQLITE_DB_PATH
#             os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
#             self.sqlite_conn = sqlite3.connect(sqlite_path, check_same_thread=False)
#             self.sqlite_conn.row_factory = sqlite3.Row
#             self._init_sqlite_schema()
#             logger.info(f"✅ SQLite database initialized at {sqlite_path}")
#         except Exception as e:
#             logger.error(f"Error setting up databases: {e}")


#     def _ensure_mongo_indexes(self):
#         if self.mongo_db is not None:
#             try:
#                 self.mongo_db.predictions.create_index([("timestamp", -1)])
#                 self.mongo_db.predictions.create_index([("image_filename", 1)])
#                 logger.info("✅ MongoDB indexes ensured")
#             except Exception as e:
#                 logger.error(f"Error creating MongoDB indexes: {e}")

#     def _init_sqlite_schema(self):
#         cursor = self.sqlite_conn.cursor()
#         cursor.executescript("""
#         CREATE TABLE IF NOT EXISTS predictions (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             image_filename TEXT,
#             disease TEXT,
#             confidence REAL,
#             treatment TEXT,
#             soil_data TEXT,
#             satellite_data TEXT,
#             weather_data TEXT,
#             timestamp DATETIME,
#             synced INTEGER DEFAULT 0,
#             created_at DATETIME DEFAULT CURRENT_TIMESTAMP
#         );
        
#         CREATE TABLE IF NOT EXISTS sync_queue (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             collection_name TEXT,
#             operation TEXT,
#             data TEXT,
#             timestamp DATETIME,
#             attempts INTEGER DEFAULT 0,
#             last_attempt DATETIME,
#             created_at DATETIME DEFAULT CURRENT_TIMESTAMP
#         );
#         """)
#         self.sqlite_conn.commit()

#     def _init_csv(self):
#         """Initialize CSV with headers if not exists."""
#         if not os.path.exists(self.csv_path):
#             with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
#                 writer = csv.DictWriter(f, fieldnames=[
#                     "image_filename", "disease", "confidence", "treatment",
#                     "soil_data", "satellite_data", "weather_data", "timestamp"
#                 ])
#                 writer.writeheader()
#             logger.info(f"✅ CSV file initialized at {self.csv_path}")

#     def save_prediction(self, prediction: Union[PredictionResult, Dict[str, Any]]):
#         """
#         Save a prediction to SQLite, MongoDB, and CSV.
#         Accepts either PredictionResult Pydantic model or dict.
#         Handles soil warnings/risk levels properly.
#         """
#         try:
#             # Convert Pydantic model to dict if needed
#             if isinstance(prediction, PredictionResult):
#                 prediction_data = prediction.dict()
#             else:
#                 prediction_data = prediction

#             # Predict soil_type dynamically if not already provided
#             if "predicted_soil_type" not in prediction_data:
#                 prediction_data["predicted_soil_type"] = self._predict_soil_type(prediction_data)

#             # Ensure JSON fields are dicts
#             for key in ["soil_data", "satellite_data", "weather_data"]:
#                 val = prediction_data.get(key)
#                 if isinstance(val, str):
#                     try:
#                         prediction_data[key] = json.loads(val)
#                     except Exception:
#                         prediction_data[key] = {}
#                 elif val is None:
#                     prediction_data[key] = {}

#             # --- Enforce soil object shape and inject dynamic soil_type prediction ---
#             soil_data = prediction_data.get("soil_data", {})
#             if not isinstance(soil_data, dict):
#                 soil_data = {}

#             # Use dynamic predicted soil_type if provided by ML model
#             if "predicted_soil_type" in prediction_data:
#                 soil_data["soil_type"] = prediction_data["predicted_soil_type"]

#             # Ultimate fix: preserve model prediction as source
#             soil_data = {
#                 "soil_type": soil_data.get("soil_type", "unknown"),
#                 "risk_level": soil_data.get("risk_level", "Medium" if not soil_data else "Low"),
#                 "source": soil_data.get("source", "model_prediction" if "predicted_soil_type" in prediction_data else "fallback"),
#                 **{k: v for k, v in soil_data.items() if k not in ["soil_type", "risk_level", "source"]}
#             }

#             prediction_data["soil_data"] = soil_data

#             # --- Risk/warning logging ---
#             if "warning" in soil_data:
#                 logger.warning(f"⚠️ Soil warning: {soil_data.get('warning')} (risk={soil_data.get('risk_level', 'Unknown')})")

#             # --- Save to SQLite ---
#             try:
#                 with self.sqlite_lock:  # ensures thread-safe access
#                     cursor = self.sqlite_conn.cursor()
#                     timestamp = prediction_data.get("timestamp") or datetime.utcnow().isoformat()

#                     # Ensure JSON fields are valid strings
#                     soil_json = json.dumps(prediction_data.get("soil_data", {}))
#                     satellite_json = json.dumps(prediction_data.get("satellite_data", {}))
#                     weather_json = json.dumps(prediction_data.get("weather_data", {}))

#                     cursor.execute("""
#                     INSERT INTO predictions 
#                     (image_filename, disease, confidence, treatment, soil_data, satellite_data, weather_data, timestamp, synced)
#                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
#                     """, (
#                         prediction_data.get("image_filename") or "unknown",
#                         prediction_data.get("disease") or "unknown",
#                         float(prediction_data.get("confidence") or 0.0),
#                         prediction_data.get("treatment") or "unknown",
#                         soil_json,
#                         satellite_json,
#                         weather_json,
#                         timestamp,
#                         0  # Always start as unsynced
#                     ))
#                     prediction_id = cursor.lastrowid
#                     self.sqlite_conn.commit()
#                     logger.info(f"✅ Prediction saved to SQLite (ID: {prediction_id})")
#             except Exception as e:
#                 logger.error(f"❌ Failed to save prediction to SQLite: {e}")
#                 raise

#             # --- Save to MongoDB ---
#             if self.predictions_collection is not None:
#                 try:
#                     result = self.predictions_collection.insert_one(prediction_data)
#                     logger.info(f"✅ Prediction saved to MongoDB with ID: {result.inserted_id}")

#                     # Update SQLite synced flag ONLY if MongoDB insert succeeded
#                     with self.sqlite_lock:
#                         cursor = self.sqlite_conn.cursor()
#                         cursor.execute("UPDATE predictions SET synced = 1 WHERE id = ?", (prediction_id,))
#                         self.sqlite_conn.commit()
#                 except Exception as e:
#                     logger.warning(f"❌ MongoDB save failed: {e}. Adding to sync queue.")
#                     self._add_to_sync_queue("predictions", "insert", prediction_data)

#             # --- Save to CSV ---
#             try:
#                 with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
#                     writer = csv.DictWriter(f, fieldnames=[
#                         "image_filename", "disease", "confidence", "treatment",
#                         "soil_data", "satellite_data", "weather_data", "timestamp"
#                     ])
#                     writer.writerow({
#                         "image_filename": prediction_data.get("image_filename"),
#                         "disease": prediction_data.get("disease"),
#                         "confidence": prediction_data.get("confidence"),
#                         "treatment": prediction_data.get("treatment"),
#                         "soil_data": json.dumps(prediction_data.get("soil_data", {})),
#                         "satellite_data": json.dumps(prediction_data.get("satellite_data", {})),
#                         "weather_data": json.dumps(prediction_data.get("weather_data", {})),
#                         "timestamp": prediction_data.get("timestamp") or datetime.now().isoformat()
#                     })
#                 logger.info(f"✅ Prediction logged to CSV: {self.csv_path}")
#             except Exception as e:
#                 logger.error(f"❌ Failed to write prediction to CSV: {e}")

#             return prediction_id

#         except Exception as e:
#             logger.error(f"❌ Error saving prediction: {e}")
#             raise


#     def _cleanup_malformed_json(self):
#         try:
#             cursor = self.sqlite_conn.cursor()
#             cursor.execute("SELECT id, soil_data, satellite_data, weather_data FROM predictions")
#             for row in cursor.fetchall():
#                 pred_id = row["id"]
#                 soil_fixed = self._safe_load_json(row["soil_data"])
#                 satellite_fixed = self._safe_load_json(row["satellite_data"])
#                 weather_fixed = self._safe_load_json(row["weather_data"])

#                 if json.dumps(soil_fixed) != (row["soil_data"] or "{}"):
#                     logger.info(f"Fixed malformed soil_data for prediction ID {pred_id}")
#                 if json.dumps(satellite_fixed) != (row["satellite_data"] or "{}"):
#                     logger.info(f"Fixed malformed satellite_data for prediction ID {pred_id}")
#                 if json.dumps(weather_fixed) != (row["weather_data"] or "{}"):
#                     logger.info(f"Fixed malformed weather_data for prediction ID {pred_id}")

#                 cursor.execute("""
#                     UPDATE predictions
#                     SET soil_data = ?, satellite_data = ?, weather_data = ?
#                     WHERE id = ?
#                 """, (json.dumps(soil_fixed), json.dumps(satellite_fixed), json.dumps(weather_fixed), pred_id))

#             cursor.execute("SELECT id, data FROM sync_queue")
#             for row in cursor.fetchall():
#                 op_id = row["id"]
#                 data_fixed = self._safe_load_json(row["data"])
#                 if json.dumps(data_fixed) != (row["data"] or "{}"):
#                     logger.info(f"Fixed malformed JSON in sync_queue ID {op_id}")
#                 cursor.execute("UPDATE sync_queue SET data = ? WHERE id = ?", (json.dumps(data_fixed), op_id))

#             self.sqlite_conn.commit()
#             logger.info("✅ Malformed JSON cleanup completed with fixes logged")
#         except Exception as e:
#             logger.error(f"Error during JSON cleanup: {e}")


#     def _add_to_sync_queue(self, collection_name: str, operation: str, data: dict):
#         try:
#             cursor = self.sqlite_conn.cursor()
#             cursor.execute("""
#                 INSERT INTO sync_queue (collection_name, operation, data, timestamp)
#                 VALUES (?, ?, ?, ?)
#             """, (collection_name, operation, json.dumps(data), datetime.utcnow().isoformat()))
#             self.sqlite_conn.commit()
#             logger.info(f"✅ Added operation to sync_queue: {operation} on {collection_name}")
#         except Exception as e:
#             logger.error(f"❌ Failed to add operation to sync_queue: {e}")


#     # # --- Full improved sync_offline_data method ---
#     # def sync_offline_data(self):
#     #     """Synchronize unsynced SQLite predictions and queued operations to MongoDB."""
        
#     #     if self.predictions_collection is None:
#     #         logger.warning("MongoDB not available, skipping sync")
#     #         return False

#     #     synced_count = 0
#     #     ops_count = 0

#     #     try:
#     #         cursor = self.sqlite_conn.cursor()

#     #         # --- Sync unsynced predictions ---
#     #         cursor.execute("SELECT * FROM predictions WHERE synced = 0")
#     #         unsynced_predictions = cursor.fetchall()
#     #         for prediction in unsynced_predictions:
#     #             try:
#     #                 prediction_data = {
#     #                     "image_filename": prediction["image_filename"],
#     #                     "disease": prediction["disease"],
#     #                     "confidence": prediction["confidence"],
#     #                     "treatment": prediction["treatment"],
#     #                     "soil_data": self._safe_load_json(prediction["soil_data"]),
#     #                     "satellite_data": self._safe_load_json(prediction["satellite_data"]),
#     #                     "weather_data": self._safe_load_json(prediction["weather_data"]),
#     #                     "timestamp": prediction["timestamp"]
#     #                 }
#     #                 # ✅ Use the Collection object safely
#     #                 self.predictions_collection.insert_one(prediction_data)
#     #                 cursor.execute("UPDATE predictions SET synced = 1 WHERE id = ?", (prediction["id"],))
#     #                 synced_count += 1
#     #             except Exception as e:
#     #                 logger.error(f"Failed to sync prediction {prediction['id']}: {e}")

#     #         # --- Sync queued operations ---
#     #         cursor.execute("SELECT * FROM sync_queue WHERE attempts < 5")
#     #         queued_operations = cursor.fetchall()
#     #         for operation in queued_operations:
#     #             try:
#     #                 data = self._safe_load_json(operation["data"])
                    
#     #                 # Ensure collection exists
#     #                 collection_name = operation["collection_name"]
#     #                 if not hasattr(self.mongo_db, collection_name):
#     #                     logger.warning(f"Collection '{collection_name}' not found in MongoDB. Skipping operation.")
#     #                     continue
#     #                 collection = self.mongo_db.get_collection(collection_name)
                    
#     #                 if operation["operation"] == "insert":
#     #                     collection.insert_one(data)
                    
#     #                 cursor.execute("DELETE FROM sync_queue WHERE id = ?", (operation["id"],))
#     #                 ops_count += 1
#     #             except Exception as e:
#     #                 logger.error(f"Failed to sync queued operation {operation['id']}: {e}")
#     #                 cursor.execute(
#     #                     "UPDATE sync_queue SET attempts = attempts + 1, last_attempt = CURRENT_TIMESTAMP WHERE id = ?",
#     #                     (operation["id"],)
#     #                 )

#     #         self.sqlite_conn.commit()
#     #         logger.info(f"✅ Synced {synced_count} predictions and {ops_count} operations")
#     #         return True

#     #     except Exception as e:
#     #         logger.error(f"Error during sync: {e}")
#             # return False


#     def sync_offline_data(self):
#         """Synchronize unsynced SQLite predictions and queued operations to MongoDB."""

#         if self.predictions_collection is None:
#             logger.warning("MongoDB not available, skipping sync")
#             return False

#         synced_count = 0
#         ops_count = 0

#         try:
#             cursor = self.sqlite_conn.cursor()

#             # --- Sync unsynced predictions ---
#             cursor.execute("SELECT * FROM predictions WHERE synced = 0")
#             unsynced_predictions = cursor.fetchall()
#             for prediction in unsynced_predictions:
#                 try:
#                     prediction_data = {
#                         "image_filename": prediction["image_filename"],
#                         "disease": prediction["disease"],
#                         "confidence": prediction["confidence"],
#                         "treatment": prediction["treatment"],
#                         "soil_data": self._safe_load_json(prediction["soil_data"]),
#                         "satellite_data": self._safe_load_json(prediction["satellite_data"]),
#                         "weather_data": self._safe_load_json(prediction["weather_data"]),
#                         "timestamp": prediction["timestamp"]
#                     }

#                     # --- Insert into MongoDB ---
#                     self.predictions_collection.insert_one(prediction_data)

#                     # --- Mark as synced in SQLite ---
#                     cursor.execute("UPDATE predictions SET synced = 1 WHERE id = ?", (prediction["id"],))
#                     synced_count += 1
#                 except Exception as e:
#                     logger.error(f"Failed to sync prediction ID {prediction['id']}: {e}")

#             # --- Sync queued operations ---
#             cursor.execute("SELECT * FROM sync_queue WHERE attempts < 5")
#             queued_operations = cursor.fetchall()
#             for operation in queued_operations:
#                 try:
#                     data = self._safe_load_json(operation["data"])
#                     collection_name = operation["collection_name"]

#                     # ✅ Always get collection dynamically from MongoDB
#                     collection = self.mongo_db.get_collection(collection_name)

#                     if operation["operation"] == "insert":
#                         collection.insert_one(data)

#                     # Delete from queue after successful operation
#                     cursor.execute("DELETE FROM sync_queue WHERE id = ?", (operation["id"],))
#                     ops_count += 1
#                 except Exception as e:
#                     logger.error(f"Failed to sync queued operation ID {operation['id']}: {e}")
#                     cursor.execute(
#                         "UPDATE sync_queue SET attempts = attempts + 1, last_attempt = CURRENT_TIMESTAMP WHERE id = ?",
#                         (operation["id"],)
#                     )

#             self.sqlite_conn.commit()
#             logger.info(f"✅ Synced {synced_count} predictions and {ops_count} operations")
#             return True

#         except Exception as e:
#             logger.error(f"Error during sync: {e}")
#             return False




#     def get_recent_predictions(self, limit: int = 10):
#         try:
#             cursor = self.sqlite_conn.cursor()
#             cursor.execute("""
#             SELECT * FROM predictions 
#             ORDER BY timestamp DESC 
#             LIMIT ?
#             """, (limit,))
#             predictions = []
#             for row in cursor.fetchall():
#                 predictions.append({
#                     "id": row["id"],
#                     "image_filename": row["image_filename"],
#                     "disease": row["disease"],
#                     "confidence": row["confidence"],
#                     "treatment": row["treatment"],
#                     "soil_data": self._safe_load_json(row["soil_data"]),
#                     "satellite_data": self._safe_load_json(row["satellite_data"]),
#                     "weather_data": self._safe_load_json(row["weather_data"]),
#                     "timestamp": row["timestamp"],
#                     "synced": bool(row["synced"])
#                 })
#             return predictions
#         except Exception as e:
#             logger.error(f"Error fetching recent predictions: {e}")
#             return []

#     def close(self):
#         try:
#             if self.mongo_client:
#                 self.mongo_client.close()
#             if self.sqlite_conn:
#                 self.sqlite_conn.close()
#             logger.info("Database connections closed")
#         except Exception as e:
#             logger.error(f"Error closing database connections: {e}")


# # =======================
# # Async Test for Offline Sync
# # =======================
# async def test_offline_sync():
#     db_service = DatabaseService()

#     # --- Simulate MongoDB being down ---
#     db_service.mongo_db = None
#     print("Simulating offline mode: MongoDB is unavailable.")

#     # --- Create dummy predictions ---
#     predictions = [
#         {
#             "image_filename": f"test_image_{i}.jpg",
#             "disease": "Late Blight" if i % 2 == 0 else "Healthy",
#             "confidence": 0.85 + i * 0.01,
#             "treatment": "Fungicide" if i % 2 == 0 else "None",
#             "soil_data": {"ph_level": 6.0 + i*0.1},
#             "satellite_data": {"ndvi": 0.2 + i*0.05},
#             "weather_data": {"humidity": 75 + i},
#             "timestamp": datetime.now().isoformat()
#         }
#         for i in range(3)
#     ]

#     # --- Save predictions (will fail MongoDB, queue them) ---
#     for pred in predictions:
#         db_service.save_prediction(pred)

#     # Check SQLite and CSV logs
#     recent = db_service.get_recent_predictions(limit=5)
#     print("Recent predictions saved in SQLite (offline):")
#     for r in recent:
#         print(r)

#     # --- Simulate MongoDB coming back online ---
#     from pymongo import MongoClient
#     try:
#         mongo_client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
#         mongo_client.admin.command("ping")
#         db_service.mongo_db = mongo_client["crop_monitor"]
#         print("\nMongoDB back online! Syncing offline data...")
#     except Exception:
#         print("\nMongoDB still unavailable. Test will only check queue behavior.")

#     # --- Sync offline data ---
#     synced = db_service.sync_offline_data()
#     print(f"\nSync completed: {synced}")

#     # --- Verify sync_queue is empty if MongoDB is online ---
#     cursor = db_service.sqlite_conn.cursor()
#     cursor.execute("SELECT COUNT(*) as cnt FROM sync_queue")
#     queue_count = cursor.fetchone()["cnt"]
#     print(f"Remaining operations in sync_queue: {queue_count}")

#     db_service.close()


# # Run the test (for standalone execution)
# if __name__ == "__main__":
#     asyncio.run(test_offline_sync())

























import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
import sqlite3
import os
import json
import csv
from pymongo import MongoClient, errors
import asyncio
import threading
import joblib


from src.crop_monitor.config.settings import settings
from src.crop_monitor.core.models import PredictionResult


logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        self.mongo_client = None
        self.mongo_db = None
        self.sqlite_conn = None
        self.sqlite_lock = threading.Lock() 
        self.soil_model = None  # Initialize as None

        # Ensure base predictions directory exists
        os.makedirs(settings.BASE_PREDICTIONS_DIR, exist_ok=True)
        
        # Load soil models from settings paths
        self._load_soil_models()

        # Setup databases (MongoDB + SQLite)
        self.setup_databases()

        # Initialize CSV
        self.csv_path = settings.CSV_FILE_PATH
        self._init_csv()
        
        # One-time cleanup on startup to fix any malformed JSON
        self._cleanup_malformed_json()

    def _load_soil_models(self):
        """Load soil models using paths from settings - Production version without fallback"""
        try:
            # Load the main soil model using correct attribute name
            if hasattr(settings, 'SOIL_MODEL_PATH') and settings.SOIL_MODEL_PATH:
                if os.path.exists(settings.SOIL_MODEL_PATH):
                    self.soil_model = joblib.load(settings.SOIL_MODEL_PATH)
                    logger.info(f"✅ Soil model loaded from {settings.SOIL_MODEL_PATH}")
                else:
                    logger.warning(f"⚠️ Soil model file not found: {settings.SOIL_MODEL_PATH}")
                    self.soil_model = None
            else:
                logger.warning("⚠️ Soil model path not configured in settings")
                self.soil_model = None
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load soil model: {e}")
            self.soil_model = None

    def _safe_load_json(self, data: str) -> dict:
        """Safely load a JSON string into a dictionary. Returns empty dict if parsing fails."""
        try:
            if not data:
                return {}
            if isinstance(data, dict):
                return data
            return json.loads(data)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return {}
        
    def _predict_soil_type(self, input_features: Dict[str, Any]) -> str:
        """Predict soil type using the loaded model - Production version"""
        try:
            # Use the trained model if available
            if self.soil_model is not None and hasattr(self.soil_model, 'predict'):
                # Get feature names the model expects
                if hasattr(self.soil_model, 'feature_names_in_'):
                    model_features = self.soil_model.feature_names_in_
                else:
                    # If no feature names, use common soil features
                    model_features = ['clay', 'silt', 'sand', 'ph_level', 'organic_matter']
                
                # Prepare input for model
                model_input = []
                for feature in model_features:
                    model_input.append(input_features.get(feature, 0.0))
                
                # Make prediction
                prediction = self.soil_model.predict([model_input])[0]
                
                # Map numeric prediction to soil types
                soil_types = ["sandy", "loam", "clay", "silty", "peaty", "chalky"]
                if isinstance(prediction, (int, float)):
                    prediction_idx = int(prediction)
                    if 0 <= prediction_idx < len(soil_types):
                        return soil_types[prediction_idx]
                    else:
                        # For regression output, use thresholds
                        if prediction < 1.5:
                            return "sandy"
                        elif prediction < 2.5:
                            return "loam"
                        elif prediction < 3.5:
                            return "clay"
                        else:
                            return "silty"
                return "loam"  # Default fallback
            else:
                logger.warning("Soil model not available for prediction, using basic classification")
                # Basic soil classification based on texture
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
            logger.error(f"Failed to predict soil type: {e}")
            return "loam"

    def setup_databases(self):
        """Initialize both MongoDB and SQLite connections"""
        try:
            # Ensure SQLite directory exists
            os.makedirs(os.path.dirname(settings.SQLITE_DB_PATH), exist_ok=True)
            
            mongodb_url = settings.MONGODB_ATLAS_URI
            if mongodb_url and mongodb_url != "mongodb+srv://username:password@cluster.mongodb.net/":
                try:
                    # Connect to MongoDB
                    self.mongo_client = MongoClient(mongodb_url, serverSelectionTimeoutMS=5000)
                    self.mongo_client.admin.command('ping')

                    # ✅ Use database name from settings
                    db_name = settings.MONGODB_ATLAS_DB_NAME
                    self.mongo_db = self.mongo_client.get_database(db_name)
                    
                    # Debug log
                    logger.info(f"✅ Connected to MongoDB database: {self.mongo_db.name}")

                    # ✅ Ensure `predictions` is a Collection object
                    self.predictions_collection = self.mongo_db.get_collection(settings.COLLECTION_NAME)

                    logger.info("✅ MongoDB Atlas connection established")
                    self._ensure_mongo_indexes()
                except errors.ServerSelectionTimeoutError:
                    logger.warning("MongoDB Atlas connection timeout - proceeding in offline mode")
                    self.mongo_db = None
                    self.predictions_collection = None
                except Exception as e:
                    logger.warning(f"MongoDB Atlas connection failed: {e} - proceeding in offline mode")
                    self.mongo_db = None
                    self.predictions_collection = None
            else:
                logger.warning("MongoDB Atlas URI not configured - proceeding in offline mode")
                self.mongo_db = None
                self.predictions_collection = None

            # SQLite setup - thread-safe with check_same_thread=False
            self.sqlite_conn = sqlite3.connect(settings.SQLITE_DB_PATH, check_same_thread=False)
            self.sqlite_conn.row_factory = sqlite3.Row
            self._init_sqlite_schema()
            logger.info(f"✅ SQLite database initialized at {settings.SQLITE_DB_PATH}")
        except Exception as e:
            logger.error(f"Error setting up databases: {e}")

    def _ensure_mongo_indexes(self):
        if self.mongo_db is not None:
            try:
                self.mongo_db.predictions.create_index([("timestamp", -1)])
                self.mongo_db.predictions.create_index([("image_filename", 1)])
                logger.info("✅ MongoDB indexes ensured")
            except Exception as e:
                logger.error(f"Error creating MongoDB indexes: {e}")

    def _init_sqlite_schema(self):
        cursor = self.sqlite_conn.cursor()
        cursor.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT,
            disease TEXT,
            confidence REAL,
            treatment TEXT,
            soil_data TEXT,
            satellite_data TEXT,
            weather_data TEXT,
            timestamp DATETIME,
            synced INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS sync_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_name TEXT,
            operation TEXT,
            data TEXT,
            timestamp DATETIME,
            attempts INTEGER DEFAULT 0,
            last_attempt DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)
        self.sqlite_conn.commit()

    def _init_csv(self):
        """Initialize CSV with headers if not exists."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "image_filename", "disease", "confidence", "treatment",
                    "soil_data", "satellite_data", "weather_data", "timestamp"
                ])
                writer.writeheader()
            logger.info(f"✅ CSV file initialized at {self.csv_path}")

    def save_prediction(self, prediction: Union[PredictionResult, Dict[str, Any]]):
        """
        Save a prediction to SQLite, CSV, and MongoDB.
        Accepts either PredictionResult Pydantic model or dict.
        """
        try:
            # Convert Pydantic model to dict if needed
            if isinstance(prediction, PredictionResult):
                prediction_data = prediction.dict()
            else:
                prediction_data = prediction

            # Predict soil_type dynamically if not already provided
            if "predicted_soil_type" not in prediction_data:
                prediction_data["predicted_soil_type"] = self._predict_soil_type(prediction_data)

            # Ensure JSON fields are dicts
            for key in ["soil_data", "satellite_data", "weather_data"]:
                val = prediction_data.get(key)
                if isinstance(val, str):
                    try:
                        prediction_data[key] = json.loads(val)
                    except Exception:
                        prediction_data[key] = {}
                elif val is None:
                    prediction_data[key] = {}

            # Enforce soil object shape and inject dynamic soil_type prediction
            soil_data = prediction_data.get("soil_data", {})
            if not isinstance(soil_data, dict):
                soil_data = {}

            # Use dynamic predicted soil_type if provided by ML model
            if "predicted_soil_type" in prediction_data:
                soil_data["soil_type"] = prediction_data["predicted_soil_type"]

            # Preserve model prediction as source
            soil_data = {
                "soil_type": soil_data.get("soil_type", "unknown"),
                "risk_level": soil_data.get("risk_level", "Medium" if not soil_data else "Low"),
                "source": soil_data.get("source", "model_prediction" if "predicted_soil_type" in prediction_data else "fallback"),
                **{k: v for k, v in soil_data.items() if k not in ["soil_type", "risk_level", "source"]}
            }

            prediction_data["soil_data"] = soil_data

            # Risk/warning logging
            if "warning" in soil_data:
                logger.warning(f"⚠️ Soil warning: {soil_data.get('warning')} (risk={soil_data.get('risk_level', 'Unknown')})")

            # Save to SQLite - START WITH synced=0 (NOT SYNCED)
            try:
                with self.sqlite_lock:  # ensures thread-safe access
                    cursor = self.sqlite_conn.cursor()
                    timestamp = prediction_data.get("timestamp") or datetime.utcnow().isoformat()

                    # Ensure JSON fields are valid strings
                    soil_json = json.dumps(prediction_data.get("soil_data", {}))
                    satellite_json = json.dumps(prediction_data.get("satellite_data", {}))
                    weather_json = json.dumps(prediction_data.get("weather_data", {}))

                    cursor.execute("""
                    INSERT INTO predictions 
                    (image_filename, disease, confidence, treatment, soil_data, satellite_data, weather_data, timestamp, synced)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        prediction_data.get("image_filename") or "unknown",
                        prediction_data.get("disease") or "unknown",
                        float(prediction_data.get("confidence") or 0.0),
                        prediction_data.get("treatment") or "unknown",
                        soil_json,
                        satellite_json,
                        weather_json,
                        timestamp,
                        0  # Always start as unsynced - FIXED
                    ))
                    prediction_id = cursor.lastrowid
                    self.sqlite_conn.commit()
                    logger.info(f"✅ Prediction saved to SQLite (ID: {prediction_id})")
            except Exception as e:
                logger.error(f"❌ Failed to save prediction to SQLite: {e}")
                raise

            # Save to CSV
            try:
                with self.sqlite_lock:  # Thread-safe CSV access
                    with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=[
                            "image_filename", "disease", "confidence", "treatment",
                            "soil_data", "satellite_data", "weather_data", "timestamp"
                        ])
                        writer.writerow({
                            "image_filename": prediction_data.get("image_filename"),
                            "disease": prediction_data.get("disease"),
                            "confidence": prediction_data.get("confidence"),
                            "treatment": prediction_data.get("treatment"),
                            "soil_data": json.dumps(prediction_data.get("soil_data", {})),
                            "satellite_data": json.dumps(prediction_data.get("satellite_data", {})),
                            "weather_data": json.dumps(prediction_data.get("weather_data", {})),
                            "timestamp": prediction_data.get("timestamp") or datetime.now().isoformat()
                        })
                    logger.info(f"✅ Prediction logged to CSV: {self.csv_path}")
            except Exception as e:
                logger.error(f"❌ Failed to write prediction to CSV: {e}")

            # --- Automatic sync to MongoDB immediately after saving ---
            if self.predictions_collection is not None:
                try:
                    # Prepare MongoDB-ready document
                    mongo_data = prediction_data.copy()
                    result = self.predictions_collection.insert_one(mongo_data)
                    logger.info(f"✅ Auto-synced prediction to MongoDB with ID: {result.inserted_id}")

                    # Update SQLite synced flag ONLY AFTER SUCCESSFUL SYNC - FIXED
                    with self.sqlite_lock:
                        cursor = self.sqlite_conn.cursor()
                        cursor.execute("UPDATE predictions SET synced = 1 WHERE id = ?", (prediction_id,))
                        self.sqlite_conn.commit()
                        logger.info(f"✅ Marked prediction {prediction_id} as synced")
                except Exception as e:
                    logger.warning(f"❌ Auto-sync failed: {e}. Prediction remains unsynced.")
                    # Don't mark as synced - let periodic sync handle it
                    # self._add_to_sync_queue("predictions", "insert", prediction_data)

            return prediction_id

        except Exception as e:
            logger.error(f"❌ Error saving prediction: {e}")
            raise

    def store_dsm_map(self, map_type: str, map_data: dict, bbox: list):
        """
        Store DSM-generated soil maps in the database
        """
        try:
            if self.predictions_collection:
                self.predictions_collection.insert_one({
                    "type": "dsm_map",
                    "map_type": map_type,  # "soc", "clay", "soil_type"
                    "bbox": bbox,
                    "data": map_data,
                    "timestamp": datetime.utcnow(),
                    "resolution": "30m",  # DSM pipeline resolution
                    "source": "dsm_pipeline"
                })
        except Exception as e:
            logger.error(f"Failed to store DSM map: {e}")

    def _cleanup_malformed_json(self):
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT id, soil_data, satellite_data, weather_data FROM predictions")
            for row in cursor.fetchall():
                pred_id = row["id"]
                soil_fixed = self._safe_load_json(row["soil_data"])
                satellite_fixed = self._safe_load_json(row["satellite_data"])
                weather_fixed = self._safe_load_json(row["weather_data"])

                if json.dumps(soil_fixed) != (row["soil_data"] or "{}"):
                    logger.info(f"Fixed malformed soil_data for prediction ID {pred_id}")
                if json.dumps(satellite_fixed) != (row["satellite_data"] or "{}"):
                    logger.info(f"Fixed malformed satellite_data for prediction ID {pred_id}")
                if json.dumps(weather_fixed) != (row["weather_data"] or "{}"):
                    logger.info(f"Fixed malformed weather_data for prediction ID {pred_id}")

                cursor.execute("""
                    UPDATE predictions
                    SET soil_data = ?, satellite_data = ?, weather_data = ?
                    WHERE id = ?
                """, (json.dumps(soil_fixed), json.dumps(satellite_fixed), json.dumps(weather_fixed), pred_id))

            cursor.execute("SELECT id, data FROM sync_queue")
            for row in cursor.fetchall():
                op_id = row["id"]
                data_fixed = self._safe_load_json(row["data"])
                if json.dumps(data_fixed) != (row["data"] or "{}"):
                    logger.info(f"Fixed malformed JSON in sync_queue ID {op_id}")
                cursor.execute("UPDATE sync_queue SET data = ? WHERE id = ?", (json.dumps(data_fixed), op_id))

            self.sqlite_conn.commit()
            logger.info("✅ Malformed JSON cleanup completed with fixes logged")
        except Exception as e:
            logger.error(f"Error during JSON cleanup: {e}")

    def _add_to_sync_queue(self, collection_name: str, operation: str, data: dict):
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO sync_queue (collection_name, operation, data, timestamp)
                VALUES (?, ?, ?, ?)
            """, (collection_name, operation, json.dumps(data), datetime.utcnow().isoformat()))
            self.sqlite_conn.commit()
            logger.info(f"✅ Added operation to sync_queue: {operation} on {collection_name}")
        except Exception as e:
            logger.error(f"❌ Failed to add operation to sync_queue: {e}")

    def sync_offline_data(self):
        """Synchronize unsynced SQLite predictions and queued operations to MongoDB"""
        if self.predictions_collection is None:
            logger.warning("MongoDB not available, skipping sync")
            return False

        synced_count = 0
        ops_count = 0

        try:
            cursor = self.sqlite_conn.cursor()

            # --- Sync ALL unsynced predictions (remove time filter) ---
            cursor.execute("SELECT * FROM predictions WHERE synced = 0")
            unsynced_predictions = cursor.fetchall()
            logger.info(f"🔄 Found {len(unsynced_predictions)} unsynced predictions in SQLite")

            for prediction in unsynced_predictions:
                try:
                    # Prepare data for MongoDB with proper JSON parsing
                    prediction_data = {
                        "image_filename": prediction["image_filename"],
                        "disease": prediction["disease"],
                        "confidence": float(prediction["confidence"]),
                        "treatment": prediction["treatment"],
                        "soil_data": self._safe_load_json(prediction["soil_data"]),
                        "satellite_data": self._safe_load_json(prediction["satellite_data"]),
                        "weather_data": self._safe_load_json(prediction["weather_data"]),
                        "timestamp": prediction["timestamp"],  # Keep as string for MongoDB
                        "sqlite_id": prediction["id"]  # Track original ID
                    }

                    # Validate required fields
                    if not prediction_data["image_filename"] or prediction_data["image_filename"] == "unknown":
                        logger.warning(f"⚠️ Skipping prediction {prediction['id']} - invalid filename")
                        continue

                    # Attempt sync
                    result = self.predictions_collection.insert_one(prediction_data)
                    
                    # Mark as synced in SQLite
                    with self.sqlite_lock:
                        cursor.execute("UPDATE predictions SET synced = 1 WHERE id = ?", (prediction["id"],))
                        self.sqlite_conn.commit()
                    
                    synced_count += 1
                    logger.info(f"✅ Synced prediction ID {prediction['id']} to MongoDB")

                except Exception as e:
                    logger.error(f"❌ Failed to sync prediction ID {prediction['id']}: {e}")
                    # Don't break - continue with other predictions

            # --- Sync queued operations ---
            cursor.execute("SELECT * FROM sync_queue WHERE attempts < 5")
            queued_operations = cursor.fetchall()
            logger.info(f"🔄 Found {len(queued_operations)} queued operations to sync")

            for operation in queued_operations:
                try:
                    data = self._safe_load_json(operation["data"])
                    collection_name = operation["collection_name"]
                    collection = self.mongo_db.get_collection(collection_name)

                    if operation["operation"] == "insert":
                        result = collection.insert_one(data)
                        logger.info(f"✅ Inserted queued operation ID {operation['id']}")

                    # Remove from queue after successful sync
                    cursor.execute("DELETE FROM sync_queue WHERE id = ?", (operation["id"],))
                    ops_count += 1
                    self.sqlite_conn.commit()

                except Exception as e:
                    logger.error(f"❌ Failed to sync queued operation ID {operation['id']}: {e}")
                    # Increment attempt count
                    cursor.execute(
                        "UPDATE sync_queue SET attempts = attempts + 1, last_attempt = CURRENT_TIMESTAMP WHERE id = ?",
                        (operation["id"],)
                    )
                    self.sqlite_conn.commit()

            logger.info(f"✅ Successfully synced {synced_count} predictions and {ops_count} operations")
            return synced_count > 0 or ops_count > 0

        except Exception as e:
            logger.error(f"❌ Error during sync: {e}")
            return False

    def trigger_sync_now(self):
        """Manually trigger the offline-to-MongoDB sync (useful for testing)."""
        logger.info("⏳ Manual sync triggered...")
        success = self.sync_offline_data()
        if success:
            logger.info("✅ Manual sync completed successfully")
        else:
            logger.warning("⚠️ Manual sync did not complete")

    def get_recent_predictions(self, limit: int = 10):
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
            """, (limit,))
            predictions = []
            for row in cursor.fetchall():
                predictions.append({
                    "id": row["id"],
                    "image_filename": row["image_filename"],
                    "disease": row["disease"],
                    "confidence": row["confidence"],
                    "treatment": row["treatment"],
                    "soil_data": self._safe_load_json(row["soil_data"]),
                    "satellite_data": self._safe_load_json(row["satellite_data"]),
                    "weather_data": self._safe_load_json(row["weather_data"]),
                    "timestamp": row["timestamp"],
                    "synced": bool(row["synced"])
                })
            return predictions
        except Exception as e:
            logger.error(f"Error fetching recent predictions: {e}")
            return []

    def close(self):
        try:
            if self.mongo_client:
                self.mongo_client.close()
            if self.sqlite_conn:
                self.sqlite_conn.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

async def test_offline_sync():
    db_service = DatabaseService()

    # --- Simulate MongoDB being down ---
    db_service.mongo_db = None
    db_service.predictions_collection = None
    print("Simulating offline mode: MongoDB is unavailable.")

    # --- Create dummy predictions ---
    predictions = [
        {
            "image_filename": f"test_image_{i}.jpg",
            "disease": "Late Blight" if i % 2 == 0 else "Healthy",
            "confidence": 0.85 + i * 0.01,
            "treatment": "Fungicide" if i % 2 == 0 else "None",
            "soil_data": {"ph_level": 6.0 + i*0.1},
            "satellite_data": {"ndvi": 0.2 + i*0.05},
            "weather_data": {"humidity": 75 + i},
            "timestamp": datetime.now().isoformat()
        }
        for i in range(3)
    ]

    # --- Save predictions (will queue them for sync) ---
    for pred in predictions:
        pred_id = db_service.save_prediction(pred)
        print(f"Prediction saved offline with SQLite ID: {pred_id}")

    # Check SQLite and CSV logs
    recent = db_service.get_recent_predictions(limit=5)
    print("\nRecent predictions saved in SQLite (offline):")
    for r in recent:
        print(r)

    # --- Simulate MongoDB coming back online ---
    try:
        mongo_client = MongoClient(settings.MONGODB_ATLAS_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command("ping")
        db_service.mongo_db = mongo_client[settings.MONGODB_ATLAS_DB_NAME]
        db_service.predictions_collection = db_service.mongo_db.get_collection(settings.COLLECTION_NAME)
        print("\nMongoDB back online! Syncing offline data...")
    except Exception:
        print("\nMongoDB still unavailable. Offline queue will remain.")

    # --- Sync offline data ---
    synced = db_service.sync_offline_data()
    if synced:
        print("\n✅ Sync completed successfully (True)")
    else:
        print("\n⚠️ Sync did not complete (False)")

    # --- Verify remaining sync_queue entries ---
    cursor = db_service.sqlite_conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM sync_queue")
    queue_count = cursor.fetchone()["cnt"]
    print(f"Remaining operations in sync_queue: {queue_count}")

    # --- Close connections ---
    db_service.close()

# Run the test
if __name__ == "__main__":
    asyncio.run(test_offline_sync())