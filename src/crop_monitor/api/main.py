# # src/crop_monitor/main.py
# import os
# import csv
# import asyncio
# from collections import OrderedDict
# from typing import Dict, Any
# from datetime import datetime
# from fastapi import Form, FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import anyio
# import traceback
# from fastapi import Body
# from PIL import Image
# import io
# import numpy as np
# import mlflow

# # Import services
# from src.crop_monitor.services.disease_service import DiseaseService
# from src.crop_monitor.services.soil_service import SoilService
# from src.crop_monitor.services.satellite_service import SatelliteService
# from src.crop_monitor.services.weather_service import WeatherService
# from src.crop_monitor.services.database_service import DatabaseService
# from src.crop_monitor.core.models import PredictionResult

# # Import Pydantic schemas and settings
# from src.crop_monitor.core.schemas import PredictionResponse, Location
# from src.crop_monitor.config.settings import settings
# from src.crop_monitor.config.logging_config import setup_logging

# # ------------------ Logging ------------------ #
# logger = setup_logging()

# # ------------------ Initialize Services ------------------ #
# disease_service = DiseaseService()
# soil_service = SoilService(
#     soil_data_path=settings.SOIL_DATA_PATH,
#     model_path=settings.soil_model_path
# )
# satellite_service = SatelliteService()
# weather_service = WeatherService()
# database_service = DatabaseService()

# # ------------------ FastAPI App ------------------ #
# app = FastAPI(title="AI Crop Monitor API", version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # For development only
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ------------------ CSV Logging ------------------ #
# CSV_FILE_PATH = getattr(settings, "CSV_FILE_PATH", None) or os.getenv(
#     "CSV_FILE_PATH",
#     os.path.join(os.getcwd(), "predictions_log.csv")
# )

# csv_dir = os.path.dirname(CSV_FILE_PATH)
# if csv_dir and not os.path.exists(csv_dir):
#     os.makedirs(csv_dir, exist_ok=True)
#     logger.info(f"Created missing directory for CSV: {csv_dir}")

# logger.info(f"Predictions CSV will be saved at: {CSV_FILE_PATH}")


# def append_prediction_to_csv(pred: PredictionResult):
#     """
#     Robustly append prediction results to a CSV file.
#     Creates the CSV if missing and dynamically updates headers.
#     Ensures all keys are strings and prevents silent failures.
#     """
#     try:
#         soil_data = pred.soil_data or {}
#         satellite_data = pred.satellite_data or {}
#         weather_data = pred.weather_data or {}

#         # Prefix data keys to avoid collisions
#         soil_prefixed = {f"soil_{k}": v for k, v in soil_data.items()}
#         satellite_prefixed = {f"satellite_{k}": v for k, v in satellite_data.items()}
#         weather_prefixed = {f"weather_{k}": v for k, v in weather_data.items()}

#         row_data = OrderedDict(
#             timestamp=str(pred.timestamp),
#             image_filename=str(pred.image_filename),
#             disease=str(pred.disease),
#             confidence=float(pred.confidence),
#             treatment=str(pred.treatment),
#             **{str(k): v for k, v in soil_prefixed.items()},
#             **{str(k): v for k, v in satellite_prefixed.items()},
#             **{str(k): v for k, v in weather_prefixed.items()}
#         )

#         csv_path = CSV_FILE_PATH
#         csv_dir = os.path.dirname(csv_path)
#         if csv_dir and not os.path.exists(csv_dir):
#             os.makedirs(csv_dir, exist_ok=True)
#             logger.info(f"Created missing directory for CSV: {csv_dir}")

#         # Determine current fieldnames
#         file_exists = os.path.isfile(csv_path)
#         if file_exists:
#             with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
#                 reader = csv.DictReader(f)
#                 fieldnames = reader.fieldnames or []
#             # Add any new keys dynamically
#             for key in row_data.keys():
#                 if key not in fieldnames:
#                     fieldnames.append(key)
#         else:
#             fieldnames = list(row_data.keys())

#         # Write the row
#         with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             if not file_exists:
#                 writer.writeheader()
#             writer.writerow(row_data)

#         logger.info(f"Prediction appended to CSV successfully: {csv_path}")

#     except Exception as e:
#         logger.error(f"Failed to append prediction to CSV: {e}")
#         logger.error(traceback.format_exc())


# # ------------------ API Endpoints ------------------ #
# @app.get("/")
# async def root():
#     return {"status": "AI Crop Monitor API is running", "environment": settings.APP_ENV}


# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "environment": settings.APP_ENV}


# @app.post("/predict", response_model=PredictionResponse)
# async def predict(
#     file: UploadFile = File(...),
#     lat: float = Form(...),
#     lon: float = Form(...),
# ):
#     location = Location(lat=lat, lon=lon)

#     try:
#         logger.info(f"Starting prediction for location: {location}")

#         # ------------------ File Logging & Preprocessing ------------------ #
#         file_content = await file.read()
#         await file.seek(0)
#         try:
#             img = Image.open(io.BytesIO(file_content))

#             # Capture original format and size before conversion/resizing
#             orig_format = img.format
#             orig_size = img.size

#             # Convert and resize for model input
#             img = img.convert("RGB")
#             #img = img.resize((128, 128))  # Resize to model input size

#             logger.info(
#                 f"Uploaded image - Original Format: {orig_format}, "
#                 f"Original Size: {orig_size}, Resized Size: {img.size}, Mode: {img.mode}"
#             )
#         except Exception as img_e:
#             logger.warning(f"Unable to read image: {img_e}")
#             raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

#         # ------------------ Disease Prediction ------------------ #
#         try:
#             disease_result = await disease_service.predict_disease_unified(
#                 files=file,                       # single file or list of files
#                 lats=[lat],                       # wrap single lat in list
#                 lons=[lon],                       # wrap single lon in list
#                 soil_service=soil_service,
#                 weather_service=weather_service,
#                 db_sqlite_conn=database_service.sqlite_conn,  # your SQLite connection
#                 mongo_collection=database_service.mongo_db,   # your MongoDB collection
#                 output_csv_path=CSV_FILE_PATH
#             )
#             # Since it's a list, take the first prediction for a single file
#             disease_result = disease_result[0] if disease_result else {}
#             predicted_disease = disease_result.get("disease", "Unavailable")
#             confidence = disease_result.get("confidence", 0.0)

#             predicted_disease = disease_result.get("disease", "Unavailable")
#             confidence = disease_result.get("confidence", 0.0)
#             logger.info(f"Disease predicted: {predicted_disease}, confidence: {confidence}")

#             mlflow.log_param("file_name", file.filename)
#             mlflow.log_param("lat", lat)
#             mlflow.log_param("lon", lon)
#             mlflow.log_metric("confidence", confidence)

#         except Exception as e:
#             logger.error(f"Disease detection error: {e}")
#             logger.error(traceback.format_exc())
#             predicted_disease, confidence = "Unavailable", 0.0

#         # ------------------ Soil Analysis ------------------ #
#         try:
#             raw_soil = await anyio.to_thread.run_sync(
#                 lambda: soil_service.get_soil_analysis(location.lat, location.lon)
#             )
#             logger.info(f"Soil analysis raw result: {raw_soil}")

#             if raw_soil and "soil_type" in raw_soil:
#                 soil_result = {
#                     "soil_type": raw_soil.get("soil_type", "unknown"),
#                     "risk_level": raw_soil.get("risk_level", "Low"),
#                     "source": "soil_features.json"
#                 }
#             else:
#                 soil_result = {
#                     "soil_type": "unknown",
#                     "warning": "Soil data may be generalized",
#                     "risk_level": "Medium",
#                     "source": "fallback"
#                 }

#         except Exception as e:
#             logger.error(f"Soil pipeline error: {e}")
#             logger.error(traceback.format_exc())
#             soil_result = {
#                 "soil_type": "unknown",
#                 "warning": "Soil data may be generalized",
#                 "risk_level": "Medium",
#                 "source": "fallback"
#             }


#         # ------------------ Satellite Data ------------------ #
#         try:
#             satellite_result = await anyio.to_thread.run_sync(
#                 lambda: satellite_service.get_vegetation_data(location.lat, location.lon)
#             )
#             logger.info(f"Satellite data result: {satellite_result}")
#         except Exception as e:
#             logger.error(f"Satellite pipeline error: {e}")
#             logger.error(traceback.format_exc())
#             satellite_result = {}

#         # ------------------ Weather Data ------------------ #
#         try:
#             weather_result = await anyio.to_thread.run_sync(
#                 lambda: weather_service.get_weather_data(location.lat, location.lon)
#             )
#             logger.info(f"Weather data result: {weather_result}")
#         except Exception as e:
#             logger.error(f"Weather pipeline error: {e}")
#             logger.error(traceback.format_exc())
#             weather_result = {}

#         # ------------------ Treatment Recommendation ------------------ #
#         try:
#             treatment = disease_service.enhance_recommendation(
#                 disease=predicted_disease,
#                 weather_data=weather_result,
#                 soil_data=soil_result,
#                 satellite_data=satellite_result
#             )
#             logger.info(f"Treatment recommendation: {treatment}")
#         except Exception as e:
#             logger.error(f"Treatment recommendation error: {e}")
#             logger.error(traceback.format_exc())
#             treatment = "Treatment recommendation unavailable."

#         # ------------------ Save Prediction ------------------ #
#         prediction_result = PredictionResult(
#             image_filename=file.filename,
#             disease=predicted_disease,
#             confidence=confidence,
#             treatment=treatment,
#             soil_data=soil_result,
#             satellite_data=satellite_result,
#             weather_data=weather_result,
#             timestamp=datetime.now()
#         )

#         try:
#             database_service.save_prediction(prediction_result)
#             logger.info("Prediction saved to database successfully.")
#         except Exception as e:
#             logger.error(f"Database save error: {e}")
#             logger.error(traceback.format_exc())

#         try:
#             append_prediction_to_csv(prediction_result)
#             logger.info("Prediction appended to CSV successfully.")
#         except Exception as e:
#             logger.error(f"CSV save error: {e}")
#             logger.error(traceback.format_exc())

#         return PredictionResponse(
#             disease=predicted_disease,
#             confidence=confidence,
#             treatment=treatment,
#             soil=soil_result,
#             satellite=satellite_result,
#             weather=weather_result
#         )

#     except Exception as e:
#         logger.error(f"Prediction pipeline global error: {e}")
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail="Unexpected server error")
    
# @app.post("/predict/batch", response_model=list[PredictionResponse])
# async def predict_batch(
#     files: list[UploadFile] = File(...),
#     lats: list[float] = Form(None),
#     lons: list[float] = Form(None)
# ):

#     """
#     Batch prediction endpoint.
#     Accepts multiple images and matching lat/lon lists.
#     Returns a list of fully populated PredictionResponse objects.
#     """
#     raw_results = await disease_service.predict_disease_unified(
#         files=files,
#         lats=lats,
#         lons=lons,
#         soil_service=soil_service,
#         weather_service=weather_service,
#         db_sqlite_conn=database_service.sqlite_conn,
#         mongo_collection=database_service.mongo_db,
#         output_csv_path=CSV_FILE_PATH
#     )

#     formatted_results = []
#     for idx, res in enumerate(raw_results):
#         formatted_results.append(PredictionResponse(
#             disease=res.get("disease", "Unavailable"),
#             confidence=res.get("confidence", 0.0),
#             treatment=res.get("treatment") or disease_service.enhance_recommendation(
#                 disease=res.get("disease", "Unavailable"),
#                 weather_data=res.get("weather") or {},
#                 soil_data=res.get("soil") or {},
#                 satellite_data=res.get("satellite") or {}
#             ),
#             soil=res.get("soil") or {},
#             satellite=res.get("satellite") or {},
#             weather=res.get("weather") or {}
#         ))

#     return formatted_results



# # ------------------ Other Endpoints ------------------ #
# @app.get("/sync")
# async def sync_data():
#     try:
#         success = database_service.sync_offline_data()
#         return {"status": "success" if success else "partial_failure", "message": "Data synchronization completed"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


# @app.get("/predictions/recent")
# async def get_recent_predictions(limit: int = 10):
#     try:
#         predictions = database_service.get_recent_predictions(limit)
#         return {"predictions": predictions}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to fetch predictions: {str(e)}")


# @app.get("/status")
# async def get_system_status():
#     mongo_status = "connected" if database_service.mongo_db else "disconnected"
#     return {
#         "status": "healthy",
#         "databases": {
#             "mongodb": mongo_status,
#             "sqlite": "connected"
#         },
#         "environment": settings.APP_ENV
#     }


# # ------------------ Background Sync ------------------ #
# stop_event = asyncio.Event()

# async def periodic_sync():
#     """
#     Periodically sync offline data every 5 minutes (300s).
#     Logs success/failure and handles exceptions without stopping the loop.
#     """
#     while not stop_event.is_set():
#         try:
#             logger.info("Starting periodic offline data sync...")
#             success = await anyio.to_thread.run_sync(database_service.sync_offline_data)
#             if success:
#                 logger.info("Periodic sync completed successfully.")
#             else:
#                 logger.warning("Periodic sync completed with partial failures.")
#         except Exception as e:
#             logger.error(f"Periodic sync encountered an error: {e}")
#             logger.error(traceback.format_exc())
#         await asyncio.sleep(300)  # Wait 5 minutes before next sync

# @app.on_event("startup")
# async def startup_event():
#     """
#     Startup event: starts the periodic sync in background.
#     """
#     asyncio.create_task(periodic_sync())
#     logger.info("Periodic sync task started.")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """
#     Shutdown event: stops the periodic sync.
#     """
#     stop_event.set()
#     logger.info("Periodic sync task stopped.")



# @app.get("/soil")
# async def get_soil(lat: float, lon: float):
#     """Returns SoilGrids soil info for any coordinates."""
#     try:
#         soil_info = await anyio.to_thread.run_sync(
#             lambda: soil_service.get_soil_analysis(lat, lon)
#         )
#         return {"soil": soil_info}
#     except Exception as e:
#         logger.error(f"Error fetching soil info: {e}")
#         raise HTTPException(status_code=500, detail=f"Soil service error: {e}")








































# src/crop_monitor/main.py
import os
import csv
import asyncio
from collections import OrderedDict
from typing import Dict, Any
from datetime import datetime
from fastapi import Form, FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import anyio
import traceback
from io import BytesIO
import base64
from PIL import Image
import io
import numpy as np
import mlflow
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from datetime import datetime
import anyio
import os
# import csv as _csv
import csv
import logging
import traceback
from typing_extensions import Annotated
from typing import List, Union
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from collections import OrderedDict
from pathlib import Path
from fastapi import Request
import re
import json
import uuid
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Import services
from src.crop_monitor.services.disease_service import DiseaseService
from src.crop_monitor.services.soil_service import SoilService
from src.crop_monitor.services.satellite_service import SatelliteService
from src.crop_monitor.services.weather_service import WeatherService
from src.crop_monitor.services.database_service import DatabaseService
from src.crop_monitor.core.models import PredictionResult

# Import Pydantic schemas and settings
from src.crop_monitor.core.schemas import PredictionResponse, Location
from src.crop_monitor.config.settings import settings
from src.crop_monitor.config.logging_config import setup_logging

# ------------------ Logging ------------------ #
logger = setup_logging()

# ------------------ Initialize Services ------------------ #
disease_service = DiseaseService()
soil_service = SoilService(
    soil_data_path=settings.SOIL_DATA_PATH
    # model_path=settings.SOIL_MODEL_PATH
)
satellite_service = SatelliteService()
weather_service = WeatherService()
database_service = DatabaseService()

# ------------------ FastAPI App ------------------ #
app = FastAPI(title="AI Crop Monitor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ CSV Logging ------------------ #
CSV_FILE_PATH = getattr(settings, "CSV_FILE_PATH", None) or os.getenv(
    "CSV_FILE_PATH",
    os.path.join(os.getcwd(), "predictions_log.csv")
)

csv_dir = os.path.dirname(CSV_FILE_PATH)
if csv_dir and not os.path.exists(csv_dir):
    os.makedirs(csv_dir, exist_ok=True)
    logger.info(f"Created missing directory for CSV: {csv_dir}")

logger.info(f"Predictions CSV will be saved at: {CSV_FILE_PATH}")


def append_prediction_to_csv(pred: PredictionResult):
    """
    Robustly append prediction results to a CSV file.
    Creates the CSV if missing and dynamically updates headers.
    Ensures all keys are strings and prevents silent failures.
    """
    try:
        soil_data = pred.soil_data or {}
        satellite_data = pred.satellite_data or {}
        weather_data = pred.weather_data or {}

        # Prefix data keys to avoid collisions
        soil_prefixed = {f"soil_{k}": v for k, v in soil_data.items()}
        satellite_prefixed = {f"satellite_{k}": v for k, v in satellite_data.items()}
        weather_prefixed = {f"weather_{k}": v for k, v in weather_data.items()}

        row_data = OrderedDict(
            timestamp=str(pred.timestamp),
            image_filename=str(pred.image_filename),
            disease=str(pred.disease),
            confidence=float(pred.confidence),
            treatment=str(pred.treatment),
            **{str(k): v for k, v in soil_prefixed.items()},
            **{str(k): v for k, v in satellite_prefixed.items()},
            **{str(k): v for k, v in weather_prefixed.items()}
        )

        csv_path = CSV_FILE_PATH
        csv_dir = os.path.dirname(csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
            logger.info(f"Created missing directory for CSV: {csv_dir}")

        # Determine current fieldnames
        file_exists = os.path.isfile(csv_path)
        if file_exists:
            with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
            # Add any new keys dynamically
            for key in row_data.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        else:
            fieldnames = list(row_data.keys())

        # Write the row
        with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

        logger.info(f"Prediction appended to CSV successfully: {csv_path}")

    except Exception as e:
        logger.error(f"Failed to append prediction to CSV: {e}")
        logger.error(traceback.format_exc())


# ------------------ API Endpoints ------------------ #
@app.get("/")
async def root():
    return {"status": "AI Crop Monitor API is running", "environment": settings.APP_ENV}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": settings.APP_ENV}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
):
    location = Location(lat=lat, lon=lon)

    try:
        logger.info(f"Starting prediction for location: {location}")

        # ------------------ File Logging & Preprocessing ------------------ #
        file_content = await file.read()
        await file.seek(0)
        try:
            img = Image.open(io.BytesIO(file_content))

            # Capture original format and size before conversion/resizing
            orig_format = img.format
            orig_size = img.size

            # Convert and resize for model input
            img = img.convert("RGB")
            #img = img.resize((128, 128))  # Resize to model input size

            logger.info(
                f"Uploaded image - Original Format: {orig_format}, "
                f"Original Size: {orig_size}, Resized Size: {img.size}, Mode: {img.mode}"
            )
        except Exception as img_e:
            logger.warning(f"Unable to read image: {img_e}")
            raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

        # ------------------ Disease Prediction ------------------ #
        try:
            file_content = await file.read()
            await file.seek(0)
            disease_result = await disease_service.predict_disease_unified(
                files=[file],                       
                lats=[lat],                       
                lons=[lon],                       
                soil_service=soil_service,
                weather_service=weather_service,
                db_sqlite_conn=database_service.sqlite_conn,                  
                predictions_collection=database_service.predictions_collection,
                output_csv_path=CSV_FILE_PATH
            )
            
            disease_result = disease_result[0] if disease_result else {}
            predicted_disease = disease_result.get("disease", "Unavailable")
            confidence = disease_result.get("confidence", 0.0)

            predicted_disease = disease_result.get("disease", "Unavailable")
            confidence = disease_result.get("confidence", 0.0)
            logger.info(f"Disease predicted: {predicted_disease}, confidence: {confidence}")

            mlflow.log_param("file_name", file.filename)
            mlflow.log_param("lat", lat)
            mlflow.log_param("lon", lon)
            mlflow.log_metric("confidence", confidence)

        except Exception as e:
            logger.error(f"Disease detection error: {e}")
            logger.error(traceback.format_exc())
            predicted_disease, confidence = "Unavailable", 0.0

        # ------------------ Soil Analysis ------------------ #
        try:
            raw_soil = await anyio.to_thread.run_sync(
                lambda: soil_service.get_soil_analysis(location.lat, location.lon)
            )
            logger.info(f"Soil analysis raw result: {raw_soil}")

            if raw_soil and "soil_type" in raw_soil:
                soil_result = {
                    "soil_type": raw_soil.get("soil_type", "unknown"),
                    "risk_level": raw_soil.get("risk_level", "Low"),
                    "source": "soil_features.json"
                }
            else:
                soil_result = {
                    "soil_type": "unknown",
                    "warning": "Soil data may be generalized",
                    "risk_level": "Medium",
                    "source": "fallback"
                }

        except Exception as e:
            logger.error(f"Soil pipeline error: {e}")
            logger.error(traceback.format_exc())
            soil_result = {
                "soil_type": "unknown",
                "warning": "Soil data may be generalized",
                "risk_level": "Medium",
                "source": "fallback"
            }


        # ------------------ Satellite Data ------------------ #
        try:
            satellite_result = await anyio.to_thread.run_sync(
                lambda: satellite_service.get_vegetation_data(location.lat, location.lon)
            )
            logger.info(f"Satellite data result: {satellite_result}")
        except Exception as e:
            logger.error(f"Satellite pipeline error: {e}")
            logger.error(traceback.format_exc())
            satellite_result = {}

        # ------------------ Weather Data ------------------ #
        try:
            weather_result = await anyio.to_thread.run_sync(
                lambda: weather_service.get_weather_data(location.lat, location.lon)
            )
            logger.info(f"Weather data result: {weather_result}")
        except Exception as e:
            logger.error(f"Weather pipeline error: {e}")
            logger.error(traceback.format_exc())
            weather_result = {}

        # ------------------ Treatment Recommendation ------------------ #
        try:
            treatment = disease_service.enhance_recommendation(
                disease=predicted_disease,
                weather_data=weather_result,
                soil_data=soil_result,
                satellite_data=satellite_result
            )
            logger.info(f"Treatment recommendation: {treatment}")
        except Exception as e:
            logger.error(f"Treatment recommendation error: {e}")
            logger.error(traceback.format_exc())
            treatment = "Treatment recommendation unavailable."


        # ------------------ Save Prediction ------------------ #
        prediction_result = PredictionResult(
            image_filename=file.filename,
            disease=predicted_disease,
            confidence=confidence,
            treatment=treatment,
            soil_data=soil_result,
            satellite_data=satellite_result,
            weather_data=weather_result,
            timestamp=datetime.now()
        )

        try:
            database_service.save_prediction(prediction_result)
            logger.info("Prediction saved to database successfully.")
        except Exception as e:
            logger.error(f"Database save error: {e}")
            logger.error(traceback.format_exc())

        try:
            append_prediction_to_csv(prediction_result)
            logger.info("Prediction appended to CSV successfully.")
        except Exception as e:
            logger.error(f"CSV save error: {e}")
            logger.error(traceback.format_exc())

        return PredictionResponse(
            disease=predicted_disease,
            confidence=confidence,
            treatment=treatment,
            soil=soil_result,
            satellite=satellite_result,
            weather=weather_result
        )

    except Exception as e:
        logger.error(f"Prediction pipeline global error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Unexpected server error")
     


# ------------------ Helper: Flatten dictionary ------------------ #
def _flatten_dict(prefix: str, d: dict, all_keys: set) -> dict:
    """Recursively flatten a dict and record keys into all_keys."""
    flat = {}

    def _recurse(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _recurse(v, path + [str(k)])
        else:
            key = prefix + "_" + "_".join(path)
            flat[key] = obj
            all_keys.add(key)
    _recurse(d, [])
    return flat


logger = logging.getLogger(__name__)

# --------------------------- Helper ---------------------------
def _parse_number_list(input_list: List[str]) -> List[float]:
    """Parse list of strings or comma-separated numbers into floats."""
    result = []
    for item in input_list:
        parts = str(item).split(",")
        for part in parts:
            part = part.strip()
            if part:  # ignore empty strings
                try:
                    result.append(float(part))
                except ValueError:
                    raise HTTPException(status_code=422, detail=f"Invalid number: {part}")
    return result

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    files: List[UploadFile] = File(..., description="Upload one or more images"),
    lats: List[str] = Form(..., description="Latitudes (comma-separated or multiple)"),
    lons: List[str] = Form(..., description="Longitudes (comma-separated or multiple)")
):
    """Batch prediction endpoint with file + lat/lon input, fully robust."""

    # Clear cache before batch prediction - FIX FOR SAME RESULTS
    disease_service.clear_prediction_cache()

    # --- Validate non-empty files ---
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded for prediction")

    # --- parse lat/lon robustly ---
    lat_list = _parse_number_list(lats)
    lon_list = _parse_number_list(lons)

    # --- validate length matches ---
    if len(files) != len(lat_list) or len(files) != len(lon_list):
        raise HTTPException(
            status_code=422,
            detail=f"Number of uploaded images ({len(files)}) must match number of latitudes ({len(lat_list)}) "
                   f"and longitudes ({len(lon_list)})."
        )

    # --- perform unified disease prediction ---
    try:
        raw_results = await disease_service.predict_disease_unified(
            files=files,
            lats=lat_list,
            lons=lon_list,
            soil_service=soil_service,
            weather_service=weather_service,
            db_sqlite_conn=database_service.sqlite_conn,
            # mongo_collection=database_service.mongo_db,
            mongo_collection=database_service.predictions_collection,
            output_csv_path=database_service.csv_path
        )
    except Exception as e:
        logger.error(f"Unified disease prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Disease prediction failed")

    formatted_results, analytics_rows, all_flat_keys = [], [], set()

    # --- ensure prediction images directory exists ---
    images_dir = Path(settings.BASE_PREDICTIONS_DIR) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    numeric_to_label = {
        0: "sandy", 1: "loam", 2: "clay", 3: "silty",
        4: "peaty", 5: "chalky", 6: "organic"
    }

    for idx, res in enumerate(raw_results):
        # --- skip if index out of bounds for files ---
        if idx >= len(files):
            logger.warning(f"Skipping prediction result index {idx} – no corresponding file uploaded")
            continue

        upload_file = files[idx]

        # --- soil data ---
        try:
            raw_soil = await anyio.to_thread.run_sync(
                lambda: soil_service.get_soil_analysis(lat_list[idx], lon_list[idx])
            )
        except Exception as e:
            logger.warning(f"Soil fetch failed for index {idx}: {e}")
            raw_soil = soil_service.get_fallback_soil_data()

        clay = raw_soil.get("clay", raw_soil.get("Clay(%)")) or soil_service._default_value("clay")
        silt = raw_soil.get("silt", raw_soil.get("Silt(%)")) or soil_service._default_value("silt")
        sand = raw_soil.get("sand", raw_soil.get("Sand(%)")) or soil_service._default_value("sand")
        ph = raw_soil.get("ph_level", raw_soil.get("ph")) or soil_service._default_value("ph_level")
        organic = raw_soil.get("organic_matter", raw_soil.get("SOC(g/kg)")) or soil_service._default_value("organic_matter")
        cec = raw_soil.get("cec", raw_soil.get("CEC", 15))

        # --- soil type label safely ---
        soil_type_raw = raw_soil.get("soil_type")
        try:
            if isinstance(soil_type_raw, (int, float)):
                soil_type_label = numeric_to_label.get(int(round(float(soil_type_raw))), str(soil_type_raw))
            elif isinstance(soil_type_raw, str):
                if re.fullmatch(r"^-?\d+(\.\d+)?$", soil_type_raw.strip()):
                    soil_type_label = numeric_to_label.get(int(round(float(soil_type_raw.strip()))), soil_type_raw)
                else:
                    soil_type_label = soil_type_raw
            elif soil_type_raw is None:
                soil_type_label = soil_service._predict_soil_type({
                    "clay": clay, "silt": silt, "sand": sand, "ph_level": ph
                })
            else:
                soil_type_label = str(soil_type_raw)
        except Exception:
            soil_type_label = "unknown"

        # --- risk level ---
        health_score = raw_soil.get("health_score")
        if health_score is not None:
            try:
                hs = float(health_score)
                risk_level = "High" if hs < 0.4 else "Medium" if hs < 0.7 else "Low"
            except Exception:
                risk_level = raw_soil.get("risk_level", "Medium")
        else:
            risk_level = raw_soil.get("risk_level", "Medium")

        # --- satellite data ---
        try:
            sat_data = await anyio.to_thread.run_sync(
                lambda: satellite_service.get_vegetation_data(lat_list[idx], lon_list[idx])
            )
        except Exception:
            sat_data = {}

        # --- weather data ---
        weather_data = res.get("weather") or {}
        if not weather_data:
            try:
                weather_data = await anyio.to_thread.run_sync(
                    lambda: weather_service.get_weather_data(lat_list[idx], lon_list[idx])
                )
            except Exception:
                weather_data = {}

        # --- image saving ---
        try:
            contents = await upload_file.read()
            upload_file.file.seek(0)

            tmp_path = Path("/tmp") / upload_file.filename
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(contents)

            permanent_name = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex}_{upload_file.filename}"
            permanent_path = images_dir / permanent_name
            permanent_path.write_bytes(contents)
        except Exception as e:
            logger.warning(f"Failed saving image for index {idx}: {e}")
            permanent_path = ""

        soil_result = {
            "soil_type": soil_type_label,
            "risk_level": risk_level,
            "source": raw_soil.get("source", "model_prediction"),
            "sand": float(sand),
            "silt": float(silt),
            "clay": float(clay),
            "organic_matter": float(organic),
            "ph_level": float(ph),
            "cec": float(cec)
        }

        # --- prediction data ---
        prediction_data = {
            "disease": res.get("disease", "Unavailable"),
            "confidence": res.get("confidence", 0.0),
            "treatment": res.get("treatment") or disease_service.enhance_recommendation(
                disease=res.get("disease", "Unavailable"),
                weather_data=weather_data,
                soil_data=soil_result,
                satellite_data=sat_data
            ),
            "soil_data": soil_result,
            "satellite_data": sat_data,
            "weather_data": weather_data,
            "lat": float(lat_list[idx]),
            "lon": float(lon_list[idx]),
            "timestamp": datetime.utcnow().isoformat(),
            "image_filename": getattr(upload_file, "filename", f"image_{idx}.jpg"),
            "stored_image_path": str(permanent_path)
        }

        # --- save to database ---
        try:
            database_service.save_prediction(prediction_data)
        except Exception as e:
            logger.error(f"Failed to save prediction for index {idx}: {e}")
            logger.error(traceback.format_exc())

        # --- prepare analytics row ---
        flat_soil = _flatten_dict("soil", prediction_data["soil_data"], all_flat_keys)
        flat_weather = _flatten_dict("weather", prediction_data["weather_data"], all_flat_keys)
        flat_sat = _flatten_dict("satellite", prediction_data["satellite_data"], all_flat_keys)

        analytics_rows.append({
            "timestamp": prediction_data["timestamp"],
            "image_filename": prediction_data["image_filename"],
            "stored_image_path": prediction_data["stored_image_path"],
            "lat": prediction_data["lat"],
            "lon": prediction_data["lon"],
            "disease": prediction_data["disease"],
            "confidence": prediction_data["confidence"],
            "treatment": prediction_data["treatment"],
            **flat_soil, **flat_weather, **flat_sat
        })

        # --- final response ---
        formatted_results.append({
            "disease": prediction_data["disease"],
            "confidence": prediction_data["confidence"],
            "treatment": prediction_data["treatment"],
            "soil": prediction_data["soil_data"],
            "satellite": prediction_data["satellite_data"],
            "weather": prediction_data["weather_data"]
        })

    # --- write analytics CSV ---
    if analytics_rows:
        csv_path = Path(settings.BASE_PREDICTIONS_DIR) / "analytics_ready_predictions.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["timestamp", "image_filename", "stored_image_path", "lat", "lon",
                      "disease", "confidence", "treatment"] + sorted(all_flat_keys)
        try:
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if f.tell() == 0:
                    writer.writeheader()
                for row in analytics_rows:
                    writer.writerow(row)
        except Exception as e:
            logger.error(f"Analytics CSV logging failed: {e}")
            logger.error(traceback.format_exc())

    return formatted_results


# ------------------ Other Endpoints ------------------ #
@app.get("/sync")
async def sync_data():
    try:
        success = database_service.sync_offline_data()
        return {"status": "success" if success else "partial_failure", "message": "Data synchronization completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")
    

@app.get("/predictions/recent")
async def get_recent_predictions(limit: int = 10):
    try:
        predictions = database_service.get_recent_predictions(limit)
        return {"predictions": predictions}
    except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch predictions: {str(e)}")


@app.get("/status")
async def get_system_status():
    mongo_status = "connected" if database_service.mongo_db else "disconnected"
    return {
        "status": "healthy",
        "databases": {
            "mongodb": mongo_status,
            "sqlite": "connected"
        },
        "environment": settings.APP_ENV
    }


@app.get("/debug/sync")
async def debug_sync():
    """Debug MongoDB sync status"""
    try:
        # Check SQLite unsynced predictions
        cursor = database_service.sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE synced = 0")
        unsynced_count = cursor.fetchone()["count"]
        
        # Check MongoDB connection
        mongo_status = "connected" if database_service.mongo_db else "disconnected"
        
        # Check recent predictions in both databases
        sqlite_recent = database_service.get_recent_predictions(limit=5)
        
        mongo_recent = []
        if database_service.predictions_collection:
            mongo_recent = list(database_service.predictions_collection.find().sort("timestamp", -1).limit(5))
        
        return {
            "sync_status": {
                "unsynced_predictions": unsynced_count,
                "mongodb_connection": mongo_status,
                "sqlite_total": len(sqlite_recent),
                "mongodb_total": len(mongo_recent)
            },
            "recent_sqlite": sqlite_recent,
            "recent_mongodb": mongo_recent
        }
        
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}    



# ------------------ Background Sync ------------------ #
stop_event = asyncio.Event()

async def periodic_sync():
    """
    Periodically sync offline data every 5 minutes (300s).
    Logs success/failure and handles exceptions without stopping the loop.
    """
    while not stop_event.is_set():
        try:
            logger.info("Starting periodic offline data sync...")
            success = await anyio.to_thread.run_sync(database_service.sync_offline_data)
            if success:
                logger.info("Periodic sync completed successfully.")
            else:
                logger.warning("Periodic sync completed with partial failures.")
        except Exception as e:
            logger.error(f"Periodic sync encountered an error: {e}")
            logger.error(traceback.format_exc())
        await asyncio.sleep(300)  # Wait 5 minutes before next sync

@app.on_event("startup")
async def startup_event():
    """
    Startup event: starts the periodic sync in background.
    """
    asyncio.create_task(periodic_sync())
    logger.info("Periodic sync task started.")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event: stops the periodic sync.
    """
    stop_event.set()
    logger.info("Periodic sync task stopped.")