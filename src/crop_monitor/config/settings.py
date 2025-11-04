# import os
# import json
# from pydantic import Field, field_validator, ValidationError
# from pydantic_settings import BaseSettings
# from typing import Optional, List, Any

# # Pre-compute base directories outside the Settings class
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# MODEL_DIR_DEFAULT = os.path.join(BASE_DIR, "src", "crop_monitor", "models")
# SOIL_DATA_DEFAULT = os.path.join(BASE_DIR, "data", "external", "soil_data.csv")
# CSV_FILE_PATH = os.getenv("CSV_FILE_PATH") or os.path.join(os.getcwd(), "predictions_log.csv")
# SOIL_FEATURES_PATH = r"C:\Users\ADEGOKE\Desktop\AI_Crop_Disease_Monitoring\src\crop_monitor\models\soil_features.json"
# SOILGRIDS_API_URL: str = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# # ✅ Database defaults
# SQLITE_DB_DEFAULT = os.path.join(BASE_DIR, "src", "crop_monitor", "db", "local_predictions.db")
# MONGODB_ATLAS_URI_DEFAULT = "mongodb+srv://AI_Crop_Monitor:CShxEb2FOjwU7pEj@cluster0.oqtogab.mongodb.net/AI_Crop_Monitor?retryWrites=true&w=majority&appName=Cluster0"
# MONGODB_ATLAS_DB_NAME_DEFAULT = "AI_Crop_Monitor"




# class Settings(BaseSettings):
#     # ================================
#     # FastAPI / Core
#     # ================================
#     APP_ENV: str = Field(default="development", env="APP_ENV")
#     APP_HOST: str = Field(default="0.0.0.0", env="APP_HOST")
#     APP_PORT: int = Field(default=8000, env="APP_PORT")
#     ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
#     DEBUG: bool = Field(default=True, env="DEBUG")
#     LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

#     # ================================
#     # Security
#     # ================================
#     SECRET_KEY: Optional[str] = Field(None, env="SECRET_KEY")
#     JWT_SECRET_KEY: Optional[str] = Field(None, env="JWT_SECRET_KEY")
#     JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
#     JWT_EXPIRE_MINUTES: int = Field(default=30, env="JWT_EXPIRE_MINUTES")
#     ENCRYPTION_KEY: Optional[str] = Field(None, env="ENCRYPTION_KEY")


#     # ================================
#     # Database
#     # ================================
#     MONGODB_ATLAS_URI: Optional[str] = Field(default=MONGODB_ATLAS_URI_DEFAULT, env="MONGODB_ATLAS_URI")
#     MONGODB_ATLAS_DB_NAME: Optional[str] = Field(default=MONGODB_ATLAS_DB_NAME_DEFAULT, env="MONGODB_ATLAS_DB_NAME")
#     SQLITE_DB_PATH: str = Field(default=SQLITE_DB_DEFAULT, env="SQLITE_DB_PATH")
#     COLLECTION_NAME: str = Field(default="predictions", env="COLLECTION_NAME")




#     MONGO_INITDB_DATABASE: Optional[str] = Field(None, env="MONGO_INITDB_DATABASE")
#     MONGO_INITDB_ROOT_USERNAME: Optional[str] = Field(None, env="MONGO_INITDB_ROOT_USERNAME")
#     MONGO_INITDB_ROOT_PASSWORD: Optional[str] = Field(None, env="MONGO_INITDB_ROOT_PASSWORD")

#     REDIS_HOST: str = Field(default="redis", env="REDIS_HOST")
#     REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
#     REDIS_PASSWORD: Optional[str] = Field(None, env="REDIS_PASSWORD")

#     GRAFANA_ADMIN_PASSWORD: Optional[str] = Field(None, env="GRAFANA_ADMIN_PASSWORD")

#     # ================================
#     # Model Selection / Paths
#     # ================================
#     SOIL_MODEL_SIZE: str = Field(default="small", env="SOIL_MODEL_SIZE")
#     MODEL_DIR: str = Field(default=MODEL_DIR_DEFAULT, env="MODEL_DIR")
#     DISEASE_MODEL_PATH: str = Field(default=os.path.join(MODEL_DIR_DEFAULT, "plantvillage_efficientnet.tflite"), env="DISEASE_MODEL_PATH")
#     SOIL_MODEL_LARGE_PATH: str = Field(default=os.path.join(MODEL_DIR_DEFAULT, "stacked_model.pkl"), env="SOIL_MODEL_LARGE_PATH")
#     SOIL_MODEL_SMALL_PATH: str = Field(default=os.path.join(MODEL_DIR_DEFAULT, "best_rf.pkl"), env="SOIL_MODEL_SMALL_PATH")
#     SOIL_DATA_PATH: str = Field(default=SOIL_DATA_DEFAULT, env="SOIL_DATA_PATH")

#     DISEASE_DETECTION_MODEL_PATH: Optional[str] = Field(None, env="DISEASE_DETECTION_MODEL_PATH")
#     YIELD_FORECASTING_MODEL_PATH: Optional[str] = Field(None, env="YIELD_FORECASTING_MODEL_PATH")

#     # ================================
#     # External APIs
#     # ================================
#     OPENWEATHER_API_KEY: Optional[str] = Field(None, env="OPENWEATHER_API_KEY")
#     WEATHER_API_URL: Optional[str] = Field(None, env="WEATHER_API_URL")

#     SENTINEL_API_URL: Optional[str] = Field(None, env="SENTINEL_API_URL")
#     SENTINELHUB_CLIENT_ID: Optional[str] = Field(None, env="SENTINELHUB_CLIENT_ID")
#     SENTINELHUB_CLIENT_SECRET: Optional[str] = Field(None, env="SENTINELHUB_CLIENT_SECRET")
#     SENTINELHUB_INSTANCE_ID: Optional[str] = Field(None, env="SENTINELHUB_INSTANCE_ID")

#     FAO_API_URL: Optional[str] = Field(None, env="FAO_API_URL")
#     FAO_API_KEY: Optional[str] = Field(None, env="FAO_API_KEY")
#     MARKET_DATA_URL: Optional[str] = Field(None, env="MARKET_DATA_URL")

#     TWILIO_ACCOUNT_SID: Optional[str] = Field(None, env="TWILIO_ACCOUNT_SID")
#     TWILIO_AUTH_TOKEN: Optional[str] = Field(None, env="TWILIO_AUTH_TOKEN")

#     # ================================
#     # MLflow / Training
#     # ================================
#     MLFLOW_TRACKING_URI: Optional[str] = Field(None, env="MLFLOW_TRACKING_URI")
#     BATCH_SIZE: int = Field(default=32, env="BATCH_SIZE")
#     MAX_WORKERS: int = Field(default=4, env="MAX_WORKERS")
#     CHUNK_SIZE: int = Field(default=100000, env="CHUNK_SIZE")

#     # ================================
#     # App Config
#     # ================================
#     MAX_IMAGE_SIZE: int = Field(default=10485760, env="MAX_IMAGE_SIZE")
#     ALLOWED_IMAGE_TYPES: List[str] = Field(default=["image/jpeg", "image/png", "image/jpg"], env="ALLOWED_IMAGE_TYPES")
#     CORS_ORIGINS: List[str] = Field(default=["http://localhost:3000"], env="CORS_ORIGINS")

#     CSV_FILE_PATH: str = Field(
#     default=os.path.join(os.getcwd(), "predictions_log.csv"),
#     env="CSV_FILE_PATH")

#     SOIL_FEATURES_JSON_PATH: str = Field(
#     default=os.path.join(MODEL_DIR_DEFAULT, "soil_features.json"),
#     env="SOIL_FEATURES_JSON_PATH")


#     @field_validator("ALLOWED_IMAGE_TYPES", "CORS_ORIGINS", mode="before")
#     def parse_list(cls, v: Any) -> List[str]:
#         """Allow list fields to be provided as CSV or JSON array in .env"""
#         if not v:
#             return []
#         if isinstance(v, str):
#             v = v.strip()
#             # JSON array style
#             if (v.startswith("[") and v.endswith("]")):
#                 try:
#                     return json.loads(v)
#                 except json.JSONDecodeError:
#                     raise ValueError(f"Invalid JSON list: {v}")
#             # CSV style
#             return [s.strip() for s in v.split(",") if s.strip()]
#         if isinstance(v, (list, tuple)):
#             return list(v)
#         raise ValueError(f"Unsupported type for list field: {type(v)}")

#     @field_validator("SOIL_MODEL_SIZE", mode="after")
#     def validate_soil_model_size(cls, v: str) -> str:
#         if v not in ["small", "large"]:
#             raise ValueError(f"SOIL_MODEL_SIZE must be either 'small' or 'large', not '{v}'")
#         return v

#     # ================================
#     # Monitoring / Logging
#     # ================================
#     SENTRY_DSN: Optional[str] = Field(None, env="SENTRY_DSN")
#     METRICS_PORT: int = Field(default=8001, env="METRICS_PORT")

#     # ================================
#     # Dynamic helpers
#     # ================================
#     @property
#     def soil_model_path(self) -> str:
#         """Pick soil model based on configured size"""
#         if self.SOIL_MODEL_SIZE == "large":
#             return self.SOIL_MODEL_LARGE_PATH
#         return self.SOIL_MODEL_SMALL_PATH

#     class Config:
#         env_file = ".env"
#         case_sensitive = True
#         extra = "ignore"

# # Create global settings object
# settings = Settings()






































# src/crop_monitor/config/settings.py

import os
import json
from typing import List, Any
from dotenv import load_dotenv

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

# Load .env
load_dotenv()

# ---------------------------------------------------------
# Base directories
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
BASE_PREDICTIONS_DIR = os.getenv(
    "BASE_PREDICTIONS_DIR",
    r"C:\Users\ADEGOKE\Desktop\AI_Crop_Disease_Monitoring\predictions"
)
MODEL_DIR_DEFAULT = os.getenv(
    "MODEL_DIR",
    os.path.join(BASE_DIR, "src", "crop_monitor", "models")
)
SOIL_DATA_DEFAULT = os.getenv(
    "SOIL_DATA_PATH",
    os.path.join(BASE_DIR, "data", "external", "global_soil_data.csv")
)
SOIL_FEATURES_PATH = os.getenv(
    "SOIL_FEATURES_JSON_PATH",
    os.path.join(MODEL_DIR_DEFAULT, "soil_features.json")
)

# ---------------------------------------------------------
# Database defaults
# ---------------------------------------------------------
SQLITE_DB_DEFAULT = os.getenv(
    "SQLITE_DB_PATH",
    os.path.join(BASE_PREDICTIONS_DIR, "local_predictions.db")
)
CSV_FILE_DEFAULT = os.getenv(
    "CSV_FILE_PATH",
    os.path.join(BASE_PREDICTIONS_DIR, "predictions.csv")
)
MONGODB_ATLAS_URI_DEFAULT = os.getenv(
    "MONGODB_ATLAS_URI",
    "mongodb+srv://AI_Crop_Monitor:CShxEb2FOjwU7pEj@cluster0.oqtogab.mongodb.net/AI_Crop_Monitor?retryWrites=true&w=majority"
)
MONGODB_ATLAS_DB_NAME_DEFAULT = os.getenv("MONGODB_ATLAS_DB_NAME", "AI_Crop_Monitor")

# ---------------------------------------------------------
# Settings class - FIXED VERSION
# ---------------------------------------------------------
class Settings(BaseSettings):
    # ----------------------
    # App / Environment
    # ----------------------
    BASE_PREDICTIONS_DIR: str = Field(default=BASE_PREDICTIONS_DIR, env="BASE_PREDICTIONS_DIR")
    APP_ENV: str = Field(default="development", env="APP_ENV")
    APP_HOST: str = Field(default="0.0.0.0", env="APP_HOST")
    APP_PORT: int = Field(default=8000, env="APP_PORT")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    # ----------------------
    # Security / Secrets
    # ----------------------
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    JWT_SECRET_KEY: str = Field(default="dev-jwt-secret-change-in-production", env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    JWT_EXPIRE_MINUTES: int = Field(default=30, env="JWT_EXPIRE_MINUTES")
    ENCRYPTION_KEY: str = Field(default="dev-encryption-key-change-in-production", env="ENCRYPTION_KEY")

    # ----------------------
    # Database
    # ----------------------
    MONGODB_ATLAS_URI: str = Field(default=MONGODB_ATLAS_URI_DEFAULT, env="MONGODB_ATLAS_URI")
    MONGODB_ATLAS_DB_NAME: str = Field(default=MONGODB_ATLAS_DB_NAME_DEFAULT, env="MONGODB_ATLAS_DB_NAME")
    SQLITE_DB_PATH: str = Field(default=SQLITE_DB_DEFAULT, env="SQLITE_DB_PATH")
    COLLECTION_NAME: str = Field(default="predictions", env="COLLECTION_NAME")
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_PASSWORD: str = Field(default="dev-redis-password", env="REDIS_PASSWORD")

    # ----------------------
    # Model Paths - FIXED: Using lowercase to match your main.py
    # ----------------------
    # SOIL_MODEL_PATH: str = Field(default="", env="SOIL_MODEL_PATH")
    MODEL_DIR: str = Field(default=MODEL_DIR_DEFAULT, env="MODEL_DIR")
    SOIL_DATA_PATH: str = Field(default=SOIL_DATA_DEFAULT, env="SOIL_DATA_PATH")
    SOIL_FEATURES_JSON_PATH: str = Field(default=SOIL_FEATURES_PATH, env="SOIL_FEATURES_JSON_PATH")
    SOIL_SERVICE_PATH: str = Field(default="", env="SOIL_SERVICE_PATH")
    SOC_REGRESSION_MODEL_PATH: str = Field(default="", env="SOC_REGRESSION_MODEL_PATH")
    SOIL_CLASSIFICATION_MODEL_PATH: str = Field(default="", env="SOIL_CLASSIFICATION_MODEL_PATH")
    SCALER_CLASSIFICATION_PATH: str = Field(default="", env="SCALER_CLASSIFICATION_PATH")
    SCALER_SOC_PATH: str = Field(default="", env="SCALER_SOC_PATH")
    TEXTURE_LABEL_ENCODER_PATH: str = Field(default="", env="TEXTURE_LABEL_ENCODER_PATH")
    DISEASE_MODEL_PATH: str = Field(default="", env="DISEASE_MODEL_PATH")
    DISEASE_JSON_PATH: str = Field(default="", env="DISEASE_JSON_PATH")
    DISEASE_DETECTION_MODEL_PATH: str = Field(default="", env="DISEASE_DETECTION_MODEL_PATH")
    YIELD_FORECASTING_MODEL_PATH: str = Field(default="", env="YIELD_FORECASTING_MODEL_PATH")


        # ----------------------
    # Soil Service Model Paths - ADD THESE
    # ----------------------
    SOIL_CLASSIFICATION_MODEL_PATH: str = Field(
        default=os.path.join(MODEL_DIR_DEFAULT, "soil", "soil_classification_model.joblib"), 
        env="SOIL_CLASSIFICATION_MODEL_PATH"
    )
    SOC_REGRESSION_MODEL_PATH: str = Field(
        default=os.path.join(MODEL_DIR_DEFAULT, "soil", "soc_regression_model.joblib"), 
        env="SOC_REGRESSION_MODEL_PATH"
    )
    SCALER_CLASSIFICATION_PATH: str = Field(
        default=os.path.join(MODEL_DIR_DEFAULT, "soil", "scaler_classification.joblib"), 
        env="SCALER_CLASSIFICATION_PATH"
    )
    SCALER_SOC_PATH: str = Field(
        default=os.path.join(MODEL_DIR_DEFAULT, "soil", "scaler_soc.joblib"), 
        env="SCALER_SOC_PATH"
    )
    TEXTURE_LABEL_ENCODER_PATH: str = Field(
        default=os.path.join(MODEL_DIR_DEFAULT, "soil", "texture_label_encoder.joblib"), 
        env="TEXTURE_LABEL_ENCODER_PATH"
    )

    # ----------------------
    # External APIs
    # ----------------------
    OPENWEATHER_API_KEY: str = Field(default="", env="OPENWEATHER_API_KEY")
    WEATHER_API_URL: str = Field(default="https://api.openweathermap.org/data/2.5/weather", env="WEATHER_API_URL")
    SENTINEL_API_URL: str = Field(default="https://services.sentinel-hub.com/api/v1", env="SENTINEL_API_URL")
    SENTINELHUB_CLIENT_ID: str = Field(default="", env="SENTINELHUB_CLIENT_ID")
    SENTINELHUB_CLIENT_SECRET: str = Field(default="", env="SENTINELHUB_CLIENT_SECRET")
    SENTINELHUB_INSTANCE_ID: str = Field(default="", env="SENTINELHUB_INSTANCE_ID")
    FAO_API_URL: str = Field(default="https://fenixservices.fao.org/faostat/api/v1/en/data/QCL", env="FAO_API_URL")
    FAO_API_KEY: str = Field(default="", env="FAO_API_KEY")
    MARKET_DATA_URL: str = Field(default="https://fenixservices.fao.org/faostat/api/v1/en/data/PP", env="MARKET_DATA_URL")
    SOILGRIDS_API_URL: str = Field(default="https://rest.isric.org/soilgrids/v2.0/properties/query", env="SOILGRIDS_API_URL")

    # ----------------------
    # SMTP / Email
    # ----------------------
    SMTP_SERVER: str = Field(default="smtp.gmail.com", env="SMTP_SERVER")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_USER: str = Field(default="", env="SMTP_USER")
    SMTP_PASSWORD: str = Field(default="", env="SMTP_PASSWORD")
    SMTP_FROM: str = Field(default="", env="SMTP_FROM")

    # ----------------------
    # Twilio / SMS
    # ----------------------
    TWILIO_ACCOUNT_SID: str = Field(default="", env="TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN: str = Field(default="", env="TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER: str = Field(default="", env="TWILIO_PHONE_NUMBER")

    # ----------------------
    # MLflow / Training
    # ----------------------
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    BATCH_SIZE: int = Field(default=32, env="BATCH_SIZE")
    MAX_WORKERS: int = Field(default=4, env="MAX_WORKERS")
    CHUNK_SIZE: int = Field(default=100000, env="CHUNK_SIZE")

    # ----------------------
    # App Config
    # ----------------------
    MAX_IMAGE_SIZE: int = Field(default=10485760, env="MAX_IMAGE_SIZE")
    ALLOWED_IMAGE_TYPES: List[str] = Field(default_factory=lambda: ["image/jpeg", "image/png", "image/jpg"], env="ALLOWED_IMAGE_TYPES")
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["http://localhost:3000"], env="CORS_ORIGINS")
    CSV_FILE_PATH: str = Field(default=CSV_FILE_DEFAULT, env="CSV_FILE_PATH")

    # ----------------------
    # Monitoring / Logging
    # ----------------------
    SENTRY_DSN: str = Field(default="", env="SENTRY_DSN")
    METRICS_PORT: int = Field(default=8001, env="METRICS_PORT")

    # ----------------------
    # CRS / Geospatial
    # ----------------------
    NIGERIA_CRS_GLOBAL: str = Field(default="EPSG:4326", env="NIGERIA_CRS_GLOBAL")
    NIGERIA_CRS_PROJECTED: str = Field(default="EPSG:32632", env="NIGERIA_CRS_PROJECTED")

    # ----------------------
    # Target Soil Properties
    # ----------------------
    TARGET_SOIL_PROPERTIES: List[str] = Field(default_factory=lambda: ["SOC", "Clay", "Silt", "Sand", "DB"], env="TARGET_SOIL_PROPERTIES")

    # ----------------------
    # Grafana (from your .env)
    # ----------------------
    GRAFANA_ADMIN_PASSWORD: str = Field(default="YourStrongGrafanaPasswordHere", env="GRAFANA_ADMIN_PASSWORD")

    # ----------------------
    # Validators
    # ----------------------
    @model_validator(mode="before")
    def parse_list(cls, values: dict) -> dict:
        """
        Converts string lists from env to Python lists for specified fields.
        """
        for key in ["ALLOWED_IMAGE_TYPES", "CORS_ORIGINS", "TARGET_SOIL_PROPERTIES"]:
            v = values.get(key)
            if not v:
                # Use default from field if nothing provided
                continue
            elif isinstance(v, str):
                v_strip = v.strip()
                if v_strip.startswith("[") and v_strip.endswith("]"):
                    values[key] = json.loads(v_strip)
                else:
                    values[key] = [s.strip() for s in v_strip.split(",") if s.strip()]
            elif isinstance(v, (list, tuple)):
                values[key] = list(v)
            else:
                raise ValueError(f"Unsupported type for list field {key}: {type(v)}")
        return values

    class Config:
        env_file = ".env"
        case_sensitive = False  # ← FIXED: Allow case-insensitive access
        extra = "ignore"

# ---------------------------------------------------------
# Global settings object
# ---------------------------------------------------------
settings = Settings()
