import os
from src.crop_monitor.config.settings import settings

print("🔹 BASE_PREDICTIONS_DIR:", settings.BASE_PREDICTIONS_DIR)
print("🔹 MODEL_DIR:", settings.MODEL_DIR)
print("🔹 Soil model exists:", os.path.exists(settings.SOIL_MODEL_SMALL_PATH))
print("🔹 SQLite path:", settings.SQLITE_DB_PATH)
print("🔹 SQLite dir exists:", os.path.exists(os.path.dirname(settings.SQLITE_DB_PATH)))
print("🔹 CSV path:", settings.CSV_FILE_PATH)
print("🔹 CSV dir exists:", os.path.exists(os.path.dirname(settings.CSV_FILE_PATH)))
print("🔹 MongoDB URI configured:", settings.MONGODB_ATLAS_URI is not None)

