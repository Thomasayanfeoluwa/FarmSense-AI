# full_sync.py
import sqlite3
import os
import time
import random
from pymongo import MongoClient

# ✅ Import settings
from src.crop_monitor.config.settings import settings

# =========================
# CONFIGURATION
# =========================
DB_PATH = settings.SQLITE_DB_PATH
MONGO_URI = settings.MONGODB_ATLAS_URI
DB_NAME = settings.MONGODB_ATLAS_DB_NAME
COLLECTION_NAME = settings.COLLECTION_NAME

# =========================
# HELPER FUNCTIONS
# =========================

def ensure_db():
    """Make sure the local SQLite DB and table exist with correct schema"""
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:  # only create if folder path is not empty
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        disease TEXT,
        confidence REAL,
        treatment TEXT,
        soil_data TEXT,
        satellite_data TEXT,
        weather_data TEXT,
        synced INTEGER DEFAULT 0  -- ✅ new column
    )
    """)
    conn.commit()
    conn.close()




def fetch_recent_predictions(limit=5):
    """Fetch recent predictions from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def sync_local_to_mongo():
    """Sync unsynced predictions to MongoDB Atlas."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE synced=0")
    rows = cursor.fetchall()

    if not rows:
        print("[INFO] No unsynced predictions to sync.")
        conn.close()
        return

    for row in rows:
        data = {
            "id": row[0],
            "prediction": row[1],
            "confidence": row[2],
            "timestamp": row[3],
            "satellite_data": row[4]
        }
        collection.insert_one(data)
        cursor.execute("UPDATE predictions SET synced=1 WHERE id=?", (row[0],))

    conn.commit()
    conn.close()
    print(f"[INFO] Synced {len(rows)} predictions to MongoDB Atlas.")


def retry_sync(attempts=5):
    """Retry syncing with exponential backoff."""
    for i in range(attempts):
        try:
            sync_local_to_mongo()
            print("[INFO] Sync succeeded!")
            return
        except Exception as e:
            wait = (2 ** i) + random.random()
            print(f"[WARN] Attempt {i+1} failed: {e}. Retrying in {wait:.2f} seconds...")
            time.sleep(wait)
    print("[ERROR] All sync attempts failed.")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("=== INITIALIZING DATABASE ===")
    ensure_db()

    print("\n=== RECENT OFFLINE PREDICTIONS ===")
    recent = fetch_recent_predictions()
    if recent:
        for row in recent:
            print(row)
    else:
        print("[INFO] No predictions found in local DB.")

    print("\n=== STARTING SYNC TO MONGODB ATLAS ===")
    retry_sync()
    print("\n=== DONE ===")
