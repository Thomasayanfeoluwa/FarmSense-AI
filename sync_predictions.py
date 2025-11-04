# sync_predictions.py
import sqlite3
from pymongo import MongoClient
import time
import random

# =========================
# CONFIGURATION
# =========================
DB_PATH = r"C:\Users\ADEGOKE\Desktop\AI_Crop_Disease_Monitoring\src\crop_monitor\db\local_predictions.db"
MONGO_URI = "YOUR_MONGODB_ATLAS_URI"  # replace with your Atlas URI
DB_NAME = "crop_monitor"
COLLECTION_NAME = "predictions"

# =========================
# FUNCTIONS
# =========================
def fetch_recent_predictions(limit=5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def sync_local_to_mongo():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE synced=0")
    rows = cursor.fetchall()

    if not rows:
        print("No unsynced predictions found.")
        conn.close()
        return

    for row in rows:
        data = {
            "id": row[0],
            "prediction": row[1],
            "confidence": row[2],
            "timestamp": row[3],
        }
        collection.insert_one(data)
        cursor.execute("UPDATE predictions SET synced=1 WHERE id=?", (row[0],))
    
    conn.commit()
    conn.close()
    print(f"Synced {len(rows)} predictions to MongoDB Atlas.")

def retry_sync(attempts=5):
    for i in range(attempts):
        try:
            sync_local_to_mongo()
            print("Sync succeeded!")
            return
        except Exception as e:
            wait = (2 ** i) + random.random()
            print(f"Attempt {i+1} failed: {e}. Retrying in {wait:.2f} seconds...")
            time.sleep(wait)
    print("All sync attempts failed.")

# =========================
# MAIN SCRIPT
# =========================
if __name__ == "__main__":
    print("=== Recent Offline Predictions ===")
    recent = fetch_recent_predictions()
    if recent:
        for row in recent:
            print(row)
    else:
        print("No predictions found in local DB.")

    print("\n=== Starting Sync to MongoDB Atlas ===")
    retry_sync()
    print("\n=== Done ===")
