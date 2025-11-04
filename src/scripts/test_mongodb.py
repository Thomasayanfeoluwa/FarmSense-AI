import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the connection string from environment variables
connection_string = os.getenv("MONGODB_ATLAS_URI")
db_name = os.getenv("MONGODB_ATLAS_DB_NAME")

if not connection_string:
    raise ValueError("MONGODB_ATLAS_URI not found in .env file")
if not db_name:
    raise ValueError("MONGODB_ATLAS_DB_NAME not found in .env file")

try:
    # Connect to the cluster
    client = MongoClient(connection_string)
    # Send a ping to confirm a successful connection
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB Atlas!")

    # Get the database
    db = client[db_name]
    # List all collections (will be empty for now)
    collections = db.list_collection_names()
    print(f"Collections in database: {collections}")

except Exception as e:
    print(f"An error occurred: {e}")
