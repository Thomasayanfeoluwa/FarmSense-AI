# Crop Monitoring System - Setup Guide

## External Services Setup

### MongoDB Atlas
1. Create account at https://www.mongodb.com/atlas
2. Create a free tier cluster
3. Set up database user with read/write permissions
4. Add network access (0.0.0.0/0 for development)
5. Get connection string and add to .env file

### OpenWeatherMap API
1. Create account at https://openweathermap.org/api
2. Get API key from the dashboard
3. Add API key to .env file

### Copernicus Open Access Hub
1. Create account at https://scihub.copernicus.eu/
2. Note username and password
3. Add credentials to .env file

### MLflow Tracking Server
1. Install MLflow: `pip install mlflow`
2. Start server: `mlflow server --host 0.0.0.0 --port 5000`
3. Verify UI at http://localhost:5000

## Environment Configuration

### Required Environment Variables
- `MONGODB_ATLAS_URI`: MongoDB connection string
- `OPENWEATHER_API_KEY`: OpenWeatherMap API key
- `COPERNICUS_USERNAME`: Copernicus username
- `COPERNICUS_PASSWORD`: Copernicus password
- `MLFLOW_TRACKING_URI`: MLflow server URL

### Optional Environment Variables
- `TWILIO_ACCOUNT_SID`: Twilio account SID (for notifications)
- `TWILIO_AUTH_TOKEN`: Twilio auth token (for notifications)

## Testing the Setup

Run the comprehensive test script:
```bash
python src/scripts/test_complete_setup.py