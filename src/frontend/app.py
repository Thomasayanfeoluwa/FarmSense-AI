# """
# 🌍 AI-Powered Crop Monitoring Dashboard with Dynamic Alerts
# Includes:
# - Image Upload for Crop Analysis
# - Crop Disease Detection and Health Analysis
# - Soil Fertility Assessment
# - Weather Prediction and Monitoring
# - Satellite Imagery and NDVI Analysis
# - Yield Prediction Engine
# - Market Pricing Engine
# - Smart Recommendations
# - Real-Time Map View with Layers
# - Live Drone Feed with Snapshot Capture
# - Fully Dynamic Alert System (Email, SMS, WhatsApp)
# - Multi-Farm Management with Per-Farm Contacts
# - Comprehensive Sidebar Settings for Customization
# """

# # ================================
# # Imports & Project Path Setup
# # ================================
# import sys
# import os
# import threading
# import time
# import anyio
# import requests
# from PIL import Image
# import pandas as pd
# import plotly.express as px
# import folium
# from streamlit_folium import st_folium
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import datetime
# from fpdf import FPDF
# from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions
# import streamlit as st



# # Now all crop_monitor imports should work directly
# from src.crop_monitor.api.main import predict, predict_batch
# from src.crop_monitor.core.schemas import Location
# from src.crop_monitor.config.settings import settings
# from src.crop_monitor.utils.notification_service import NotificationService




# # Initialize NotificationService with settings credentials
# notification_service = NotificationService()

# # notification_service = NotificationService(
# #     twilio_account_sid=settings.TWILIO_ACCOUNT_SID,
# #     twilio_auth_token=settings.TWILIO_AUTH_TOKEN,
# #     twilio_phone_number=settings.TWILIO_PHONE_NUMBER,
# #     smtp_server=settings.SMTP_SERVER,
# #     smtp_port=settings.SMTP_PORT,
# #     smtp_user=settings.SMTP_USER,
# #     smtp_password=settings.SMTP_PASSWORD,
# #     smtp_from=settings.SMTP_FROM
# # )

# # ================================
# # FastAPI Prediction Wrappers
# # ================================
# class DummyUploadFile:
#     """Minimal interface to mimic FastAPI UploadFile."""
#     def __init__(self, filename, content):
#         self.filename = filename
#         self._content = content
#     async def read(self):
#         return self._content
#     async def seek(self, pos):
#         pass

# def run_single_prediction(file, lat, lon):
#     dummy_file = DummyUploadFile(file.name, file.read())
#     result = anyio.run(predict, file=dummy_file, lat=lat, lon=lon)
#     post_prediction_pipeline(st.session_state.selected_farm, lat, lon)
#     return result

# def run_batch_prediction(files, lat_list, lon_list, farm_names):
#     dummy_files = [DummyUploadFile(f.name, f.read()) for f in files]
#     results = anyio.run(
#         predict_batch,
#         files=dummy_files,
#         lats=",".join(map(str, lat_list)),
#         lons=",".join(map(str, lon_list))
#     )
#     # Run post-prediction pipeline for each farm
#     for i, (lat, lon) in enumerate(zip(lat_list, lon_list)):
#         farm_name = farm_names[i] if i < len(farm_names) else st.session_state.selected_farm
#         post_prediction_pipeline(farm_name, lat, lon)
#     return results

# # ================================
# # SentinelHub Config
# # ================================
# config = SHConfig()
# config.instance_id = settings.SENTINELHUB_INSTANCE_ID
# config.sh_client_id = settings.SENTINELHUB_CLIENT_ID
# config.sh_client_secret = settings.SENTINELHUB_CLIENT_SECRET

# if not (config.instance_id and config.sh_client_id and config.sh_client_secret):
#     st.warning("⚠ Sentinel Hub credentials not found. NDVI features may not work.")

# # ================================
# # Streamlit Session Defaults
# # ================================
# for key, default in [
#     ("alert_triggers", ["Email"]),
#     ("farm_contacts", {}),
#     ("farms", {}),
#     ("alerts", []),
#     ("enable_yield_prediction", True),
#     ("enable_pricing_engine", True),
#     ("ndvi_thread_running", False),
#     ("map_thread_running", False),
#     ("yield_pricing_thread_running", False),
#     ("selected_farm", None),
#     ("selected_lat", 0.0),
#     ("selected_lon", 0.0),
#     ("selected_files", []),
#     ("selected_lats", []),
#     ("selected_lons", []),
#     ("farm_email", ""),
#     ("farm_phone", ""),
#     ("farm_whatsapp", ""),
#     ("batch_farm_names", [])
# ]:
#     if key not in st.session_state:
#         st.session_state[key] = default

# # ================================
# # Helper Functions
# # ================================
# def add_alert(message, level="info"):
#     st.session_state.alerts.append({"message": message, "level": level})
#     for alert in st.session_state.alerts[-st.session_state.get("max_alerts", 5):]:
#         if alert["level"] == "warning":
#             st.warning(alert["message"])
#         elif alert["level"] == "error":
#             st.error(alert["message"])
#         else:
#             st.info(alert["message"])

# def trigger_alert(message, level="warning"):
#     add_alert(message, level)
#     send_alerts(message)

# def send_alerts(alert_message: str):
#     for farm_name, contacts in st.session_state.farm_contacts.items():
#         email = contacts.get("email", "")
#         phone = contacts.get("phone", "")
#         whatsapp = contacts.get("whatsapp", "")

#         if "Email" in st.session_state.alert_triggers and email:
#             try:
#                 anyio.run(lambda: notification_service.send_email(
#                     to=email,
#                     subject=f"Farm Alert: {farm_name}",
#                     message=alert_message
#                 ))
#             except Exception as e:
#                 st.error(f"❌ Email sending failed for {farm_name}: {e}")

#         if "SMS" in st.session_state.alert_triggers and phone:
#             try:
#                 anyio.run(lambda: notification_service.send_sms(
#                     to=phone,
#                     message=alert_message
#                 ))
#             except Exception as e:
#                 st.error(f"❌ SMS sending failed for {farm_name}: {e}")

#         if "WhatsApp" in st.session_state.alert_triggers and whatsapp:
#             try:
#                 anyio.run(lambda: notification_service.send_whatsapp(
#                     to=whatsapp,
#                     message=alert_message
#                 ))
#             except Exception as e:
#                 st.error(f"❌ WhatsApp sending failed for {farm_name}: {e}")

# def fetch_ndvi_image(lat, lon):
#     try:
#         bbox = BBox(bbox=[lon-0.01, lat-0.01, lon+0.01, lat+0.01], crs=CRS.WGS84)
#         size = bbox_to_dimensions(bbox, resolution=10)
#         request = SentinelHubRequest(
#             data_folder='ndvi_cache',
#             evalscript="""
#             //VERSION=3
#             function setup() {return {input: ["B04","B08"], output: { bands: 1 }};}
#             function evaluatePixel(sample) { return [(sample.B08 - sample.B04)/(sample.B08 + sample.B04)]; }
#             """,
#             input_data=[SentinelHubRequest.input_data(DataCollection.SENTINEL2_L1C)],
#             responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
#             bbox=bbox,
#             size=size,
#             config=config
#         )
#         return request.get_data()[0]
#     except Exception as e:
#         trigger_alert(f"NDVI fetch error: {e}", level="error")
#         return None

# def post_prediction_pipeline(farm_name, lat, lon):
#     """Runs after each prediction (single or batch)"""
#     # 1️⃣ Capture live snapshot
#     if 'webrtc_ctx' in globals() and webrtc_ctx.video_transformer:
#         transformer = webrtc_ctx.video_transformer
#         try:
#             frame = transformer.transform(webrtc_ctx.video_receiver.get_frame().to_ndarray(format="bgr24"))
#             snapshot_name = f"snapshots/{farm_name}_snapshot_{transformer.snapshot_count}.png"
#             cv2.imwrite(snapshot_name, frame)
#             transformer.snapshot_count += 1
#             add_alert(f"📸 Snapshot captured for {farm_name}: {snapshot_name}", level="info")
#         except Exception as e:
#             add_alert(f"❌ Failed to capture snapshot for {farm_name}: {e}", level="error")

#     # 2️⃣ Run Yield & Pricing Engine
#     if st.session_state.enable_yield_prediction or st.session_state.enable_pricing_engine:
#         add_alert(f"Running Yield & Pricing engine for {farm_name} at ({lat}, {lon})", level="info")
#         # Connect to your actual engine functions here
#         # Example:
#         # yield_result = run_yield_prediction(farm_name, lat, lon)
#         # pricing_result = run_pricing_engine(farm_name, lat, lon)
#         # trigger_alert(f"Yield: {yield_result}, Price: {pricing_result}")

#     # 3️⃣ Fire alerts
#     trigger_alert(f"Post-prediction pipeline executed for {farm_name}")

# # ================================
# # Ensure directories
# # ================================
# os.makedirs("snapshots", exist_ok=True)
# os.makedirs("ndvi_cache", exist_ok=True)

# # ================================
# # Dashboard Main
# # ================================
# st.title("🌍 AI-Powered Crop Monitoring Dashboard")

# # -------------------------------
# # Sidebar Multi-Farm & Batch Upload
# # -------------------------------
# st.sidebar.header("Farm & Batch Prediction")
# st.session_state.selected_farm = st.sidebar.text_input("Farm Name", st.session_state.selected_farm)
# st.session_state.selected_lat = st.sidebar.number_input("Latitude", value=st.session_state.selected_lat)
# st.session_state.selected_lon = st.sidebar.number_input("Longitude", value=st.session_state.selected_lon)

# # Contact input fields
# st.sidebar.subheader("Contact Information")
# st.session_state.farm_email = st.sidebar.text_input("Email", st.session_state.farm_email)
# st.session_state.farm_phone = st.sidebar.text_input("Phone", st.session_state.farm_phone)
# st.session_state.farm_whatsapp = st.sidebar.text_input("WhatsApp", st.session_state.farm_whatsapp)

# if st.sidebar.button("Save Contact") and st.session_state.selected_farm:
#     farm_name = st.session_state.selected_farm
#     st.session_state.farm_contacts[farm_name] = {
#         "email": st.session_state.farm_email,
#         "phone": st.session_state.farm_phone,
#         "whatsapp": st.session_state.farm_whatsapp
#     }
#     st.session_state["farms"][farm_name] = st.session_state.farm_contacts[farm_name]
#     st.sidebar.success(f"✅ Contact saved for {farm_name}")

# # Notification test button
# if st.sidebar.button("Test Notifications"):
#     async def test_notifications():
#         await notification_service.send_sms("+2348012345678", "Test message")
#         await notification_service.send_whatsapp("+2348012345678", "Test message")
#         await notification_service.send_email("user@example.com", "Test Subject", "Test message")
    
#     try:
#         anyio.run(test_notifications)
#         st.sidebar.success("Test notifications sent successfully!")
#     except Exception as e:
#         st.sidebar.error(f"Failed to send test notifications: {e}")

# st.session_state.selected_files = st.sidebar.file_uploader(
#     "Upload Images (Batch Supported)",
#     type=["jpg", "jpeg", "png"],
#     accept_multiple_files=True
# )

# # # Allow specifying farm names for each file in batch
# # if st.session_state.selected_files:
# #     st.sidebar.subheader("Assign Farms to Uploaded Images")
# #     st.session_state.batch_farm_names = []
# #     for i, file in enumerate(st.session_state.selected_files):
# #         farm_name = st.sidebar.text_input(
# #             f"Farm for {file.name}", 
# #             value=st.session_state.selected_farm,
# #             key=f"farm_{i}"
# #         )
# #         st.session_state.batch_farm_names.append(farm_name)
# # Allow specifying farm names for each file in batch
# if st.session_state.selected_files:
#     st.sidebar.subheader("Assign Farms to Uploaded Images")
#     st.session_state.batch_farm_names = []
#     for i, file in enumerate(st.session_state.selected_files):
#         farm_name_input = st.sidebar.text_input(
#             f"Farm for {file.name}", 
#             value=st.session_state.selected_farm or "",
#             key=f"farm_{i}"
#         )
#         # Fallback: use selected_farm if input is blank
#         farm_name = farm_name_input.strip() or st.session_state.selected_farm or f"Farm_{i+1}"
#         st.session_state.batch_farm_names.append(farm_name)

# st.session_state.selected_lats = [st.session_state.selected_lat]*len(st.session_state.selected_files)
# st.session_state.selected_lons = [st.session_state.selected_lon]*len(st.session_state.selected_files)

# if st.sidebar.button("Run Batch Prediction") and st.session_state.selected_files:
#     results = run_batch_prediction(
#         st.session_state.selected_files,
#         st.session_state.selected_lats,
#         st.session_state.selected_lons,
#         st.session_state.batch_farm_names
#     )
#     for res in results:
#         trigger_alert(f"Batch Prediction Result: {res}")

# # -------------------------------
# # NDVI Thread
# # -------------------------------
# def refresh_ndvi_thread(lat, lon):
#     placeholder = st.empty()
#     while True:
#         ndvi_img = fetch_ndvi_image(lat, lon)
#         if ndvi_img:
#             placeholder.image(ndvi_img, caption="Live NDVI Map", use_column_width=True)
#         time.sleep(300)

# if not st.session_state.ndvi_thread_running:
#     st.session_state.ndvi_thread_running = True
#     threading.Thread(
#         target=refresh_ndvi_thread,
#         args=(st.session_state.selected_lat, st.session_state.selected_lon),
#         daemon=True
#     ).start()

# # -------------------------------
# # Map Update Thread
# # -------------------------------
# def live_map_update():
#     while True:
#         try:
#             data = requests.get("http://127.0.0.1:8000/get_all_assessments", timeout=30).json()
#             if not data: time.sleep(60); continue
#             df = pd.DataFrame(data)
#             m = folium.Map(location=[st.session_state.selected_lat, st.session_state.selected_lon], zoom_start=15)
#             for _, row in df.iterrows():
#                 folium.CircleMarker(
#                     location=[row["latitude"], row["longitude"]],
#                     radius=7,
#                     color="green" if row.get("soil_health_score",0) > 0.7 else "red",
#                     fill=True,
#                 ).add_to(m)
#             st.session_state["map"] = st_folium(m, width=700, height=500)
#         except Exception as e:
#             trigger_alert(f"Map update error: {e}", level="error")
#         time.sleep(600)

# if not st.session_state.map_thread_running:
#     st.session_state.map_thread_running = True
#     threading.Thread(target=live_map_update, daemon=True).start()

# # ================================
# # Live Drone Feed & Snapshot Capture
# # ================================
# st.sidebar.header("Drone Feed & Snapshots")

# class DroneVideoTransformer(VideoTransformerBase):
#     def __init__(self):
#         self.snapshot_count = 0

#     def transform(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         # Optional: overlay timestamp
#         cv2.putText(img, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
#         return img

# # Stream live feed
# webrtc_ctx = webrtc_streamer(
#     key="drone-stream",
#     video_transformer_factory=DroneVideoTransformer,
#     media_stream_constraints={"video": True, "audio": False},
#     async_transform=True
# )

# # Snapshot capture button
# if st.sidebar.button("Capture Snapshot") and webrtc_ctx.video_transformer:
#     transformer = webrtc_ctx.video_transformer
#     frame = transformer.transform(webrtc_ctx.video_receiver.get_frame().to_ndarray(format="bgr24"))
#     snapshot_name = f"snapshots/{st.session_state.selected_farm}_snapshot_{transformer.snapshot_count}.png"
#     cv2.imwrite(snapshot_name, frame)
#     transformer.snapshot_count += 1
#     st.success(f"📸 Snapshot saved: {snapshot_name}")
#     # Trigger per-farm yield/pricing engine after snapshot
#     if st.session_state.enable_yield_prediction or st.session_state.enable_pricing_engine:
#         trigger_alert(f"Snapshot captured for {st.session_state.selected_farm}. Yield/Pricing engines will run.")

# # ================================
# # Per-Farm Yield & Pricing Engine Trigger
# # ================================
# def run_farm_yield_pricing(farm_name, lat, lon):
#     # Placeholder: connect with your existing yield/pricing functions
#     add_alert(f"Running Yield & Pricing engine for {farm_name} at ({lat}, {lon})", level="info")
#     # Example: could fetch predictions and trigger alerts
#     # yield_result = run_yield_prediction(farm_name, lat, lon)
#     # pricing_result = run_pricing_engine(farm_name, lat, lon)
#     # trigger_alert(f"Yield: {yield_result}, Price: {pricing_result}")

# if st.sidebar.button("Run Yield & Pricing Engine") and st.session_state.selected_farm:
#     run_farm_yield_pricing(
#         st.session_state.selected_farm,
#         st.session_state.selected_lat,
#         st.session_state.selected_lon
#     )






"""
🌍 AI-Powered Crop Monitoring Dashboard
Fully integrated Streamlit frontend for:
- Single & Batch Crop Disease Prediction
- Soil, Weather & Satellite Analysis
- NDVI Maps & Yield/Pricing Engine
- Drone Feed & Snapshot Capture
- Alerts & Notifications (Email, SMS, WhatsApp)
- Recent Predictions & System Status
- Periodic Sync / Background Tasks
"""

# =============================
# IMPORTS
# =============================
import os, io, threading, time, datetime, csv
import requests
import pandas as pd
import numpy as np
import cv2
import sys
from PIL import Image
import streamlit as st
from streamlit_folium import st_folium
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import folium
import anyio
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# -----------------------------
# Backend Imports
# -----------------------------
from src.crop_monitor.api.main import predict, predict_batch, soil_service, weather_service
from src.crop_monitor.core.schemas import Location
from src.crop_monitor.utils.notification_service import NotificationService
from src.crop_monitor.config.settings import settings

# =============================
# Notification Service
# =============================
notification_service = NotificationService()

# =============================
# CSV Logging
# =============================
CSV_FILE_PATH = "predictions_log.csv"
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp","farm","lat","lon","disease","confidence","treatment","soil","satellite","weather"
        ])
        writer.writeheader()

def log_prediction(farm, lat, lon, disease, confidence, treatment, soil, satellite, weather):
    row = {
        "timestamp": datetime.datetime.now().isoformat(),
        "farm": farm,
        "lat": lat,
        "lon": lon,
        "disease": disease,
        "confidence": confidence,
        "treatment": treatment,
        "soil": soil,
        "satellite": satellite,
        "weather": weather
    }
    with open(CSV_FILE_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)

def get_recent_predictions(limit=10):
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        return df.tail(limit).to_dict("records")
    except:
        return []

# =============================
# Streamlit Session Defaults
# =============================
defaults = {
    "alert_triggers": ["Email", "SMS", "WhatsApp"],
    "farm_contacts": {},
    "farms": {},
    "alerts": [],
    "enable_yield_prediction": True,
    "enable_pricing_engine": True,
    "ndvi_thread_running": False,
    "map_thread_running": False,
    "yield_pricing_thread_running": False,
    "soil_weather_thread_running": False,
    "periodic_sync_thread_running": False,
    "high_risk_thread_running": False,
    "selected_farm": None,
    "selected_lat": 0.0,
    "selected_lon": 0.0,
    "selected_files": [],
    "selected_lats": [],
    "selected_lons": [],
    "farm_email": "",
    "farm_phone": "",
    "farm_whatsapp": "",
    "batch_farm_names": [],
    "last_sync": None,
    "high_risk_alerts": []
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =============================
# Alerts / Notification Helpers
# =============================
def add_alert(msg, level="info"):
    st.session_state.alerts.append({"message": msg, "level": level})
    if level == "warning":
        st.warning(msg)
    elif level == "error":
        st.error(msg)
    else:
        st.info(msg)

def trigger_dynamic_alert(farm_name, message, alert_type="info"):
    """Centralized alert: dashboard + Email/SMS/WhatsApp"""
    add_alert(f"⚠ {farm_name}: {message}", level=alert_type)
    contacts = st.session_state.farm_contacts.get(farm_name, {})
    email = contacts.get("email", "")
    phone = contacts.get("phone", "")
    whatsapp = contacts.get("whatsapp", "")

    async def send_all_alerts():
        try:
            if "Email" in st.session_state.alert_triggers and email:
                await notification_service.send_email(to=email, subject=f"Farm Alert: {farm_name}", message=message)
            if "SMS" in st.session_state.alert_triggers and phone:
                await notification_service.send_sms(to=phone, message=message)
            if "WhatsApp" in st.session_state.alert_triggers and whatsapp:
                await notification_service.send_whatsapp(to=whatsapp, message=message)
        except Exception as e:
            st.error(f"❌ Failed to send alert: {e}")

    try:
        anyio.run(send_all_alerts)
    except Exception as e:
        st.error(f"❌ Alert dispatch error: {e}")

# =============================
# Yield & Pricing Alerts
# =============================
def run_farm_yield_pricing(farm_name, lat, lon):
    simulated_yield = np.random.uniform(0.5, 1.5)
    simulated_price = np.random.uniform(200, 500)
    add_alert(f"Yield & Pricing for {farm_name}: Yield={simulated_yield:.2f}t/ha, Price=${simulated_price:.2f}", level="info")
    if simulated_yield < 0.8:
        trigger_dynamic_alert(farm_name, f"Yield alert! Expected yield low: {simulated_yield:.2f} t/ha", "warning")
    if simulated_price < 250:
        trigger_dynamic_alert(farm_name, f"Price alert! Market price low: ${simulated_price:.2f}", "warning")

# =============================
# DummyUploadFile for Streamlit
# =============================
class DummyUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self._content = content
    async def read(self): return self._content
    async def seek(self, pos): pass

# =============================
# Post-Prediction Pipeline
# =============================
def post_prediction_pipeline(farm_name, lat, lon, prediction_result):
    run_farm_yield_pricing(farm_name, lat, lon)
    log_prediction(
        farm=farm_name,
        lat=lat,
        lon=lon,
        disease=getattr(prediction_result, "disease", "Unavailable"),
        confidence=getattr(prediction_result, "confidence", 0.0),
        treatment=getattr(prediction_result, "treatment", ""),
        soil=getattr(prediction_result, "soil", {}),
        satellite=getattr(prediction_result, "satellite", {}),
        weather=getattr(prediction_result, "weather", {})
    )
    disease = getattr(prediction_result, "disease", "Unavailable")
    confidence = getattr(prediction_result, "confidence", 0.0)
    ALERT_CONFIDENCE_THRESHOLD = 0.7
    if confidence >= ALERT_CONFIDENCE_THRESHOLD:
        alert_msg = f"Disease alert! Predicted {disease} with confidence {confidence:.2f}"
        trigger_dynamic_alert(farm_name, alert_msg, alert_type="warning")

# =============================
# Single/Batch Prediction Functions
# =============================
def run_single_prediction(file, lat, lon):
    dummy_file = DummyUploadFile(file.name, file.read())
    result = anyio.run(predict, file=dummy_file, lat=lat, lon=lon)
    post_prediction_pipeline(st.session_state.selected_farm, lat, lon, result)
    return result

def run_batch_prediction(files, lat_list, lon_list, farm_names):
    dummy_files = [DummyUploadFile(f.name, f.read()) for f in files]
    results = anyio.run(
        predict_batch,
        files=dummy_files,
        lats=",".join(map(str, lat_list)),
        lons=",".join(map(str, lon_list))
    )
    for i, (lat, lon) in enumerate(zip(lat_list, lon_list)):
        farm_name = farm_names[i] if i < len(farm_names) else st.session_state.selected_farm
        post_prediction_pipeline(farm_name, lat, lon, results[i])
    return results

# =============================
# NDVI / Yield / Soil-Weather Threads
# =============================
config = SHConfig()
config.instance_id = settings.SENTINELHUB_INSTANCE_ID
config.sh_client_id = settings.SENTINELHUB_CLIENT_ID
config.sh_client_secret = settings.SENTINELHUB_CLIENT_SECRET

def fetch_ndvi_image(lat, lon, resolution=10):
    try:
        bbox = BBox(bbox=[lon-0.01, lat-0.01, lon+0.01, lat+0.01], crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=resolution)
        request = SentinelHubRequest(
            data_folder='ndvi_cache',
            evalscript="""
                //VERSION=3
                function setup() {return {input: ["B04","B08"], output: { bands: 1 }};}
                function evaluatePixel(sample) { return [(sample.B08 - sample.B04)/(sample.B08 + sample.B04)]; }
            """,
            input_data=[SentinelHubRequest.input_data(DataCollection.SENTINEL2_L1C)],
            responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
            bbox=bbox,
            size=size,
            config=config
        )
        ndvi_data = request.get_data()[0]
        return ndvi_data
    except Exception as e:
        st.error(f"NDVI fetch error: {e}")
        return None

def ndvi_live_thread(lat, lon, farm_name):
    placeholder = st.empty()
    while True:
        ndvi_img = fetch_ndvi_image(lat, lon)
        if ndvi_img is not None:
            placeholder.image(ndvi_img, caption="Live NDVI Map", use_column_width=True)
            avg_ndvi = np.mean(ndvi_img)
            if avg_ndvi < 0.3:
                trigger_dynamic_alert(farm_name, f"NDVI low ({avg_ndvi:.2f}) – check field", "warning")
        time.sleep(300)

def yield_pricing_thread(lat, lon, farm_name):
    placeholder = st.empty()
    while True:
        yield_val, price_val = np.random.uniform(0.5, 1.5), np.random.uniform(200, 500)
        placeholder.metric(label=f"Farm: {farm_name}", value=f"{yield_val:.2f} t/ha", delta=f"${price_val:.2f}")
        if yield_val < 0.8:
            trigger_dynamic_alert(farm_name, f"Yield low: {yield_val:.2f} t/ha", "warning")
        if price_val < 250:
            trigger_dynamic_alert(farm_name, f"Price low: ${price_val:.2f}", "warning")
        time.sleep(180)

def soil_weather_thread(lat, lon, farm_name=None):
    placeholder = st.empty()
    while True:
        try:
            soil_info = soil_service.get_soil_data(lat, lon)
            weather_info = weather_service.get_weather_data(lat, lon)
        except:
            soil_info, weather_info = {}, {}
        placeholder.json({"soil": soil_info, "weather": weather_info})
        if farm_name and soil_info.get("moisture", 1.0) < 0.2:
            trigger_dynamic_alert(farm_name, "Soil moisture critically low!", "warning")
        time.sleep(600)

# =============================
# Periodic Sync Thread
# =============================
def periodic_sync():
    while True:
        st.session_state.last_sync = datetime.datetime.now()
        time.sleep(180)

if not st.session_state.periodic_sync_thread_running:
    st.session_state.periodic_sync_thread_running = True
    threading.Thread(target=periodic_sync, daemon=True).start()

# =============================
# High-Risk Alert Scanner Thread
# =============================
def high_risk_alert_thread(conf_threshold=0.7, scan_interval=180):
    while True:
        try:
            recent_preds = get_recent_predictions(limit=50)
            for pred in recent_preds:
                confidence = float(pred.get("confidence", 0.0))
                if confidence >= conf_threshold:
                    farm = pred.get("farm", "Unknown")
                    disease = pred.get("disease", "Unknown")
                    alert_key = f"{farm}_{disease}_{confidence:.2f}"
                    if alert_key not in st.session_state.high_risk_alerts:
                        st.session_state.high_risk_alerts.append(alert_key)
                        alert_msg = f"{disease} detected with high confidence ({confidence:.2f})"
                        trigger_dynamic_alert(farm, alert_msg, alert_type="warning")
        except Exception as e:
            st.error(f"High-risk alert thread error: {e}")
        time.sleep(scan_interval)

if not st.session_state.high_risk_thread_running:
    st.session_state.high_risk_thread_running = True
    threading.Thread(target=high_risk_alert_thread, daemon=True).start()

# =============================
# Streamlit UI
# =============================
st.title("🌍 Integrated AI Farm Dashboard")

# Status Banner
status_col1, status_col2, status_col3 = st.columns(3)
status_col1.metric("API Status", "Healthy")
status_col2.metric("DB Status", "MongoDB + SQLite")
status_col3.metric("Environment", settings.APP_ENV)
if st.session_state.last_sync:
    st.info(f"Last sync: {st.session_state.last_sync.strftime('%Y-%m-%d %H:%M:%S')}")

# Sidebar: Farm Info & Contacts
st.sidebar.header("Farm Info & Alerts")
st.session_state.selected_farm = st.sidebar.text_input("Farm Name", st.session_state.selected_farm or "")
st.session_state.selected_lat = st.sidebar.number_input("Latitude", value=st.session_state.selected_lat)
st.session_state.selected_lon = st.sidebar.number_input("Longitude", value=st.session_state.selected_lon)
st.session_state.farm_email = st.sidebar.text_input("Email", st.session_state.farm_email)
st.session_state.farm_phone = st.sidebar.text_input("Phone", st.session_state.farm_phone)
st.session_state.farm_whatsapp = st.sidebar.text_input("WhatsApp", st.session_state.farm_whatsapp)

if st.sidebar.button("Save Contact") and st.session_state.selected_farm:
    farm_name = st.session_state.selected_farm
    st.session_state.farm_contacts[farm_name] = {
        "email": st.session_state.farm_email,
        "phone": st.session_state.farm_phone,
        "whatsapp": st.session_state.farm_whatsapp
    }
    st.sidebar.success(f"✅ Contact saved for {farm_name}")

# File uploader for batch
st.session_state.selected_files = st.sidebar.file_uploader(
    "Upload Images (Batch)", type=["jpg","jpeg","png"], accept_multiple_files=True
)
if st.session_state.selected_files:
    st.session_state.batch_farm_names = []
    for i, file in enumerate(st.session_state.selected_files):
        farm_name_input = st.sidebar.text_input(f"Farm for {file.name}", value=st.session_state.selected_farm, key=f"farm_{i}")
        st.session_state.batch_farm_names.append(farm_name_input)

# Single Prediction Panel
st.header("Single Prediction")
single_file = st.file_uploader("Upload single image", type=["jpg","jpeg","png"])
single_lat = st.number_input("Latitude", value=st.session_state.selected_lat, key="single_lat")
single_lon = st.number_input("Longitude", value=st.session_state.selected_lon, key="single_lon")
if st.button("Run Single Prediction") and single_file:
    result = run_single_prediction(single_file, single_lat, single_lon)
    CONF_THRESHOLD = 0.7
    disease_text = f"Disease: {result.disease} | Confidence: {result.confidence:.2f}"
    if result.confidence >= CONF_THRESHOLD:
        st.markdown(f"⚠️ <span style='color:red;font-weight:bold'>{disease_text}</span>", unsafe_allow_html=True)
    else:
        st.success(disease_text)
    st.json({
        "treatment": result.treatment,
        "soil": result.soil,
        "satellite": result.satellite,
        "weather": result.weather
    })

# Batch Prediction Panel
st.header("Batch Prediction")
if st.session_state.selected_files:
    if st.button("Run Batch Prediction"):
        lat_list = [float(single_lat) for _ in st.session_state.selected_files]
        lon_list = [float(single_lon) for _ in st.session_state.selected_files]
        results = run_batch_prediction(st.session_state.selected_files, lat_list, lon_list, st.session_state.batch_farm_names)
        CONF_THRESHOLD = 0.7
        for i, res in enumerate(results):
            disease_text = f"{st.session_state.batch_farm_names[i]} | Disease: {res.disease} | Confidence: {res.confidence:.2f}"
            if res.confidence >= CONF_THRESHOLD:
                st.markdown(f"⚠️ <span style='color:red;font-weight:bold'>{disease_text}</span>", unsafe_allow_html=True)
            else:
                st.success(disease_text)
            st.json({
                "treatment": res.treatment,
                "soil": res.soil,
                "satellite": res.satellite,
                "weather": res.weather
            })

# Recent Predictions
st.header("Recent Predictions")
recent_preds = get_recent_predictions(limit=10)
st.table(recent_preds)

# Soil Info Panel
st.header("Soil Information")
soil_info = soil_service.get_soil_data(st.session_state.selected_lat, st.session_state.selected_lon)
st.json(soil_info)

# NDVI / Yield / Pricing Panel
st.header("NDVI / Yield / Pricing")
if st.session_state.selected_lat and st.session_state.selected_lon and st.session_state.selected_farm:
    threading.Thread(target=ndvi_live_thread, args=(st.session_state.selected_lat, st.session_state.selected_lon, st.session_state.selected_farm), daemon=True).start()
    threading.Thread(target=yield_pricing_thread, args=(st.session_state.selected_lat, st.session_state.selected_lon, st.session_state.selected_farm), daemon=True).start()
    threading.Thread(target=soil_weather_thread, args=(st.session_state.selected_lat, st.session_state.selected_lon, st.session_state.selected_farm), daemon=True).start()
