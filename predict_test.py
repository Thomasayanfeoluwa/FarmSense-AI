import os
import requests

# Folder containing images
image_folder = r"C:\Users\ADEGOKE\Desktop\AI_Crop_Disease_Monitoring\src\crop_monitor\image_to_predict"

# Your API endpoint
url = "http://localhost:8000/predict"

# Latitude and longitude to use (replace with your values)
lat = 6.465422
lon = 3.406448

# Loop through all images in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        file_path = os.path.join(image_folder, filename)
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "multipart/form-data")}
            data = {"lat": str(lat), "lon": str(lon)}
            response = requests.post(url, files=files, data=data)
        print(f"File: {filename} => Response: {response.json()}")


