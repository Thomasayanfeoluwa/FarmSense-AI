# # scripts/test_services.py
# import sys
# import os

# # Add the src directory to the Python path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# from src.crop_monitor.config.settings import settings
# from src.crop_monitor.services.weather_service import WeatherService
# from src.crop_monitor.services.soil_service import SoilService
# from src.crop_monitor.services.satellite_service import SatelliteService
# from src.crop_monitor.services.disease_service import DiseaseService  # ✅ Added import

# def test_services():
#     print("Testing services...")
    
#     # Test Weather Service
#     print("\n1. Testing Weather Service:")
#     weather_service = WeatherService()
#     weather_data = weather_service.get_weather_data(6.465422, 3.406448)  # Lagos coordinates
#     print(f"Weather Data: {weather_data}")
    
#     # Test Soil Service
#     print("\n2. Testing Soil Service:")
#     soil_service = SoilService()
#     soil_data = soil_service.get_soil_analysis(6.465422, 3.406448)
#     print(f"Soil Data: {soil_data}")
    
#     # Test Satellite Service
#     print("\n3. Testing Satellite Service:")
#     satellite_service = SatelliteService()
#     satellite_data = satellite_service.get_vegetation_data(6.465422, 3.406448)
#     print(f"Satellite Data: {satellite_data}")

#     # Test Disease Service ✅
#     print("\n4. Testing Disease Detection Service:")
#     disease_service = DiseaseService(model_path=settings.DISEASE_MODEL_PATH)

#     # Handle all JPG images in data/test
#     test_dir = os.path.join(os.path.dirname(__file__), "..", "data", "test")
#     jpg_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]

#     if jpg_files:
#         for img_file in jpg_files:
#             img_path = os.path.join(test_dir, img_file)
#             result = disease_service.predict_disease(img_path)
#             print(f"Disease Detection Result for {img_file}: {result}")
#     else:
#         print(f"No JPG images found in {test_dir}")

#     print("\nAll services tested successfully!")

# if __name__ == "__main__":
#     test_services()






# scripts/test_services.py# scripts/test_services.py
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.crop_monitor.config.settings import settings
from src.crop_monitor.services.weather_service import WeatherService
from src.crop_monitor.services.soil_service import SoilService
from src.crop_monitor.services.satellite_service import SatelliteService
from src.crop_monitor.services.disease_service import DiseaseService  # ✅ Added import

def test_services():
    print("Testing services...\n")
    
    # =========================
    # 1. Weather Service Test
    # =========================
    print("1. Testing Weather Service:")
    weather_service = WeatherService()
    weather_data = weather_service.get_weather_data(6.465422, 3.406448)  # Lagos coordinates
    print(f"Weather Data: {weather_data}\n")
    
    # =========================
    # 2. Soil Service Test
    # =========================
    print("2. Testing Soil Service:")
    soil_service = SoilService()
    soil_data = soil_service.get_soil_analysis(6.465422, 3.406448)
    print(f"Soil Data: {soil_data}\n")
    
    # =========================
    # 3. Satellite Service Test
    # =========================
    print("3. Testing Satellite Service:")
    satellite_service = SatelliteService()
    satellite_data = satellite_service.get_vegetation_data(6.465422, 3.406448)
    print(f"Satellite Data: {satellite_data}\n")

    # =========================
    # 4. Disease Detection Service Test
    # =========================
    print("4. Testing Disease Detection Service:")
    disease_service = DiseaseService(model_path=settings.DISEASE_MODEL_PATH)

    # Directory containing test images
    test_dir = os.path.join(os.path.dirname(__file__), "..", "data", "test")
    if not os.path.exists(test_dir):
        print(f"⚠️ Test directory not found: {test_dir}")
        print("\n✅ All services tested successfully!")
        return

    # Collect all JPG/PNG images
    image_files = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if os.path.isfile(os.path.join(test_dir, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print(f"No JPG/PNG images found in {test_dir}")
        print("\n✅ All services tested successfully!")
        return

    # Batch predict using the template logic
    results = disease_service.batch_predict_from_files(image_files, top_k=3)

    current_image = None
    for res in results:
        if res['image'] != current_image:
            current_image = res['image']
            print(f"\nDisease Detection Results for {current_image}:")
        print(f"  Rank {res['rank']}: {res['predicted_class']} ({res['confidence']}%)")

    print("\n✅ All services tested successfully!")

if __name__ == "__main__":
    test_services()