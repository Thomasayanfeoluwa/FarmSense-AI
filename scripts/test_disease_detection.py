# # scripts/test_disease_detection.py
# import sys
# import os

# # Add the src directory to the Python path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# from src.crop_monitor.services.disease_service import DiseaseService

# def test_disease_detection():
#     print("Testing Disease Detection Service...")
    
#     # Initialize the service
#     disease_service = DiseaseService()
    
#     # Test with a sample image (you'll need to provide a test image)
#     test_image_path = "data/test/test_plant.jpg"  # Create this path and add a test image
    
#     if os.path.exists(test_image_path):
#         # For this test, we'll just check that the model loads correctly
#         print("Disease model loaded successfully!")
#         print(f"Model path: {disease_service.model_path}")
#     else:
#         print("No test image found, but disease service initialized correctly.")
#         print(f"Model path: {disease_service.model_path}")

# if __name__ == "__main__":
#     test_disease_detection()







