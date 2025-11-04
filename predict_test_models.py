import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import pandas as pd
import os
from tqdm import tqdm  # progress bar

# =========================
# Paths
# =========================
SAVED_MODEL_PATH = r"C:\Users\ADEGOKE\Desktop\CROP DISEASE DETECTION PROJECT\AI_Crop_Disease_Monitoring\src\crop_monitor\models\plantvillage_model_tf"
IMAGE_FOLDER = r"C:\Users\ADEGOKE\Desktop\CROP DISEASE DETECTION PROJECT\AI_Crop_Disease_Monitoring\src\crop_monitor\image_to_predict"
OUTPUT_CSV = r"C:\Users\ADEGOKE\Desktop\CROP DISEASE DETECTION PROJECT\AI_Crop_Disease_Monitoring\src\crop_monitor\predictions.csv"
IMG_SIZE = (128, 128)
TOP_K = 3  # top-k predictions

# =========================
# Load model
# =========================
model = tf.keras.models.load_model(SAVED_MODEL_PATH)
print("✅ Model loaded successfully.")

# Replace with your actual class names in the correct order
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# =========================
# Helper functions
# =========================
def preprocess_image(img_path):
    """Load and preprocess a single image"""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))
    return img_array

# =========================
# Collect all image paths
# =========================
image_files = [
    os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
    if os.path.isfile(os.path.join(IMAGE_FOLDER, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    print("⚠️ No valid image files found in folder!")
    exit()

# =========================
# Preprocess all images in a batch
# =========================
batch_images = np.vstack([preprocess_image(f) for f in image_files])

# =========================
# Predict in batch
# =========================
preds = model.predict(batch_images, verbose=1)

# =========================
# Collect results
# =========================
results = []
for idx, img_path in enumerate(image_files):
    top_indices = preds[idx].argsort()[-TOP_K:][::-1]
    for rank, i in enumerate(top_indices, start=1):
        results.append({
            "image": os.path.basename(img_path),
            "rank": int(rank),
            "predicted_class": str(CLASS_NAMES[i]),
            "confidence": float(round(float(preds[idx][i]) * 100, 2))
        })

# =========================
# Save to CSV
# =========================
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Predictions saved to {OUTPUT_CSV}")

