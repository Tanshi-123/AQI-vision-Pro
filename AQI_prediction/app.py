from flask import Flask, request, jsonify
import io
import os
from PIL import Image
import numpy as np
import tensorflow as tf


app = Flask(__name__)

AQI_API_KEY = os.environ.get("AQI_API_KEY")
AQI_API_URL = "https://api.waqi.info/feed/"

# Model path
MODEL_PATH = os.environ.get("AQI_MODEL_PATH", "models/model.h5")

# Load model globally at startup (Replaces the deprecated @app.before_first_request)
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return tf.keras.models.load_model(path)

try:
    model = load_model()
    print("✅ Model loaded successfully on startup!")
except Exception as e:
    print(f"⚠️ Warning: Model not loaded at startup. Error: {e}")
    model = None

# AQI Categories (Make sure these match the order your model was trained on!)
AQI_CLASSES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

def preprocess_image(img, target_size=(224, 224)):
    """Resizes the image and converts to a numpy array.
       No scaling (like /255) is done here because MobileNetV2 has 
       a built-in preprocess_input layer!"""
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return img_array

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    global model
    
    # Fallback to load model if it failed on startup
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided. Use form field 'image'"}), 400

    f = request.files['image']
    img_bytes = f.read()
    
    try:
        # Load and convert image to RGB (drops alpha channels if PNG)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({"error": "Invalid image file provided."}), 400

    # Preprocess and predict
    x = preprocess_image(img)
    x = np.expand_dims(x, axis=0) # Creates a batch of 1: shape (1, 224, 224, 3)
    
    preds = model.predict(x)
    preds_list = preds[0].tolist()

    top_idx = int(np.argmax(preds_list))
    
    # Map to human-readable class safely
    predicted_category = AQI_CLASSES[top_idx] if top_idx < len(AQI_CLASSES) else "Unknown"

    return jsonify({
        "predictions": preds_list, 
        "predicted_index": top_idx,
        "predicted_category": predicted_category
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)