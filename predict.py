import numpy as np
import tensorflow as tf
import cv2
import os
import requests
from geopy.geocoders import Nominatim

# --- CONFIGURATION ---
API_KEY = "aa3f8be4d56a2dc8d99f5d632745ebcc"  # <--- Put your key here

def get_live_metrics(city_name):
    """Fetches real-time AQI metrics for a given city."""
    try:
        # 1. Convert City to Lat/Lon
        geolocator = Nominatim(user_agent="aqi_app")
        location = geolocator.geocode(city_name)
        if not location:
            raise ValueError(f"Could not find coordinates for {city_name}")
        
        lat, lon = location.latitude, location.longitude
        
        # 2. Call OpenWeather Air Pollution API
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        response = requests.get(url).json()
        
        if "list" not in response:
            raise ValueError("API Error: " + response.get("message", "Unknown error"))
            
        comps = response['list'][0]['components']
        
        # 3. Extract pollutants in the exact order your model expects:
        # Order: [PM2.5, PM10, O3, CO, SO2, NO2]
        metrics = [
            comps['pm2_5'], 
            comps['pm10'], 
            comps['o3'], 
            comps['co'], 
            comps['so2'], 
            comps['no2']
        ]
        
        print(f"✅ Fetched live metrics for {city_name}: {metrics}")
        return metrics

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def prepare_image(image_path, img_size=224):
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return np.expand_dims(img / 255.0, axis=0)

def predict_aqi_auto(image_path, city_name, model_path='models/multimodal_model.keras'):
    # 1. Get Metrics Automatically
    sensor_data = get_live_metrics(city_name)
    if not sensor_data: return

    # 2. Load Model
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        return
    model = tf.keras.models.load_model(model_path)

    # 3. Prep Inputs
    image_input = prepare_image(image_path)
    if image_input is None:
        print("❌ Image file not found!")
        return
    numeric_input = np.array(sensor_data).reshape(1, 1, 6)

    # 4. Predict
    predictions = model.predict([image_input, numeric_input])
    classes = ['Good', 'Moderate', 'Poor', 'Severe', 'Unhealthy', 'Very Unhealthy']
    class_idx = np.argmax(predictions)
    
    print("\n" + "="*30)
    print(f"📍 Location: {city_name}")
    print(f"🌍 AI Prediction: {classes[class_idx]}")
    print(f"📊 Confidence: {predictions[0][class_idx]*100:.1f}%")
    print("="*30)

if __name__ == "__main__":
    # TO TEST: 
    # 1. Update your API_KEY
    # 2. Provide an image path and a city name
    img = r"C:\Users\HP\Desktop\AQI_prediction\data\Air Pollution Image Dataset\Air Pollution Image Dataset\Combined_Dataset\All_img\DEL_VUH_2023-02-04-17.00-2-2.jpg"
    predict_aqi_auto(img, "Delhi")