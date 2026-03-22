import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
import requests
from geopy.geocoders import Nominatim
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- CONFIGURATION ---
API_KEY = "aa3f8be4d56a2dc8d99f5d632745ebcc"  # Your API Key

def setup_prediction_environment():
    print("⚙️ Setting up scalers and labels from training data...")
    train_df = pd.read_csv("new_imd_train_data.csv")
    
    # 1. Setup Label Encoder (To map 0,1,2,3,4,5 to the exact text labels)
    le = LabelEncoder()
    le.fit(train_df['AQI_Class'])
    
    # 2. Setup StandardScaler
    scaler = StandardScaler()
    numeric_cols = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
    scaler.fit(train_df[numeric_cols].fillna(0))
    
    return le, scaler

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
        
        # 3. Extract pollutants in the EXACT order the model was trained on!
        # Order: ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
        metrics = [
            comps['pm2_5'], 
            comps['pm10'], 
            comps['co'], 
            comps['no2'], 
            comps['so2'], 
            comps['o3']
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
    
    # EXACT mathematical preprocessing used during training
    img_array = (img / 127.5) - 1.0  
    return np.expand_dims(img_array, axis=0) # Shape: (1, 224, 224, 3)

def predict_aqi_auto(image_path, city_name, model, le, scaler):
    # 1. Get Metrics Automatically
    sensor_data = get_live_metrics(city_name)
    if not sensor_data: return

    # 2. Prep Inputs
    image_input = prepare_image(image_path)
    if image_input is None:
        print("❌ Image file not found!")
        return
        
    # Scale the numbers using the training scaler!
    raw_numbers = pd.DataFrame([sensor_data], columns=['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3'])
    scaled_numbers = scaler.transform(raw_numbers)

    # 3. Predict
    predictions = model.predict([image_input, scaled_numbers], verbose=0)
    class_idx = np.argmax(predictions, axis=1)[0]
    
    predicted_class = le.inverse_transform([class_idx])[0]
    confidence = predictions[0][class_idx] * 100
    
    print("\n" + "="*40)
    print(f"📍 Location: {city_name}")
    print(f"🖼️ Image: {os.path.basename(image_path)}")
    print(f"🌍 AI Prediction: {predicted_class.upper()}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    # You will need to install geopy if you haven't yet:
    # pip install geopy requests
    
    print("🧠 Loading the 93% accuracy model...")
    model = tf.keras.models.load_model('models/multimodal_model.keras')
    
    le, scaler = setup_prediction_environment()
    
    # Let's test Delhi!
    img = r"C:\Users\HP\Desktop\AQI-vision-Pro\data\Air Pollution Image Dataset\Air Pollution Image Dataset\Combined_Dataset\All_img\DEL_VUH_2023-02-04-17.00-2-2.jpg"
    
    predict_aqi_auto(img, "Delhi", model, le, scaler)