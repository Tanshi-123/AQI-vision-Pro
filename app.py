import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from PIL import Image
import requests
from streamlit_geolocation import streamlit_geolocation
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- CONFIGURATION ---
st.set_page_config(page_title="AI AQI Predictor", layout="centered", page_icon="🌍")
API_KEY = "0c19719463bcf98090357f75665e62b3" 

# --- HEALTH ADVICE ENGINE ---
def get_health_advice(prediction):
    advice = {
        "Good": ("🟢", "Air quality is satisfactory. Enjoy your outdoor activities!"),
        "Satisfactory": ("🟡", "Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion."),
        "Moderate": ("🟠", "Sensitive groups should reduce prolonged or heavy outdoor exertion. General public is fine."),
        "Poor": ("🔴", "Everyone may begin to experience health effects. Limit prolonged outdoor exertion."),
        "Very Poor": ("🟣", "Health alert: Everyone may experience more serious health effects. Avoid outdoor physical activity."),
        "Very Unhealthy": ("🟣", "Health alert: Everyone may experience more serious health effects. Avoid outdoor physical activity."),
        "Severe": ("🟤", "Health warning of emergency conditions. The entire population is more likely to be affected. Stay indoors and use air purifiers.")
    }
    return advice.get(prediction, ("⚠️", "Please take standard precautions for air pollution."))

# --- AI SELF-AWARENESS (CONFLICT CHECKER) ---
def check_sensor_visual_conflict(predicted_class, pm25):
    """Detects if the AI's visual guess drastically contradicts the live sensors."""
    severe_classes = ["Poor", "Very Poor", "Very Unhealthy", "Severe"]
    good_classes = ["Good", "Satisfactory"]
    
    # Conflict 1: Sensors say clean (< 40 PM2.5), but AI sees heavy gray/smog
    if pm25 < 40 and predicted_class in severe_classes:
        return "👀 **AI Vision Alert:** The live sensors say the air is relatively clean, but the AI's eyes see heavy grayness/smog. *Are you indoors, or is it just cloudy/raining?*"
        
    # Conflict 2: Sensors say toxic (> 100 PM2.5), but AI sees clear air
    elif pm25 > 100 and predicted_class in good_classes:
        return "👀 **AI Vision Alert:** The live sensors detect high pollution, but the AI sees a clear image. *Did you upload a picture of a bright screen or a fake blue sky?*"
        
    return None

# --- THE AQI.in UI COMPONENT ---
def render_aqi_ui(metrics):
    pm25, pm10, co, no2, so2, o3 = metrics
    cigarettes = pm25 / 22.0
    st.markdown("---")
    if cigarettes >= 1:
        st.error(f"🚬 **Air Toxicity Equivalent:** Breathing this air today is equivalent to smoking **{int(cigarettes)} cigarettes**.")
    else:
        st.success("🍃 **Air Toxicity Equivalent:** The air is relatively clean today. (Equivalent to 0 cigarettes).")

    st.markdown("### ⚠️ Pollutant Danger Levels")
    st.markdown("*(Bars fill up as pollutants reach dangerous World Health Organization limits)*")
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"**PM2.5** ({pm25} µg/m³) - Fine Dust")
        st.progress(min(pm25 / 250.0, 1.0))
        st.caption(f"**PM10** ({pm10} µg/m³) - Coarse Dust")
        st.progress(min(pm10 / 430.0, 1.0))
        st.caption(f"**CO** ({co} µg/m³) - Carbon Monoxide")
        st.progress(min(co / 10000.0, 1.0))
        
    with col2:
        st.caption(f"**NO2** ({no2} µg/m³) - Nitrogen Dioxide")
        st.progress(min(no2 / 200.0, 1.0))
        st.caption(f"**SO2** ({so2} µg/m³) - Sulfur Dioxide")
        st.progress(min(so2 / 100.0, 1.0))
        st.caption(f"**O3** ({o3} µg/m³) - Ozone")
        st.progress(min(o3 / 160.0, 1.0))
    st.markdown("---")

# --- 1. LOAD THE REAL AI ---
@st.cache_resource
def load_ai_environment():
    model = tf.keras.models.load_model('models/multimodal_model.keras')
    train_df = pd.read_csv("new_imd_train_data.csv")
    le = LabelEncoder()
    le.fit(train_df['AQI_Class'])
    scaler = StandardScaler()
    numeric_cols = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
    scaler.fit(train_df[numeric_cols].fillna(0))
    return model, le, scaler

# --- 2. DATA PIPELINES ---
def get_live_api_metrics(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        comps = data['list'][0]['components']
        return [comps.get('pm2_5'), comps.get('pm10'), comps.get('co'), 
                comps.get('no2'), comps.get('so2'), comps.get('o3')]
    except Exception as e:
        return None

def prepare_image_for_model(image, img_size=224):
    img_array = np.array(image.convert('RGB'))
    h, w, _ = img_array.shape
    cropped_array = img_array[0:int(h * 0.6), 0:w] 
    img_resized = cv2.resize(cropped_array, (img_size, img_size))
    img_normalized = (img_resized / 127.5) - 1.0  
    return np.expand_dims(img_normalized, axis=0), cropped_array

def predict_with_real_ai(model, le, scaler, image, metrics):
    image_input, cropped_array = prepare_image_for_model(image)
    raw_numbers = pd.DataFrame([metrics], columns=['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3'])
    scaled_numbers = scaler.transform(raw_numbers)
    predictions = model.predict([image_input, scaled_numbers], verbose=0)
    class_idx = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][class_idx] * 100
    predicted_class = le.inverse_transform([class_idx])[0]
    display_class = predicted_class.split('_', 1)[-1].replace('_', ' ').title()
    if display_class == "Very Unhealthy": display_class = "Very Unhealthy"
    return display_class, confidence, cropped_array

# --- 3. UI DESIGN ---
st.title("🌍 Visual AQI Predictor")
st.markdown("Check your local air quality instantly using AI.")

st.info("ℹ️ **Tip:** For best results, ensure your photo is taken outdoors and includes the sky!")

with st.spinner("Loading AI Brain..."):
    model, le, scaler = load_ai_environment()

tab1, tab2 = st.tabs(["📸 Snap a Photo", "📁 Upload an Image"])

# ================= TAB 1: CAMERA =================
with tab1:
    st.header("Live Air Quality Check")
    st.markdown("**Step 1:** Tap below to get your location for accurate live sensor data.")
    location = streamlit_geolocation()
    
    st.markdown("**Step 2:** Snap a photo of the sky outside.")
    camera_image = st.camera_input("Take a photo", label_visibility="collapsed")
    
    if camera_image is not None:
        img = Image.open(camera_image)
        if not location.get('latitude'):
            st.error("Please click the Geolocation button above first!")
        else:
            with st.spinner("Analyzing atmosphere & fetching live data..."):
                metrics = get_live_api_metrics(location['latitude'], location['longitude'])
                if metrics:
                    predicted_class, confidence, cropped_img = predict_with_real_ai(model, le, scaler, img, metrics)
                    
                    st.success(f"### Predicted Air Quality: **{predicted_class.upper()}**")
                    
                    # --- NEW: TRIGGER CONFLICT CHECKER ---
                    conflict_msg = check_sensor_visual_conflict(predicted_class, pm25=metrics[0])
                    if conflict_msg:
                        st.info(conflict_msg)
                        
                    icon, advice_text = get_health_advice(predicted_class)
                    st.warning(f"{icon} **Health Advice:** {advice_text}")
                    st.caption(f"AI Confidence: {confidence:.2f}% | Source: Live GPS Data")
                    
                    render_aqi_ui(metrics)
                    
                    st.markdown("### 🔍 AI Vision Analysis:")
                    st.image(cropped_img, caption="Foreground Auto-Removed for Accuracy", use_container_width=True)
                else:
                    st.error("Could not fetch API data for this location.")

# ================= TAB 2: UPLOAD =================
with tab2:
    st.header("Check Air Quality from a Photo")
    city_name = st.text_input("Enter the city where this photo was taken:", placeholder="e.g., Delhi, Mumbai, London")
    uploaded_file = st.file_uploader("Upload a landscape photo", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and city_name:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Analyze Air Quality", type="primary"):
            with st.spinner(f"Analyzing image and fetching live data for {city_name}..."):
                try:
                    from geopy.geocoders import Nominatim
                    geolocator = Nominatim(user_agent="aqi_app")
                    loc = geolocator.geocode(city_name)
                    metrics = get_live_api_metrics(loc.latitude, loc.longitude)
                except:
                    metrics = None
                
                if metrics:
                    predicted_class, confidence, cropped_img = predict_with_real_ai(model, le, scaler, img, metrics)
                    
                    st.success(f"### Predicted Air Quality: **{predicted_class.upper()}**")
                    
                    # --- NEW: TRIGGER CONFLICT CHECKER ---
                    conflict_msg = check_sensor_visual_conflict(predicted_class, pm25=metrics[0])
                    if conflict_msg:
                        st.info(conflict_msg)
                        
                    icon, advice_text = get_health_advice(predicted_class)
                    st.warning(f"{icon} **Health Advice:** {advice_text}")
                    st.caption(f"AI Confidence: {confidence:.2f}% | Source: Live API Data for {city_name}")
                    
                    render_aqi_ui(metrics)
                    
                    st.markdown("### 🔍 AI Vision Analysis:")
                    st.image(cropped_img, caption="Foreground Auto-Removed for Accuracy", use_container_width=True)
                else:
                    st.error("Could not fetch API data. Please check the city name and try again.")