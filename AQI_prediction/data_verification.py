import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2

# 1. DEFINE PATHS (Change these based on your specific folder names)
BASE_PATH = "data/Country_wise_Dataset/India/"
CSV_PATH = os.path.join(BASE_PATH, "city_data.csv") 
IMAGE_DIR = os.path.join(BASE_PATH, "Images")

# 2. LOAD DATA
df = pd.read_csv(CSV_PATH)

# 3. LINK IMAGES
# We create a full path so the CNN can find them easily later
df['full_path'] = df['Filename'].apply(lambda x: os.path.join(IMAGE_DIR, x))

# 4. VERIFY DATA INTEGRITY
print(f"Total entries: {len(df)}")
missing_images = df['full_path'].apply(lambda x: not os.path.exists(x)).sum()
print(f"Missing images: {missing_images}")

# 5. DATA CLEANING
# Check for nulls in pollutants
pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
print("\nMissing Values per Pollutant:")
print(df[pollutants].isnull().sum())

# Add this after your data cleaning steps
def get_aqi_category(aqi_value):
    if aqi_value <= 50: return "Good"
    elif aqi_value <= 100: return "Satisfactory"
    elif aqi_value <= 200: return "Moderate"
    elif aqi_value <= 300: return "Poor"
    elif aqi_value <= 400: return "Very Poor"
    else: return "Severe"

# Assuming your CSV has a column named 'AQI'
df['label'] = df['AQI'].apply(get_aqi_category)

# Save the cleaned CSV so train.py can use it!
df.to_csv("data/cleaned_train_data.csv", index=False)

# Fill missing values using linear interpolation (Crucial for time-series)
df[pollutants] = df[pollutants].interpolate(method='linear', limit_direction='both')