import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler

# 1. Load Data to configure the Scaler and Labels
print("📊 Loading data to configure scaler and labels...")
df_train = pd.read_csv('train_data.csv')
numeric_cols = ['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']

# Fit scaler on training data
scaler = StandardScaler()
scaler.fit(df_train[numeric_cols].values)

# Get class names (pandas get_dummies sorts them alphabetically by default)
class_names = pd.get_dummies(df_train['AQI_Class']).columns.tolist()

# 2. Match Paths to find a valid sample
image_dir = r"C:\Users\HP\Desktop\AQI_prediction\data\Air Pollution Image Dataset\Air Pollution Image Dataset\Country_wise_Dataset\India"
all_images = glob.glob(os.path.join(image_dir, '**', '*.*'), recursive=True)
image_path_dict = {os.path.basename(path): path for path in all_images}

df_train['clean_name'] = df_train['Filename'].apply(lambda x: os.path.basename(str(x).replace('\\', '/')))
df_train['full_path'] = df_train['clean_name'].map(image_path_dict)
df_valid = df_train.dropna(subset=['full_path']).reset_index(drop=True)

# Pick a completely random row to test!
sample = df_valid.sample(1).iloc[0]
img_path = sample['full_path']
actual_label = sample['AQI_Class']
numeric_values = sample[numeric_cols].values.astype(np.float32)

print(f"\n🎯 Selected Sample: {sample['clean_name']}")
print(f"📈 Real AQI Class: {actual_label}")
print(f"🔢 Pollutant Values: {dict(zip(numeric_cols, numeric_values))}")

# 3. Preprocess the inputs to match what the model expects
# A. Prepare Image
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_input = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

# B. Prepare Numeric Data
num_scaled = scaler.transform([numeric_values]) # Shape: (1, 6)
num_input = np.expand_dims(num_scaled, axis=1)  # Shape: (1, 1, 6)

# 4. Load Model and Predict
print("\n🧠 Loading Model...")
model = tf.keras.models.load_model('models/multimodal_model.keras')

print("🔮 Making Prediction...")
# Pass both inputs to Keras 3 as a tuple
predictions = model.predict((img_input, num_input), verbose=0) 

predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

print("\n" + "="*50)
print(f"🏆 PREDICTED AQI CLASS: {predicted_class} ({confidence:.2f}% confidence)")
print(f"✅ ACTUAL AQI CLASS:    {actual_label}")
print("="*50)
if predicted_class == actual_label:
    print("🎉 THE MODEL GOT IT RIGHT!")
else:
    print("❌ Close, but not quite. (It's still learning!)")
print("="*50 + "\n")