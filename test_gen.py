import os
import glob
import pandas as pd
from src.preprocessing import MultiModalDataGenerator

print("📊 Loading data and matching paths...")
# 1. Load CSV
df = pd.read_csv('train_data.csv')

# 2. Match Paths (using the exact absolute path that worked for you!)
image_dir = r"C:\Users\HP\Desktop\AQI_prediction\data\Air Pollution Image Dataset\Air Pollution Image Dataset\Country_wise_Dataset\India"
all_images = glob.glob(os.path.join(image_dir, '**', '*.*'), recursive=True)
image_path_dict = {os.path.basename(path): path for path in all_images}

df['clean_name'] = df['Filename'].apply(lambda x: os.path.basename(str(x).replace('\\', '/')))
df['full_path'] = df['clean_name'].map(image_path_dict)
df = df.dropna(subset=['full_path']).reset_index(drop=True)

print(f"✅ Found {len(df)} valid rows for testing.")

# 3. Initialize our new Generator
print("⚙️ Firing up the MultiModal Data Generator...")
numeric_cols = ['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']

gen = MultiModalDataGenerator(
    df=df, 
    x_col_img='full_path', 
    x_cols_num=numeric_cols, 
    y_col='AQI_Class', 
    batch_size=32
)

# 4. Fetch the very first batch
print("📦 Fetching Batch 0...")
(X_img, X_num), Y = gen[0]

print("\n🎉 SUCCESS! The generator is working perfectly. Here are your batch shapes:")
print(f"📷 Image Batch Shape:   {X_img.shape} -> (batch_size, height, width, channels)")
print(f"🔢 Numeric Batch Shape: {X_num.shape} -> (batch_size, time_steps, features)")
print(f"🏷️ Label Batch Shape:   {Y.shape} -> (batch_size, num_classes)\n")