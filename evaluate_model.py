import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Import your custom generator
from src.preprocessing import MultiModalDataGenerator 

# 1. Load Data & Configure Scaler
print("📊 Loading data and configuring scaler...")
df_train = pd.read_csv('train_data.csv')
df_val = pd.read_csv('val_data.csv')
numeric_cols = ['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']

scaler = StandardScaler()
scaler.fit(df_train[numeric_cols].values)

# Force strict category mapping so train and val match perfectly
class_names = pd.get_dummies(df_train['AQI_Class']).columns.tolist()
df_val['AQI_Class'] = pd.Categorical(df_val['AQI_Class'], categories=class_names)

# 2. Fix Image Paths for the Validation Set
print("🔍 Matching image paths...")
image_dir = r"C:\Users\HP\Desktop\AQI_prediction\data\Air Pollution Image Dataset\Air Pollution Image Dataset\Country_wise_Dataset\India"
all_images = glob.glob(os.path.join(image_dir, '**', '*.*'), recursive=True)
image_path_dict = {os.path.basename(path): path for path in all_images}

df_val['clean_name'] = df_val['Filename'].apply(lambda x: os.path.basename(str(x).replace('\\', '/')))
df_val['full_path'] = df_val['clean_name'].map(image_path_dict)
df_val = df_val.dropna(subset=['full_path']).reset_index(drop=True)

# 3. Create Validation Generator
print("⚙️ Creating validation data generator...")
val_gen = MultiModalDataGenerator(
    df=df_val,
    x_col_img='full_path',
    x_cols_num=numeric_cols,
    y_col='AQI_Class',
    batch_size=32,
    scaler=scaler,
    is_training=False # IMPORTANT: is_training=False prevents shuffling so labels match predictions!
)

# 4. Load Model and Predict
print("🧠 Loading model and grading predictions (this may take a minute or two)...")
model = tf.keras.models.load_model('models/multimodal_model.keras')
predictions = model.predict(val_gen)

y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(pd.get_dummies(df_val['AQI_Class']).values, axis=1)

# Clean up class names for the terminal and plot
clean_class_names = [name.split('_', 1)[-1].replace('_', ' ') for name in class_names]

# 5. Print Classification Report
print("\n" + "="*65)
print("📊 CLASSIFICATION REPORT (Precision, Recall, F1-Score)")
print("="*65)
print(classification_report(y_true, y_pred, target_names=clean_class_names))

# 6. Plot Confusion Matrix
print("\n🎨 Generating Confusion Matrix Plot...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=clean_class_names, 
            yticklabels=clean_class_names)
plt.title('AQI Multi-Modal Confusion Matrix')
plt.ylabel('Actual Real-World AQI')
plt.xlabel('Model Predicted AQI')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save and show the plot
plt.savefig('confusion_matrix.png')
plt.show()

print("✅ Done! Saved matrix as 'confusion_matrix.png'")