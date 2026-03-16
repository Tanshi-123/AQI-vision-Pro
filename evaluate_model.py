import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocessing import MultiModalDataGenerator

def evaluate_my_model():
    print("🧠 Loading the 93% accuracy model...")
    model = tf.keras.models.load_model('models/multimodal_model.keras')
    
    print("📊 Loading validation data...")
    val_df = pd.read_csv("new_imd_val_data.csv")
    
    # Same columns we used for training
    x_col_img = 'Filename'
    x_cols_num = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
    y_col = 'AQI_Class'
    
    # ================= THE FIX =================
    # Tell the script exactly where the images are located so it can find them!
    img_dir = "data/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset/All_img"
    val_df[x_col_img] = val_df[x_col_img].apply(lambda x: os.path.join(img_dir, str(x)))
    # ===========================================

    # Initialize generator (is_training=False so it doesn't shuffle!)
    val_gen = MultiModalDataGenerator(
        df=val_df,
        x_col_img=x_col_img,
        x_cols_num=x_cols_num,
        y_col=y_col,
        batch_size=32,
        is_training=False 
    )
    
    print("🔍 Making predictions... (this might take a minute)")
    predictions = model.predict(val_gen)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels directly from the generator
    y_true = val_gen.labels
    
    # Get the class names back from the encoder
    class_names = val_gen.label_encoder.classes_
    
    print("\n================ CLASSIFICATION REPORT ================")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Draw the Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Final Model Confusion Matrix - 93% Accuracy')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('final_confusion_matrix.png')
    print("\n✅ Confusion Matrix saved as 'final_confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_my_model()