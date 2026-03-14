import numpy as np
import os
import argparse
import pandas as pd
import glob
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.model import build_multimodal_model 
from src.preprocessing import MultiModalDataGenerator

def main(args):
    print("📊 Loading DataFrames...")
    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv) if args.val_csv and os.path.exists(args.val_csv) else None

    print(f"🔍 Searching for images in: {args.image_dir}")
    # Recursively find all images to build a lookup dictionary
    all_images = glob.glob(os.path.join(args.image_dir, '**', '*.*'), recursive=True)
    
    # Store both "filename.jpg" and "filename" as keys for maximum compatibility
    image_path_dict = {}
    for path in all_images:
        fname = os.path.basename(path)
        image_path_dict[fname.lower()] = path
        image_path_dict[os.path.splitext(fname)[0].lower()] = path

    def map_paths(df):
        # Clean the filename column from the CSV
        df['clean_name'] = df[args.x_col].astype(str).str.strip().apply(
            lambda x: os.path.basename(x.replace('\\', '/')).lower()
        )
        
        # Try to map to the actual disk path
        df['full_path'] = df['clean_name'].map(image_path_dict)
        
        # Check for failures
        missing_count = df['full_path'].isna().sum()
        if missing_count > 0:
            print(f"⚠️ Warning: {missing_count} images were not found on disk.")
            print(f"   Example missing: {df[df['full_path'].isna()][args.x_col].head(3).tolist()}")
            
        return df.dropna(subset=['full_path']).reset_index(drop=True)

    df_train = map_paths(df_train)
    if df_val is not None:
        df_val = map_paths(df_val)

    print(f"✅ Found {len(df_train)} valid training images!")
    
    if len(df_train) == 0:
        print("❌ Error: No images found. Check if --image-dir or --x-col are correct.")
        return

    # Setup numeric columns (Update these if your CSV uses different names)
    numeric_cols = ['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']

    # Initialize Generators
    train_gen = MultiModalDataGenerator(
        df=df_train, x_col_img='full_path', x_cols_num=numeric_cols, 
        y_col=args.y_col, batch_size=args.batch_size, img_size=(args.img_size, args.img_size),
        is_training=True
    )
    
    val_gen = None
    if df_val is not None and len(df_val) > 0:
        val_gen = MultiModalDataGenerator(
            df=df_val, x_col_img='full_path', x_cols_num=numeric_cols, 
            y_col=args.y_col, batch_size=args.batch_size, img_size=(args.img_size, args.img_size),
            scaler=train_gen.scaler, 
            is_training=False
        )

    print("🧠 Building Multi-Modal Model...")
    model = build_multimodal_model(
        img_input_shape=(args.img_size, args.img_size, 3), 
        num_features=len(numeric_cols), 
        num_classes=train_gen.num_classes
    )

    # Calculate Class Weights to fix imbalance
    print("⚖️ Calculating Class Weights...")
    y_labels = df_train[args.y_col]
    
    # Get the unique class names
    classes = np.unique(y_labels)
    
    # Calculate weights using sklearn
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_labels
    )
    
    # Create the weight dictionary
    # We use train_gen.classes to ensure the indices match what the generator produces
    class_weight_dict = {}
    for i, class_name in enumerate(classes):
        class_weight_dict[i] = weights[i]
    
    print(f"Class Weights applied: {class_weight_dict}")
    
   
    
      # Callbacks
    os.makedirs('models', exist_ok=True)
    ckpt = ModelCheckpoint('models/multimodal_model.keras', save_best_only=True, monitor='val_loss' if val_gen else 'loss')
    es = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss' if val_gen else 'loss')

    print(f"🚀 Starting Training for {args.epochs} epochs...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weight_dict,
        callbacks=[ckpt, es]
    )
    print("✅ Training complete. Best model saved to 'models/multimodal_model.keras'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--val-csv', default='')
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--x-col', default='Filename')
    parser.add_argument('--y-col', default='AQI_Class')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    
    args = parser.parse_args()
    main(args)