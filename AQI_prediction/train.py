import os
import argparse
import pandas as pd
import glob
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.model import build_multimodal_model
from src.preprocessing import MultiModalDataGenerator

def main(args):
    print("📊 Loading DataFrames...")
    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv) if args.val_csv and os.path.exists(args.val_csv) else None

    print(f"🔍 Hunting down nested images in: {args.image_dir}")
    all_images = glob.glob(os.path.join(args.image_dir, '**', '*.*'), recursive=True)
    image_path_dict = {os.path.basename(path): path for path in all_images}

    # Clean names and map to full paths
    def map_paths(df):
        df['clean_name'] = df[args.x_col].apply(lambda x: os.path.basename(str(x).replace('\\', '/')))
        df['full_path'] = df['clean_name'].map(image_path_dict)
        return df.dropna(subset=['full_path']).reset_index(drop=True)

    df_train = map_paths(df_train)
    if df_val is not None:
        df_val = map_paths(df_val)

    print(f"✅ Found {len(df_train)} valid training images!")

    # Setup numeric columns
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
            scaler=train_gen.scaler, # Reuse training scaler to prevent data leakage!
            is_training=False
        )

    print("🧠 Building Multi-Modal Model (CNN + LSTM)...")
    model = build_multimodal_model(
        img_input_shape=(args.img_size, args.img_size, 3), 
        num_features=len(numeric_cols), 
        num_classes=train_gen.num_classes
    )

    os.makedirs('models', exist_ok=True)
    ckpt = ModelCheckpoint('models/multimodal_model.keras', save_best_only=True, monitor='val_loss' if val_gen else 'loss')
    es = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss' if val_gen else 'loss')

    print("🚀 Starting training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[ckpt, es]
    )
    print("✅ Training complete. Best model saved to 'models/multimodal_model.keras'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--val-csv', default='')
    parser.add_argument('--image-dir', required=True, help='Root directory to search for images')
    parser.add_argument('--x-col', default='Filename')
    parser.add_argument('--y-col', default='AQI_Class')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    
    args = parser.parse_args()
    main(args)