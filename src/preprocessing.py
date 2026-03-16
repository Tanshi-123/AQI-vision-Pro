import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler, LabelEncoder

class MultiModalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, x_col_img, x_cols_num, y_col, batch_size=32, img_size=(224, 224), 
                 num_classes=6, scaler=None, is_training=True, **kwargs):
        super().__init__(**kwargs) 
        
        self.df = df.reset_index(drop=True)
        self.x_col_img = x_col_img       
        self.x_cols_num = x_cols_num     
        self.y_col = y_col               
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.is_training = is_training
        
        # Strict integer encoding 
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.df[self.y_col])
        
        self.numeric_data = self.df[self.x_cols_num].fillna(0).values.astype(np.float32)
        
        if scaler is None:
            self.scaler = StandardScaler()
            self.numeric_data = self.scaler.fit_transform(self.numeric_data)
        else:
            self.scaler = scaler
            self.numeric_data = self.scaler.transform(self.numeric_data)
            
        # Notice: No more np.expand_dims here! It is now a flat array.

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self.df.index[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        batch_images = []
        batch_numeric = []
        batch_labels = []

        for i in batch_idx:
            img_path = self.df.loc[i, self.x_col_img]
            img = load_img(img_path, target_size=self.img_size)
            img = img_to_array(img)
            img = (img / 127.5) - 1.0  
            
            num_data = self.numeric_data[i]
            label = self.labels[i] 

            batch_images.append(img)
            batch_numeric.append(num_data)
            batch_labels.append(label)

        return (np.array(batch_images), np.array(batch_numeric)), np.array(batch_labels)
    
    def on_epoch_end(self):
        if self.is_training:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            self.labels = self.label_encoder.transform(self.df[self.y_col])
            self.numeric_data = self.df[self.x_cols_num].fillna(0).values.astype(np.float32)
            self.numeric_data = self.scaler.transform(self.numeric_data)
            # Notice: No np.expand_dims here either!