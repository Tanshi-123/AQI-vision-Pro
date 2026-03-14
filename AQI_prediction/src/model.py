import tensorflow as tf
from tensorflow.keras import layers, models

def build_multimodal_model(img_input_shape=(224, 224, 3), num_features=6, num_classes=6):
    # ==========================================
    # BRANCH A: The Vision Pathway (MobileNetV2)
    # ==========================================
    img_input = tf.keras.Input(shape=img_input_shape, name="image_input")
    
    # Keras preprocessing specifically for MobileNetV2
    x_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_input)
    
    # Load base model (weights pre-trained on ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, 
        input_shape=img_input_shape,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze the base model
    
    x_img = base_model(x_img, training=False)
    x_img = layers.GlobalAveragePooling2D()(x_img)
    x_img = layers.Dropout(0.3)(x_img)
    # Output shape of Branch A: (batch_size, 1280)

    # ==========================================
    # BRANCH B: The Numeric Pathway (LSTM)
    # ==========================================
    # Input shape: (time_steps=1, features=6)
    num_input = tf.keras.Input(shape=(1, num_features), name="numeric_input")
    
    x_num = layers.LSTM(64, activation='tanh', return_sequences=False)(num_input)
    x_num = layers.Dense(32, activation='relu')(x_num)
    x_num = layers.Dropout(0.2)(x_num)
    # Output shape of Branch B: (batch_size, 32)

    # ==========================================
    # THE MERGE: Combining Senses
    # ==========================================
    combined = layers.Concatenate()([x_img, x_num])
    
    # Final dense layers to figure out the AQI class from the combined data
    z = layers.Dense(128, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    outputs = layers.Dense(num_classes, activation='softmax', name="output_layer")(z)

    # ==========================================
    # COMPILE
    # ==========================================
    model = models.Model(inputs=[img_input, num_input], outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model