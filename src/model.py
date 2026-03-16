import tensorflow as tf
from tensorflow.keras import layers, models

def build_multimodal_model(img_input_shape=(224, 224, 3), num_features=6, num_classes=6):
    # ==========================================
    # BRANCH A: The Vision Pathway (RESTORED!)
    # ==========================================
    img_input = tf.keras.Input(shape=img_input_shape, name="image_input")
    
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, 
        input_shape=img_input_shape,
        weights='imagenet'
    )
    base_model.trainable = False  
    
    x_img = base_model(img_input, training=False)
    x_img = layers.GlobalAveragePooling2D()(x_img)
    x_img = layers.Dense(128, activation='elu')(x_img)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.Dropout(0.3)(x_img)

    # ==========================================
    # BRANCH B: The Numeric Pathway
    # ==========================================
    num_input = tf.keras.Input(shape=(num_features,), name="numeric_input")
    
    x_num = layers.Dense(64, activation='elu')(num_input)
    x_num = layers.BatchNormalization()(x_num)
    x_num = layers.Dropout(0.2)(x_num)

    # ==========================================
    # THE MERGE: Combining Senses
    # ==========================================
    combined = layers.Concatenate()([x_img, x_num])
    
    z = layers.Dense(128, activation='elu')(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)
    
    z = layers.Dense(64, activation='elu')(z)
    z = layers.Dropout(0.2)(z)
    
    outputs = layers.Dense(num_classes, activation='softmax', name="output_layer")(z)

    # ==========================================
    # COMPILE (Using our bulletproof settings)
    # ==========================================
    model = models.Model(inputs=[img_input, num_input], outputs=outputs)
    custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
    
    model.compile(
        optimizer=custom_optimizer,
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model