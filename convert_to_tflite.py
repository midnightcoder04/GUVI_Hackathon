"""
Convert Keras model to TensorFlow Lite for faster inference.
Run this locally before deployment for optimal performance.

Usage:
    python convert_to_tflite.py
"""

import tensorflow as tf
import numpy as np

print("Loading Keras model...")

# Use TF 2.13's Keras which has better legacy compatibility
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

try:
    # Try loading with TF's Keras directly
    model = tf.keras.models.load_model("model/model.h5", compile=False)
except Exception as e:
    print(f"Error with standard loading: {e}")
    print("Trying alternative loading method...")
    
    # Alternative: Load weights only and reconstruct model
    # This requires knowing your model architecture
    from tensorflow.keras import layers, Model
    
    # Reconstruct the CNN architecture from README
    def build_cnn_model(input_shape=(40, 312, 1)):
        inputs = layers.Input(shape=input_shape, name='input')
        
        x = layers.Conv2D(32, (3, 3), padding='same', name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Activation('relu', name='relu1')(x)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.Dropout(0.25, name='dropout1')(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same', name='conv2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.Activation('relu', name='relu2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.Dropout(0.25, name='dropout2')(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', name='conv3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.Activation('relu', name='relu3')(x)
        x = layers.MaxPooling2D((2, 2), name='pool3')(x)
        x = layers.Dropout(0.25, name='dropout3')(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same', name='conv4')(x)
        x = layers.BatchNormalization(name='bn4')(x)
        x = layers.Activation('relu', name='relu4')(x)
        x = layers.GlobalAveragePooling2D(name='gap')(x)
        
        x = layers.Dense(128, activation='relu', name='dense1')(x)
        x = layers.Dropout(0.5, name='dropout4')(x)
        x = layers.Dense(64, activation='relu', name='dense2')(x)
        x = layers.Dropout(0.5, name='dropout5')(x)
        
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='VoiceClassifierCNN')
        return model
    
    model = build_cnn_model()
    model.load_weights("model/model.h5")
    print("Model loaded via weight loading")

print(f"Model input shape: {model.input_shape}")

# Convert to TFLite
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimizations for faster inference
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save the model
output_path = "model/model.tflite"
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved to: {output_path}")
print(f"Original model size: {len(open('model/model.h5', 'rb').read()) / 1024 / 1024:.2f} MB")
print(f"TFLite model size: {len(tflite_model) / 1024 / 1024:.2f} MB")

# Test the converted model
print("\nTesting TFLite model...")
interpreter = tf.lite.Interpreter(model_path=output_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# Test with dummy input
dummy_input = np.zeros(input_details[0]['shape'], dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(f"Test output: {output}")
print("\nConversion successful! Update app.py to use TFLite for faster inference.")
