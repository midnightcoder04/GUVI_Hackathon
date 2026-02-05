"""
Convert Keras model to TensorFlow Lite for faster inference.
Run this locally before deployment for optimal performance.

Usage:
    python convert_to_tflite.py
"""

import tensorflow as tf
import numpy as np

# Load the original model
print("Loading Keras model...")
model = tf.keras.models.load_model("model/model.h5", compile=False)
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
