"""
Benchmark all model formats on real dataset.
Tests on random 5% of samples from build/clips and build/clips_AI.

Measures:
- Accuracy
- Model size
- Inference time (mean, std)
- CPU usage

Usage:
    python benchmark.py
"""

import os
import time
import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path
import random
from io import BytesIO
import json
from datetime import datetime
import psutil
import xgboost as xgb

# Configuration
SAMPLE_RATE = 16000
N_LFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
TARGET_TIME_STEPS = 312
MAX_DURATION = 10.0
TEST_SAMPLE_PERCENT = 0.05  # 5% of data

# Paths
HUMAN_DIR = "build/clips"
AI_DIR = "build/clips_AI"
RESULTS_FILE = "benchmark_results.json"

# Load normalization params
norm_params = np.load("model/normalization_params.npz")
MEAN = np.array(norm_params['mean'], dtype=np.float32)
STD = np.array(norm_params['std'], dtype=np.float32)

# ============== Feature Extraction ==============
def extract_features(audio_path):
    """Extract LFCC features from audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
        
        lfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_LFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Pad or truncate to TARGET_TIME_STEPS
        if lfcc.shape[1] < TARGET_TIME_STEPS:
            pad_width = TARGET_TIME_STEPS - lfcc.shape[1]
            lfcc = np.pad(lfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            lfcc = lfcc[:, :TARGET_TIME_STEPS]
        
        # Normalize
        lfcc = lfcc.astype(np.float32)
        
        mean = MEAN
        std = STD
        if mean.ndim == 1:
            mean = mean[:, np.newaxis]
        if std.ndim == 1:
            std = std[:, np.newaxis]
        
        lfcc = (lfcc - mean) / (std + 1e-8)
        
        # Reshape for model input
        features = lfcc[np.newaxis, ..., np.newaxis]
        return features
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def extract_features_raw(audio_path):
    """Extract raw LFCC features (for XGBoost) without normalization"""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
        
        lfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_LFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Pad or truncate to TARGET_TIME_STEPS
        if lfcc.shape[1] < TARGET_TIME_STEPS:
            pad_width = TARGET_TIME_STEPS - lfcc.shape[1]
            lfcc = np.pad(lfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            lfcc = lfcc[:, :TARGET_TIME_STEPS]
        
        # Return as-is without normalization
        return lfcc.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# ============== Model Loaders ==============
def build_cnn_model(input_shape=(40, 312, 1)):
    """Rebuild CNN architecture"""
    from tensorflow.keras import layers, Model
    
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
    
    return Model(inputs=inputs, outputs=outputs, name='VoiceClassifierCNN')

class KerasModel:
    def __init__(self, model_path):
        print(f"Loading Keras model from {model_path}...")
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
        except:
            self.model = build_cnn_model()
            self.model.load_weights(model_path)
        self.model.trainable = False
        
    def predict(self, features):
        return self.model(features, training=False)[0][0].numpy()

class TFLiteModel:
    def __init__(self, model_path):
        print(f"Loading TFLite model from {model_path}...")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def predict(self, features):
        # Handle INT8 input
        if self.input_details[0]['dtype'] == np.uint8:
            scale, zero_point = self.input_details[0]['quantization']
            features_quant = (features / scale + zero_point).astype(np.uint8)
            self.interpreter.set_tensor(self.input_details[0]['index'], features_quant)
        else:
            self.interpreter.set_tensor(self.input_details[0]['index'], features)
        
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Handle INT8 output
        if self.output_details[0]['dtype'] == np.uint8:
            scale, zero_point = self.output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale
        
        return output[0][0]

class XGBoostModel:
    def __init__(self, model_path):
        print(f"Loading XGBoost model from {model_path}...")
        self.is_sklearn = False
        
        # Load based on file extension
        if model_path.endswith('.json'):
            self.model = xgb.Booster()
            self.model.load_model(model_path)
        elif model_path.endswith('.ubj'):
            self.model = xgb.Booster()
            self.model.load_model(model_path)
        elif model_path.endswith('.pkl'):
            import pickle
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_sklearn = True
        else:
            raise ValueError(f"Unsupported XGBoost format: {model_path}")
    
    def extract_statistical_features(self, lfcc_matrix):
        """
        Extract statistical features from LFCC for XGBoost.
        Input: (40, 312) LFCC matrix
        Output: (365,) statistical features
        """
        feat = []
        
        # Per-coefficient statistics (40 coefficients × 9 stats = 360 features)
        feat.extend(np.mean(lfcc_matrix, axis=1))       # 40
        feat.extend(np.std(lfcc_matrix, axis=1))        # 40
        feat.extend(np.min(lfcc_matrix, axis=1))        # 40
        feat.extend(np.max(lfcc_matrix, axis=1))        # 40
        feat.extend(np.percentile(lfcc_matrix, 25, axis=1))  # 40
        feat.extend(np.percentile(lfcc_matrix, 50, axis=1))  # 40
        feat.extend(np.percentile(lfcc_matrix, 75, axis=1))  # 40
        feat.extend(np.max(lfcc_matrix, axis=1) - np.min(lfcc_matrix, axis=1))  # 40 (range)
        feat.extend(np.percentile(lfcc_matrix, 75, axis=1) - np.percentile(lfcc_matrix, 25, axis=1))  # 40 (IQR)
        
        # Overall statistics (5 features)
        feat.extend([
            np.mean(lfcc_matrix),
            np.std(lfcc_matrix),
            np.min(lfcc_matrix),
            np.max(lfcc_matrix),
            np.median(lfcc_matrix)
        ])
        
        return np.array(feat)
    
    def predict(self, features_raw):
        # XGBoost expects statistical features (365 features) from RAW LFCC
        # Input: (40, 312) raw LFCC matrix (not normalized)
        stat_features = self.extract_statistical_features(features_raw)
        stat_features = stat_features.reshape(1, -1).astype(np.float32)
        
        if self.is_sklearn:
            # XGBClassifier: use predict_proba
            pred_proba = self.model.predict_proba(stat_features)[0]
            return float(pred_proba[1])  # Probability of AI class
        else:
            # Booster: use DMatrix and predict
            dmatrix = xgb.DMatrix(stat_features)
            pred = self.model.predict(dmatrix)
            return float(pred[0])

# ============== Dataset Loading ==============
def load_dataset():
    """Load 5% random samples from human and AI directories"""
    print(f"\n{'='*60}")
    print("Loading Dataset")
    print(f"{'='*60}")
    
    human_files = []
    ai_files = []
    
    # Scan human clips
    if os.path.exists(HUMAN_DIR):
        for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
            human_files.extend(list(Path(HUMAN_DIR).rglob(ext)))
    
    # Scan AI clips
    if os.path.exists(AI_DIR):
        for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
            ai_files.extend(list(Path(AI_DIR).rglob(ext)))
    
    print(f"Found: {len(human_files)} human, {len(ai_files)} AI samples")
    
    # Random 5% sample
    num_human = max(1, int(len(human_files) * TEST_SAMPLE_PERCENT))
    num_ai = max(1, int(len(ai_files) * TEST_SAMPLE_PERCENT))
    
    human_sample = random.sample(human_files, min(num_human, len(human_files)))
    ai_sample = random.sample(ai_files, min(num_ai, len(ai_files)))
    
    print(f"Testing on: {len(human_sample)} human, {len(ai_sample)} AI samples ({TEST_SAMPLE_PERCENT*100:.1f}%)")
    
    # Load features and labels
    dataset = []
    
    print("\nExtracting features from HUMAN samples...")
    for i, path in enumerate(human_sample):
        features = extract_features(str(path))
        features_raw = extract_features_raw(str(path))
        if features is not None and features_raw is not None:
            dataset.append((features, features_raw, 0, str(path)))  # 0 = HUMAN
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(human_sample)} processed")
    
    print("\nExtracting features from AI samples...")
    for i, path in enumerate(ai_sample):
        features = extract_features(str(path))
        features_raw = extract_features_raw(str(path))
        if features is not None and features_raw is not None:
            dataset.append((features, features_raw, 1, str(path)))  # 1 = AI
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(ai_sample)} processed")
    
    print(f"\nTotal valid samples: {len(dataset)}")
    return dataset

# ============== Helper Functions ==============
def get_language_from_filename(filepath):
    """Extract language code from filename (en, ma, te, ta, hi)"""
    filename = os.path.basename(filepath).lower()
    
    language_codes = ['en', 'ma', 'te', 'ta', 'hi']
    for code in language_codes:
        if filename.startswith(code):
            return code
    return 'unknown'

def get_language_name(code):
    """Convert language code to full name"""
    names = {
        'en': 'English',
        'ma': 'Malayalam',
        'te': 'Telugu',
        'ta': 'Tamil',
        'hi': 'Hindi',
        'unknown': 'Unknown'
    }
    return names.get(code, 'Unknown')

# ============== Benchmarking ==============
def benchmark_model(model, dataset, model_name, is_xgboost=False):
    """Benchmark a single model"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    predictions = []
    ground_truth = []
    inference_times = []
    error_files = []  # Track misclassified files
    
    process = psutil.Process(os.getpid())
    cpu_usage_samples = []
    
    for features, features_raw, label, path in dataset:
        # Select appropriate features
        input_features = features_raw if is_xgboost else features
        
        # Warm up
        _ = model.predict(input_features)
        
        # Measure inference time
        cpu_before = process.cpu_percent()
        start = time.perf_counter()
        pred = model.predict(input_features)
        end = time.perf_counter()
        cpu_after = process.cpu_percent()
        
        inference_time = (end - start) * 1000  # ms
        inference_times.append(inference_time)
        cpu_usage_samples.append((cpu_after + cpu_before) / 2)
        
        # Classification (threshold = 0.5)
        pred_class = 1 if pred >= 0.5 else 0
        predictions.append(pred_class)
        ground_truth.append(label)
        
        # Track misclassifications
        if pred_class != label:
            error_files.append({
                'file': os.path.basename(path),
                'actual': 'AI' if label == 1 else 'Human',
                'predicted': 'AI' if pred_class == 1 else 'Human',
                'confidence': float(pred),
                'language': get_language_from_filename(path)
            })
    
    # Calculate metrics
    correct = sum(p == gt for p, gt in zip(predictions, ground_truth))
    accuracy = correct / len(predictions) * 100
    
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    mean_cpu = np.mean(cpu_usage_samples)
    
    # Group errors by language
    errors_by_language = {}
    for error in error_files:
        lang = error['language']
        if lang not in errors_by_language:
            errors_by_language[lang] = []
        errors_by_language[lang].append(error)
    
    # Calculate language-specific error rates
    language_stats = {}
    for features, features_raw, label, path in dataset:
        lang = get_language_from_filename(path)
        if lang not in language_stats:
            language_stats[lang] = {'total': 0, 'errors': 0}
        language_stats[lang]['total'] += 1
    
    for lang, errors in errors_by_language.items():
        language_stats[lang]['errors'] = len(errors)
    
    # Add languages with no errors
    for lang in language_stats:
        if lang not in errors_by_language:
            errors_by_language[lang] = []
    
    print(f"\nResults:")
    print(f"  Accuracy:        {accuracy:.2f}%")
    print(f"  Avg Time:        {mean_time:.2f} ms")
    print(f"  Std Time:        {std_time:.2f} ms")
    print(f"  Min Time:        {min_time:.2f} ms")
    print(f"  Max Time:        {max_time:.2f} ms")
    print(f"  Avg CPU:         {mean_cpu:.1f}%")
    print(f"  Total Errors:    {len(error_files)}")
    
    # Print language-specific stats
    if language_stats:
        print(f"\n  Errors by Language:")
        for lang in sorted(language_stats.keys()):
            stats = language_stats[lang]
            error_rate = (stats['errors'] / stats['total'] * 100) if stats['total'] > 0 else 0
            lang_name = get_language_name(lang)
            print(f"    {lang_name:<12} ({lang}): {stats['errors']}/{stats['total']} errors ({error_rate:.1f}% error rate)")
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "avg_inference_ms": mean_time,
        "std_inference_ms": std_time,
        "min_inference_ms": min_time,
        "max_inference_ms": max_time,
        "avg_cpu_percent": mean_cpu,
        "total_samples": len(predictions),
        "correct_predictions": correct,
        "total_errors": len(error_files),
        "errors_by_language": {lang: len(errors) for lang, errors in errors_by_language.items()},
        "language_stats": language_stats,
        "error_details": error_files[:20]  # First 20 errors for inspection
    }

def main():
    print("="*60)
    print("Model Benchmark Suite")
    print("="*60)
    
    # Load dataset
    dataset = load_dataset()
    
    if len(dataset) == 0:
        print("❌ No samples found in build/clips or build/clips_AI")
        return
    
    # Model configurations
    models_to_test = [
        ("model/model.h5", "Keras H5", "keras"),
        ("model/model_fp32.tflite", "TFLite FP32", "tflite"),
        ("model/model_int8.tflite", "TFLite INT8", "tflite"),
        ("model/model_int8_hybrid.tflite", "TFLite INT8 Hybrid", "tflite"),
        ("model/model_xgboost.json", "XGBoost JSON", "xgboost"),
        ("model/model_xgboost_5depth.json", "XGBoost JSON", "xgboost"),
        ("model/model_xgboost.ubj", "XGBoost UBJ", "xgboost"),
        ("model/model_xgboost.pkl", "XGBoost PKL", "xgboost"),
    ]
    
    results = []
    
    for model_path, model_name, model_type in models_to_test:
        if not os.path.exists(model_path):
            print(f"\n⚠ Skipping {model_name} (file not found: {model_path})")
            continue
        
        # Get model size
        model_size_mb = os.path.getsize(model_path) / 1024 / 1024
        
        # Load model
        try:
            if model_type == "keras":
                model = KerasModel(model_path)
            elif model_type == "tflite":
                model = TFLiteModel(model_path)
            elif model_type == "xgboost":
                model = XGBoostModel(model_path)
            else:
                print(f"❌ Unknown model type: {model_type}")
                continue
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            continue
        
        # Benchmark
        is_xgboost = (model_type == "xgboost")
        result = benchmark_model(model, dataset, model_name, is_xgboost=is_xgboost)
        result["model_size_mb"] = model_size_mb
        result["model_path"] = model_path
        results.append(result)
    
    # Print comparison table
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Model':<25} {'Size (MB)':<12} {'Accuracy':<12} {'Avg Time (ms)':<15} {'Std (ms)':<12}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['model_name']:<25} {r['model_size_mb']:>10.2f}   {r['accuracy']:>9.2f}%   {r['avg_inference_ms']:>13.2f}   {r['std_inference_ms']:>10.2f}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(dataset),
        "test_percent": TEST_SAMPLE_PERCENT * 100,
        "results": results
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {RESULTS_FILE}")
    
    # Recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    
    # Find best accuracy
    best_acc = max(results, key=lambda x: x['accuracy'])
    # Find fastest
    fastest = min(results, key=lambda x: x['avg_inference_ms'])
    # Find smallest
    smallest = min(results, key=lambda x: x['model_size_mb'])
    
    print(f"🎯 Best Accuracy:  {best_acc['model_name']} ({best_acc['accuracy']:.2f}%)")
    print(f"⚡ Fastest:        {fastest['model_name']} ({fastest['avg_inference_ms']:.2f} ms)")
    print(f"📦 Smallest:       {smallest['model_name']} ({smallest['model_size_mb']:.2f} MB)")
    
    # Balanced recommendation
    print(f"\n💡 For Production (Render free tier):")
    for r in results:
        score = (r['accuracy'] / 100) * 0.7 + (1 / r['avg_inference_ms']) * 1000 * 0.3
        r['score'] = score
    
    best_balanced = max(results, key=lambda x: x['score'])
    print(f"   Recommended: {best_balanced['model_name']}")
    print(f"   - Accuracy: {best_balanced['accuracy']:.2f}%")
    print(f"   - Speed: {best_balanced['avg_inference_ms']:.2f} ms")
    print(f"   - Size: {best_balanced['model_size_mb']:.2f} MB")

if __name__ == "__main__":
    main()
