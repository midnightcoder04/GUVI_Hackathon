"""
Benchmark Ensemble Model Prediction
Combines best_model_general.h5 and best_model_tts.h5 using ensemble averaging.

Ensemble strategies tested:
- Average: Mean of both model predictions
- Weighted Average: Weighted combination based on individual model performance
- Voting: Majority vote (tie-breaker: average)

Measures:
- Accuracy
- Inference time (mean, std)
- CPU usage

Usage:
    python benchmark_3.py
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

# Configuration
SAMPLE_RATE = 16000
N_LFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
TARGET_TIME_STEPS = 312
MAX_DURATION = 10.0
TEST_SAMPLE_PERCENT = 1  # 1 = 100% of test_cases

# Paths
HUMAN_DIR = "test_cases/real"
AI_DIR = "test_cases/spoof"
RESULTS_FILE = "benchmark_3_results.json"

# Model paths
MODEL_GENERAL = "model/best_model_general.h5"
MODEL_TTS = "model/best_model_tts.h5"

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

# ============== Model Architecture ==============
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

# ============== Ensemble Model ==============
class EnsembleModel:
    def __init__(self, model1_path, model2_path, strategy='average', weights=None):
        """
        Ensemble model combining two Keras models.
        
        Args:
            model1_path: Path to first model
            model2_path: Path to second model
            strategy: 'average', 'weighted', or 'voting'
            weights: List of [weight1, weight2] for weighted strategy (must sum to 1.0)
        """
        self.model1 = KerasModel(model1_path)
        self.model2 = KerasModel(model2_path)
        self.strategy = strategy
        
        if strategy == 'weighted':
            if weights is None:
                # Default weights based on typical performance
                self.weights = [0.5, 0.5]
            else:
                assert len(weights) == 2, "Weights must be a list of 2 values"
                assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
                self.weights = weights
        
    def predict(self, features):
        """Make ensemble prediction"""
        pred1 = float(self.model1.predict(features))
        pred2 = float(self.model2.predict(features))
        
        if self.strategy == 'average':
            # Simple average
            return (pred1 + pred2) / 2.0
        
        elif self.strategy == 'weighted':
            # Weighted average
            return pred1 * self.weights[0] + pred2 * self.weights[1]
        
        elif self.strategy == 'voting':
            # Hard voting (with average as tie-breaker)
            vote1 = 1 if pred1 >= 0.5 else 0
            vote2 = 1 if pred2 >= 0.5 else 0
            
            if vote1 == vote2:
                # Agreement: return average confidence
                return (pred1 + pred2) / 2.0
            else:
                # Disagreement: use average as tie-breaker
                return (pred1 + pred2) / 2.0
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def predict_detailed(self, features):
        """Make prediction with detailed breakdown"""
        pred1 = float(self.model1.predict(features))
        pred2 = float(self.model2.predict(features))
        ensemble_pred = self.predict(features)
        
        return {
            'model1': pred1,
            'model2': pred2,
            'ensemble': ensemble_pred
        }

# ============== Dataset Loading ==============
def load_dataset():
    """Load samples from human and AI directories"""
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
    
    # Sample based on TEST_SAMPLE_PERCENT
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
        if features is not None:
            dataset.append((features, 0, str(path)))  # 0 = HUMAN
        if (i + 1) % 10 == 0 or i == len(human_sample) - 1:
            print(f"  {i+1}/{len(human_sample)} processed")
    
    print("\nExtracting features from AI samples...")
    for i, path in enumerate(ai_sample):
        features = extract_features(str(path))
        if features is not None:
            dataset.append((features, 1, str(path)))  # 1 = AI
        if (i + 1) % 10 == 0 or i == len(ai_sample) - 1:
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
def benchmark_ensemble(model, dataset, model_name, detailed=False):
    """Benchmark an ensemble model"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    predictions = []
    ground_truth = []
    inference_times = []
    error_files = []
    detailed_predictions = []
    
    process = psutil.Process(os.getpid())
    cpu_usage_samples = []
    
    for features, label, path in dataset:
        # Warm up
        _ = model.predict(features)
        
        # Measure inference time
        cpu_before = process.cpu_percent()
        start = time.perf_counter()
        
        if detailed:
            pred_details = model.predict_detailed(features)
            pred = pred_details['ensemble']
            detailed_predictions.append({
                'file': os.path.basename(path),
                'model1_pred': pred_details['model1'],
                'model2_pred': pred_details['model2'],
                'ensemble_pred': pred_details['ensemble'],
                'actual_label': label,
                'language': get_language_from_filename(path)
            })
        else:
            pred = model.predict(features)
        
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
            # Calculate confidence score (distance from decision boundary)
            confidence = float(pred) if pred_class == 1 else float(1 - pred)
            error_files.append({
                'file': os.path.basename(path),
                'actual': 'AI' if label == 1 else 'Human',
                'predicted': 'AI' if pred_class == 1 else 'Human',
                'confidence': confidence,
                'raw_prediction': float(pred),
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
    for features, label, path in dataset:
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
    
    result = {
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
        "error_details": error_files[:20]  # First 20 errors
    }
    
    if detailed:
        result["detailed_predictions"] = detailed_predictions[:50]  # First 50 for inspection
    
    return result

def main():
    print("="*60)
    print("Ensemble Model Benchmark Suite")
    print("="*60)
    
    # Load dataset
    dataset = load_dataset()
    
    if len(dataset) == 0:
        print("❌ No samples found in test directories")
        return
    
    # Ensemble configurations to test
    ensemble_configs = [
        {
            'strategy': 'average',
            'weights': None,
            'name': 'Ensemble (Average)',
            'description': 'Simple average of both models'
        },
        {
            'strategy': 'weighted',
            'weights': [0.6, 0.4],  # Favor general model slightly
            'name': 'Ensemble (Weighted 60-40)',
            'description': 'Weighted: 60% general, 40% TTS'
        },
        {
            'strategy': 'weighted',
            'weights': [0.5, 0.5],
            'name': 'Ensemble (Weighted 50-50)',
            'description': 'Weighted: 50% general, 50% TTS'
        },
        {
            'strategy': 'weighted',
            'weights': [0.4, 0.6],  # Favor TTS model slightly
            'name': 'Ensemble (Weighted 40-60)',
            'description': 'Weighted: 40% general, 60% TTS'
        },
        {
            'strategy': 'voting',
            'weights': None,
            'name': 'Ensemble (Voting)',
            'description': 'Hard voting with average tie-breaker'
        }
    ]
    
    results = []
    
    for config in ensemble_configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        try:
            # Create ensemble model
            model = EnsembleModel(
                MODEL_GENERAL,
                MODEL_TTS,
                strategy=config['strategy'],
                weights=config['weights']
            )
            
            # Benchmark (detailed only for first config)
            detailed = (config == ensemble_configs[0])
            result = benchmark_ensemble(model, dataset, config['name'], detailed=detailed)
            result['strategy'] = config['strategy']
            result['weights'] = config['weights']
            result['description'] = config['description']
            results.append(result)
            
        except Exception as e:
            print(f"❌ Failed to benchmark {config['name']}: {e}")
            continue
    
    # Print comparison table
    print(f"\n{'='*60}")
    print("ENSEMBLE BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Strategy':<30} {'Accuracy':<12} {'Avg Time (ms)':<15} {'Errors':<10}")
    print("-" * 75)
    
    for r in results:
        print(f"{r['model_name']:<30} {r['accuracy']:>9.2f}%   {r['avg_inference_ms']:>13.2f}   {r['total_errors']:>8}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(dataset),
        "test_percent": TEST_SAMPLE_PERCENT * 100,
        "model1": MODEL_GENERAL,
        "model2": MODEL_TTS,
        "results": results
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {RESULTS_FILE}")
    
    # Recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    
    if results:
        # Find best accuracy
        best_acc = max(results, key=lambda x: x['accuracy'])
        # Find fastest
        fastest = min(results, key=lambda x: x['avg_inference_ms'])
        
        print(f"🎯 Best Accuracy:  {best_acc['model_name']} ({best_acc['accuracy']:.2f}%)")
        print(f"⚡ Fastest:        {fastest['model_name']} ({fastest['avg_inference_ms']:.2f} ms)")
        
        # Overall recommendation
        print(f"\n💡 Recommendation for Production:")
        if best_acc['accuracy'] > 80:
            print(f"   Use: {best_acc['model_name']}")
            print(f"   - Accuracy: {best_acc['accuracy']:.2f}%")
            print(f"   - Speed: {best_acc['avg_inference_ms']:.2f} ms")
            print(f"   - Strategy: {best_acc['strategy']}")
            if best_acc['weights']:
                print(f"   - Weights: General={best_acc['weights'][0]}, TTS={best_acc['weights'][1]}")
        else:
            print(f"   Ensemble may not provide significant improvement.")
            print(f"   Consider using individual models instead.")

if __name__ == "__main__":
    main()
