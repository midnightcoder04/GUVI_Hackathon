"""
Benchmark LFCC-LCNN Model on audio dataset.
Tests on random samples from specified folder paths.

Measures:
- Accuracy
- Model size
- Inference time (mean, std)
- CPU usage
- Per-language breakdown

Usage:
    python benchmark_2.py
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import librosa
from pathlib import Path
import random
import json
from datetime import datetime
import psutil
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_RATE = 16000
DURATION = 5.0
MAX_LENGTH = int(SAMPLE_RATE * DURATION)

# LFCC parameters (matching the notebook)
N_FILTER = 20
N_LFCC = 60  # 20 filters * 3 (with deltas)
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
TARGET_TIME_STEPS = 500  # Fixed time steps for model input

# Test configuration
TEST_SAMPLE_PERCENT = 0.1  # 5% of data

# Paths
HUMAN_DIR = "new_dev/Eleven/real"
AI_DIR = "new_dev/Eleven/spoof"
MODEL_PATH = "model/lfcc_model_best.h5"
RESULTS_FILE = "benchmark_2_results.json"


# ============================================================================
# CUSTOM LAYERS (Required for loading the model)
# ============================================================================

class MaxFeatureMap(layers.Layer):
    """
    Max-Feature-Map (MFM) activation from the original LCNN paper.
    """
    
    def __init__(self, **kwargs):
        super(MaxFeatureMap, self).__init__(**kwargs)
    
    def call(self, inputs):
        split = tf.split(inputs, num_or_size_splits=2, axis=-1)
        return tf.maximum(split[0], split[1])
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] // 2,)


class ResidualMFMBlock(layers.Layer):
    """
    Residual block with MFM activation.
    """
    
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(ResidualMFMBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        self.conv1 = layers.Conv2D(
            filters * 2, kernel_size, padding='same',
            kernel_initializer='he_normal'
        )
        self.bn1 = layers.BatchNormalization()
        self.mfm1 = MaxFeatureMap()
        
        self.conv2 = layers.Conv2D(
            filters * 2, kernel_size, padding='same',
            kernel_initializer='he_normal'
        )
        self.bn2 = layers.BatchNormalization()
        self.mfm2 = MaxFeatureMap()
        
        self.projection = None
    
    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.projection = layers.Conv2D(
                self.filters, 1, padding='same',
                kernel_initializer='he_normal'
            )
        super(ResidualMFMBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.mfm1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.mfm2(x)
        
        shortcut = inputs
        if self.projection is not None:
            shortcut = self.projection(inputs)
        
        return x + shortcut
    
    def get_config(self):
        config = super(ResidualMFMBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ============================================================================
# FEATURE EXTRACTION (Matching the notebook)
# ============================================================================

def extract_lfcc_features(audio_path):
    """
    Extract LFCC features matching the notebook's FeatureExtractor.
    Returns features in shape (time_steps, n_features) = (500, 60)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Handle NaN/Inf
        if not np.isfinite(y).all():
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize
        max_val = np.abs(y).max()
        if max_val > 0:
            y = y / max_val
        
        # Pad or truncate to MAX_LENGTH
        if len(y) > MAX_LENGTH:
            y = y[:MAX_LENGTH]
        else:
            y = np.pad(y, (0, MAX_LENGTH - len(y)))
        
        # Compute STFT
        stft = librosa.stft(
            y,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH
        )
        magnitude = np.abs(stft)
        
        if np.max(magnitude) == 0:
            n_frames = int(len(y) / HOP_LENGTH)
            return np.zeros((TARGET_TIME_STEPS, N_LFCC), dtype=np.float32)
        
        # Linear filterbank
        linear_filters = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=N_FILTER,
            fmin=0,
            fmax=SAMPLE_RATE // 2,
            htk=True,
            norm=None
        )
        
        # Apply filterbank
        filtered = np.dot(linear_filters, magnitude)
        
        # Log compression
        log_filtered = np.log(filtered + 1e-8)
        
        if not np.isfinite(log_filtered).all():
            log_filtered = np.nan_to_num(log_filtered, nan=-8.0, posinf=0.0, neginf=-8.0)
        
        # DCT to get cepstral coefficients
        lfcc = librosa.feature.mfcc(
            S=log_filtered,
            n_mfcc=N_FILTER,
            dct_type=2,
            norm='ortho'
        )
        
        # Add delta and delta-delta features
        lfcc_delta = librosa.feature.delta(lfcc)
        lfcc_delta2 = librosa.feature.delta(lfcc, order=2)
        
        # Check deltas for NaN/Inf
        if not np.isfinite(lfcc_delta).all():
            lfcc_delta = np.nan_to_num(lfcc_delta, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(lfcc_delta2).all():
            lfcc_delta2 = np.nan_to_num(lfcc_delta2, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Concatenate all features: (60, time_steps)
        lfcc_features = np.concatenate([lfcc, lfcc_delta, lfcc_delta2], axis=0)
        
        # Final safety check
        if not np.isfinite(lfcc_features).all():
            lfcc_features = np.nan_to_num(lfcc_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Transpose to (time_steps, 60)
        lfcc_features = lfcc_features.T.astype(np.float32)
        
        # Pad or truncate to TARGET_TIME_STEPS
        if lfcc_features.shape[0] > TARGET_TIME_STEPS:
            lfcc_features = lfcc_features[:TARGET_TIME_STEPS, :]
        elif lfcc_features.shape[0] < TARGET_TIME_STEPS:
            pad_len = TARGET_TIME_STEPS - lfcc_features.shape[0]
            lfcc_features = np.pad(lfcc_features, [[0, pad_len], [0, 0]])
        
        return lfcc_features
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


# ============================================================================
# MODEL LOADER
# ============================================================================

class LFCCModel:
    """LFCC-LCNN Model wrapper"""
    
    def __init__(self, model_path):
        print(f"Loading LFCC-LCNN model from {model_path}...")
        
        # Custom objects for model loading
        custom_objects = {
            'MaxFeatureMap': MaxFeatureMap,
            'ResidualMFMBlock': ResidualMFMBlock
        }
        
        self.model = keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        self.model.trainable = False
        print(f"  Model input shape: {self.model.input_shape}")
        print(f"  Model output shape: {self.model.output_shape}")
    
    def predict(self, features):
        """
        Predict on single sample.
        
        Args:
            features: Shape (500, 60) LFCC features
            
        Returns:
            Prediction probability for AI class
        """
        # Add batch dimension: (1, 500, 60)
        features_batch = features[np.newaxis, ...]
        
        # Model outputs [fake_prob, real_prob] (softmax)
        pred = self.model(features_batch, training=False)
        
        # Return probability of AI/Fake (class 0 in notebook is Fake)
        # The notebook uses: 0 = Fake, 1 = Real
        return pred[0][0].numpy()  # Probability of Fake/AI


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(human_dir=HUMAN_DIR, ai_dir=AI_DIR, sample_percent=TEST_SAMPLE_PERCENT):
    """Load random samples from human and AI directories"""
    print(f"\n{'='*60}")
    print("Loading Dataset")
    print(f"{'='*60}")
    
    human_files = []
    ai_files = []
    
    # Scan human clips
    if os.path.exists(human_dir):
        for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
            human_files.extend(list(Path(human_dir).rglob(ext)))
    
    # Scan AI clips
    if os.path.exists(ai_dir):
        for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
            ai_files.extend(list(Path(ai_dir).rglob(ext)))
    
    print(f"Found: {len(human_files)} human, {len(ai_files)} AI samples")
    
    # Random sample
    num_human = max(1, int(len(human_files) * sample_percent))
    num_ai = max(1, int(len(ai_files) * sample_percent))
    
    human_sample = random.sample(human_files, min(num_human, len(human_files)))
    ai_sample = random.sample(ai_files, min(num_ai, len(ai_files)))
    
    print(f"Testing on: {len(human_sample)} human, {len(ai_sample)} AI samples ({sample_percent*100:.1f}%)")
    
    # Extract features
    dataset = []
    
    print("\nExtracting features from HUMAN samples...")
    for i, path in enumerate(human_sample):
        features = extract_lfcc_features(str(path))
        if features is not None:
            # Label 0 = Human (Real), 1 = AI (Fake)
            # But in the notebook: 0 = Fake, 1 = Real
            # So we use: 0 = Human (Real in notebook), predict gives AI prob
            dataset.append((features, 0, str(path)))  # 0 = HUMAN
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(human_sample)} processed")
    
    print("\nExtracting features from AI samples...")
    for i, path in enumerate(ai_sample):
        features = extract_lfcc_features(str(path))
        if features is not None:
            dataset.append((features, 1, str(path)))  # 1 = AI
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(ai_sample)} processed")
    
    print(f"\nTotal valid samples: {len(dataset)}")
    return dataset


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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


# ============================================================================
# BENCHMARKING
# ============================================================================

def benchmark_model(model, dataset, model_name):
    """Benchmark the LFCC-LCNN model"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    predictions = []
    ground_truth = []
    inference_times = []
    error_files = []
    
    process = psutil.Process(os.getpid())
    cpu_usage_samples = []
    
    for features, label, path in dataset:
        # Warm up
        _ = model.predict(features)
        
        # Measure inference time
        cpu_before = process.cpu_percent()
        start = time.perf_counter()
        pred = model.predict(features)
        end = time.perf_counter()
        cpu_after = process.cpu_percent()
        
        inference_time = (end - start) * 1000  # ms
        inference_times.append(inference_time)
        cpu_usage_samples.append((cpu_after + cpu_before) / 2)
        
        # Classification (threshold = 0.5)
        # pred is probability of Fake/AI class
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
    
    # Calculate language-specific stats
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
        "error_details": error_files[:20]  # First 20 errors
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("LFCC-LCNN Model Benchmark")
    print("=" * 60)
    
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    # Load dataset
    dataset = load_dataset()
    
    if len(dataset) == 0:
        print("No samples found in build/clips or build/clips_AI")
        return
    
    # Load model
    try:
        model = LFCCModel(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Get model size
    model_size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
    
    # Benchmark
    result = benchmark_model(model, dataset, "LFCC-LCNN (MFM Residual)")
    result["model_size_mb"] = model_size_mb
    result["model_path"] = MODEL_PATH
    
    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Model':<30} {'Size (MB)':<12} {'Accuracy':<12} {'Avg Time (ms)':<15}")
    print("-" * 80)
    print(f"{result['model_name']:<30} {result['model_size_mb']:>10.2f}   {result['accuracy']:>9.2f}%   {result['avg_inference_ms']:>13.2f}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(dataset),
        "test_percent": TEST_SAMPLE_PERCENT * 100,
        "config": {
            "sample_rate": SAMPLE_RATE,
            "duration": DURATION,
            "n_lfcc": N_LFCC,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "time_steps": TARGET_TIME_STEPS
        },
        "results": [result]
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_FILE}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: LFCC-LCNN with MFM + Residual")
    # print(f"  Size: {model_size_mb:.2f} MB")
    print(f"  Accuracy: {result['accuracy']:.2f}%")
    # print(f"  Avg Inference: {result['avg_inference_ms']:.2f} ms")
    print(f"  Total Samples: {result['total_samples']}")
    print(f"  Errors: {result['total_errors']}")


if __name__ == "__main__":
    main()
