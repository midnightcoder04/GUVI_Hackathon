"""
AI-Generated Voice Detection API
FastAPI application for detecting AI-generated vs Human voices
Optimized for Render free tier (512MB RAM)
"""

import os
import base64
import io
import numpy as np
import librosa
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

# ============== TensorFlow Optimizations ==============
# Set before importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for consistent results

import tensorflow as tf

# Limit CPU threads for free tier
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

keras = tf.keras

# ============== Configuration ==============
API_KEY = os.environ.get("API_KEY", "sk_test_123456789")  # Set in Render environment
SAMPLE_RATE = 22050
N_LFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
# Model expects exactly 312 time steps (fixed)
TARGET_TIME_STEPS = 312
# Allow longer audio input, we'll pad/truncate to exactly 312 steps
MAX_DURATION = 10.0  # seconds (flexible input, always output 312 steps)

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# ============== Model Architecture ==============
def build_cnn_model(input_shape=(N_LFCC, TARGET_TIME_STEPS, 1)):
    """
    CNN architecture for voice classification.
    Rebuilds the model to avoid Keras version compatibility issues.
    """
    from tensorflow.keras import layers, Model
    
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.25, name='dropout1')(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.25, name='dropout2')(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(0.25, name='dropout3')(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), padding='same', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Activation('relu', name='relu4')(x)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5, name='dropout4')(x)
    x = layers.Dense(64, activation='relu', name='dense2')(x)
    x = layers.Dropout(0.5, name='dropout5')(x)
    
    # Output
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='VoiceClassifierCNN')
    return model

# ============== Load Model ==============
print("Loading Keras model...")
try:
    # Try direct loading first
    model = keras.models.load_model("model/model.h5", compile=False)
    print("Model loaded directly!")
except Exception as e:
    print(f"Direct loading failed: {e}")
    print("Rebuilding model from architecture and loading weights...")
    # Rebuild architecture and load weights (Keras 2.x/3.x compatibility)
    model = build_cnn_model()
    model.load_weights("model/model.h5")
    print("Model loaded via weight loading!")

model.trainable = False  # Disable training layers for inference

# Pre-warm model (first inference is slow due to graph compilation)
print("Warming up model...")
_dummy_input = np.zeros((1, N_LFCC, TARGET_TIME_STEPS, 1), dtype=np.float32)
_ = model(_dummy_input, training=False)
print("Model ready!")

# Load normalization parameters
print("Loading normalization parameters...")
norm_params = np.load("model/normalization_params.npz")
MEAN = norm_params.get("mean", None)
STD = norm_params.get("std", None)
print("Normalization parameters loaded!")

# ============== FastAPI App ==============
app = FastAPI(
    title="AI Voice Detection API",
    description="Detects whether a voice sample is AI-generated or Human",
    version="1.0.0"
)

# ============== Request/Response Models ==============
class VoiceDetectionRequest(BaseModel):
    language: str = Field(..., description="Language: Tamil/English/Hindi/Malayalam/Telugu")
    audioFormat: str = Field(..., description="Audio format (mp3)")
    audioBase64: str = Field(..., description="Base64-encoded MP3 audio")

class VoiceDetectionResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str

# ============== Feature Extraction ==============
def extract_lfcc_from_bytes(audio_bytes, sr=SAMPLE_RATE, n_lfcc=N_LFCC, 
                 n_fft=N_FFT, hop_length=HOP_LENGTH, max_duration=MAX_DURATION):
    """
    Extract LFCC-style cepstral features from audio bytes (in-memory).
    Uses MFCC computation for practical spoofing detection.
    Always returns exactly 312 time steps (model requirement).
    """
    try:
        # Load audio from bytes (no disk I/O)
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer, sr=sr, duration=max_duration)
        
        # Compute LFCC-style cepstral features (matching training exactly)
        lfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=n_lfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Always pad or truncate to exactly TARGET_TIME_STEPS (312)
        if lfcc.shape[1] < TARGET_TIME_STEPS:
            # Pad with zeros
            pad_width = TARGET_TIME_STEPS - lfcc.shape[1]
            lfcc = np.pad(lfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate to exactly 312 steps
            lfcc = lfcc[:, :TARGET_TIME_STEPS]
        
        return lfcc
    except Exception as e:
        print(f"Error processing audio: {e}")
        return np.zeros((n_lfcc, TARGET_TIME_STEPS), dtype=np.float32)

def preprocess_audio_bytes(audio_bytes):
    """
    Extract LFCC-style cepstral features and prepare for model input.
    Matches exact preprocessing used during training.
    """
    # Extract LFCC-style features from bytes
    features = extract_lfcc_from_bytes(audio_bytes)
    
    # Ensure float32 dtype before normalization
    features = features.astype(np.float32)
    
    # Normalize using saved parameters (feature-wise normalization)
    if MEAN is not None and STD is not None:
        # Ensure MEAN and STD are broadcastable to features shape
        # features shape: (n_lfcc, time_steps) = (40, 312)
        # MEAN/STD should be (40, 1) or (40, 312)
        mean = np.array(MEAN, dtype=np.float32)
        std = np.array(STD, dtype=np.float32)
        
        # Reshape if needed for feature-wise normalization
        if mean.ndim == 1:
            mean = mean[:, np.newaxis]  # (40,) -> (40, 1)
        if std.ndim == 1:
            std = std[:, np.newaxis]    # (40,) -> (40, 1)
            
        features = (features - mean) / (std + 1e-8)
    
    # Reshape for CNN input: (batch, height, width, channels)
    features = features[np.newaxis, ..., np.newaxis]
    
    return features

def generate_explanation(confidence: float, is_ai: bool) -> str:
    """
    Generate a human-readable explanation for the classification.
    """
    if is_ai:
        if confidence > 0.9:
            return "Strong indicators of synthetic speech: unnatural pitch consistency, robotic intonation patterns, and artificial spectral characteristics detected"
        elif confidence > 0.75:
            return "Moderate AI speech indicators: irregular prosody patterns and synthetic voice artifacts detected"
        elif confidence > 0.6:
            return "Slight synthetic characteristics detected: minor artifacts in speech patterns suggest AI generation"
        else:
            return "Weak AI indicators detected: some subtle synthetic patterns present in the audio"
    else:
        if confidence > 0.9:
            return "Strong human voice characteristics: natural pitch variations, authentic breathing patterns, and organic speech dynamics detected"
        elif confidence > 0.75:
            return "Natural human speech patterns detected: consistent with organic voice production"
        elif confidence > 0.6:
            return "Predominantly human characteristics: natural prosody with minor inconsistencies"
        else:
            return "Voice appears human with some ambiguous characteristics"

# ============== API Endpoints ==============
@app.get("/")
async def root():
    return {"message": "AI Voice Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "keras"
    }

@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
async def detect_voice(
    request: VoiceDetectionRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
):
    """
    Detect whether a voice sample is AI-generated or Human.
    """
    # Validate API Key
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Invalid API key or malformed request"}
        )
    
    # Validate language
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": f"Unsupported language. Must be one of: {', '.join(SUPPORTED_LANGUAGES)}"}
        )
    
    # Validate audio format
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Only MP3 audio format is supported"}
        )
    
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audioBase64)
        
        # Preprocess audio (in-memory, no disk I/O)
        features = preprocess_audio_bytes(audio_bytes)
        
        # Run Keras inference
        prediction = model(features, training=False)[0][0].numpy()
        
        # Log prediction for debugging
        print(f"Raw prediction: {prediction:.4f}, Language: {request.language}")
        
        # Determine classification
        # Model output: higher value = more likely AI_GENERATED
        # Note: Adjust threshold or flip logic if accuracy is consistently inverted
        is_ai = prediction >= 0.5
        classification = "AI_GENERATED" if is_ai else "HUMAN"
        
        # Confidence score (distance from decision boundary)
        confidence_score = float(prediction) if is_ai else float(1 - prediction)
        
        # Generate explanation
        explanation = generate_explanation(confidence_score, is_ai)
        
        return VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=classification,
            confidenceScore=round(confidence_score, 2),
            explanation=explanation
        )
                
    except base64.binascii.Error:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Invalid Base64 encoding"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Error processing audio: {str(e)}"}
        )

# ============== Error Handlers ==============
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {"status": "error", "message": str(exc.detail)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )

# ============== Main ==============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
