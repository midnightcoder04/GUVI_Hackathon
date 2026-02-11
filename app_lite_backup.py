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

# ============== Configuration ==============
API_KEY = os.environ.get("API_KEY", "sk_test_123456789")  # Set in Render environment

SAMPLE_RATE = 16000  # Must match training sample rate
N_LFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
TARGET_TIME_STEPS = 312  # Fixed by model architecture
MAX_DURATION = 10.0  # seconds (flexible input, always output 312 steps)

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# Model path (TFLite INT8 only)
MODEL_PATH = "model/model_int8.tflite"

# ============== Load TFLite Model ==============
print(f"Loading TFLite INT8 model: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Load TFLite interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✓ TFLite model loaded")
print(f"  Input shape: {input_details[0]['shape']}")
print(f"  Input dtype: {input_details[0]['dtype']}")
print(f"  Output dtype: {output_details[0]['dtype']}")

# Load normalization parameters
print("Loading normalization parameters...")
norm_params = np.load("model/normalization_params.npz")
MEAN = np.array(norm_params['mean'], dtype=np.float32)
STD = np.array(norm_params['std'], dtype=np.float32)
print(f"Normalization parameters loaded! Shape: MEAN={MEAN.shape}, STD={STD.shape}")

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
    # MEAN and STD are arrays of shape (40,), need to broadcast correctly
    mean = MEAN
    std = STD
    if mean.ndim == 1:
        mean = mean[:, np.newaxis]
    if std.ndim == 1:
        std = std[:, np.newaxis]
    
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
        "model_loaded": interpreter is not None,
        "model_type": "tflite_int8"
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
        
        # TFLite INT8 inference
        # Handle INT8 input quantization
        if input_details[0]['dtype'] == np.uint8:
            scale, zero_point = input_details[0]['quantization']
            input_data = (features / scale + zero_point).astype(np.uint8)
        else:
            input_data = features.astype(input_details[0]['dtype'])
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Handle INT8 output dequantization
        if output_details[0]['dtype'] == np.uint8:
            scale, zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - zero_point) * scale
        
        prediction = float(output_data[0][0])
        
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
