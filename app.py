"""
AI-Generated Voice Detection API
FastAPI application for detecting AI-generated vs Human voices
"""

import os
import base64
import tempfile
import numpy as np
import librosa
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

# Use TensorFlow's built-in Keras (compatible with TF 2.13)
import tensorflow as tf
keras = tf.keras

# ============== Configuration ==============
API_KEY = os.environ.get("API_KEY", "sk_test_123456789")  # Set in Render environment
SAMPLE_RATE = 22050
N_LFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
MAX_DURATION = 10.0  # seconds

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# ============== Load Model ==============
print("Loading model...")
model = keras.models.load_model("model/model.h5")
print("Model loaded successfully!")

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
def extract_lfcc(audio_path, sr=SAMPLE_RATE, n_lfcc=N_LFCC, 
                 n_fft=N_FFT, hop_length=HOP_LENGTH, max_duration=MAX_DURATION):
    """
    Extract LFCC features from audio file.
    Returns a fixed-size feature matrix.
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=max_duration)
        
        # Compute LFCC (using MFCC with linear frequency spacing)
        lfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=n_lfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Standardize to fixed length (pad or truncate)
        target_length = int(max_duration * sr / hop_length)
        
        if lfcc.shape[1] < target_length:
            # Pad
            pad_width = target_length - lfcc.shape[1]
            lfcc = np.pad(lfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate
            lfcc = lfcc[:, :target_length]
        
        return lfcc
    except Exception as e:
        print(f"Error processing audio: {e}")
        target_length = int(max_duration * sr / hop_length)
        return np.zeros((n_lfcc, target_length))

def preprocess_audio(audio_path):
    """
    Extract features and prepare for model input.
    """
    # Extract LFCC features
    features = extract_lfcc(audio_path)
    
    # Normalize using saved parameters
    if MEAN is not None and STD is not None:
        features = (features - MEAN) / (STD + 1e-8)
    
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
    return {"status": "healthy", "model_loaded": model is not None}

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
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Preprocess audio
            features = preprocess_audio(tmp_path)
            
            # Run inference
            prediction = model.predict(features, verbose=0)[0][0]
            
            # Determine classification
            # Model output: 0 = HUMAN, 1 = AI_GENERATED (or adjust based on your training)
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
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
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
