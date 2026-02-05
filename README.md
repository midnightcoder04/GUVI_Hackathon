# AI-Generated Voice Detection API

A REST API that detects whether a voice sample is AI-generated or Human, supporting Tamil, English, Hindi, Malayalam, and Telugu.

## Quick Deploy to Render (Free)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/voice-detection-api.git
git push -u origin main
```

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com) and sign up (free)
2. Click **New** → **Web Service**
3. Connect your GitHub repository
4. Render will auto-detect the Dockerfile
5. Set **Plan** to **Free**
6. Add Environment Variable:
   - Key: `API_KEY`
   - Value: Your secret API key (e.g., `sk_your_secret_key_here`)
7. Click **Create Web Service**

Your API will be live at: `https://your-app-name.onrender.com`

## API Usage

### Endpoint
```
POST https://your-app-name.onrender.com/api/voice-detection
```

### Headers
```
Content-Type: application/json
x-api-key: YOUR_API_KEY
```

### Request Body
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_ENCODED_AUDIO_HERE"
}
```

### cURL Example
```bash
curl -X POST https://your-app-name.onrender.com/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_your_secret_key_here" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
  }'
```

### Success Response
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Strong indicators of synthetic speech: unnatural pitch consistency..."
}
```

### Error Response
```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

## Local Development

### Run with Docker
```bash
docker build -t voice-detection .
docker run -p 8000:8000 -e API_KEY=your_secret_key voice-detection
```

### Run without Docker
```bash
pip install -r requirements.txt
export API_KEY=your_secret_key
python app.py
```

Then visit: http://localhost:8000

## Project Structure
```
├── app.py                 # FastAPI application
├── model/
│   ├── model.h5           # Trained CNN model
│   └── normalization_params.npz  # Feature normalization
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── render.yaml           # Render deployment config
└── README.md
```

---

# Model Architecture


def build_cnn_model(input_shape):
    """
    CNN architecture optimized for audio classification.
    """
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


# Preprocessing

def extract_lfcc(audio_path, sr=SAMPLE_RATE, n_lfcc=N_LFCC, 
                 n_fft=N_FFT, hop_length=HOP_LENGTH, max_duration=MAX_DURATION):
    """
    Extract LFCC features from audio file.
    Returns a fixed-size feature matrix.
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=max_duration)
        
        # Compute LFCC
        # LFCC is like MFCC but uses linear frequency scale
        # We'll use MFCC with linear frequency spacing
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
        print(f"Error processing {audio_path}: {e}")
        # Return zero array in case of error
        target_length = int(max_duration * sr / hop_length)
        return np.zeros((n_lfcc, target_length))







# Problem Statement 1
AI-Generated Voice Detection (Tamil, English, Hindi, Malayalam, Telugu)
1. Introduction
AI systems can now generate very realistic human-like voices. Because of this, it is difficult to identify whether a voice recording was spoken by a real human or generated by an AI system.
In this problem, students must build an API-based solution that detects whether a given voice sample is AI-generated or Human, across five supported languages.
2. Supported Languages (Fixed)
Your system must support only these five languages:
Tamil
English
Hindi
Malayalam
Telugu
Each request will contain one audio file in one of the above languages.
3. What You Need to Build
You must design and deploy a REST API that:
Accepts one MP3 audio file at a time
Audio will be sent only as Base64
Analyzes the voice
Returns whether the voice is:
AI_GENERATED
HUMAN
Responds in JSON format
Is protected using an API Key


4. Input Rules
Audio format: MP3
Input type: Base64 encoded
One audio per request
Audio must not be modified
5. API Authentication
Your API must validate an API Key sent in request headers.
API Key Header Format
x-api-key: YOUR_SECRET_API_KEY
Requests without a valid API key must be rejected.
6. API Request (cURL Example)
Endpoint Example
POST https://your-domain.com/api/voice-detection
cURL Request Example
curl -X POST https://your-domain.com/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
  }'

7. Request Body Fields
Field
Description
language
Tamil / English / Hindi / Malayalam / Telugu
audioFormat
Always mp3
audioBase64
Base64-encoded MP3 audio

8. API Response Body (Success)
Example Response
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91, // out of 1.0
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
9. Response Field Explanation
Field
Meaning
status
success or error
language
Language of the audio
classification
AI_GENERATED or HUMAN
confidenceScore
Value between 0.0 and 1.0
explanation
Short reason for the decision

10. Classification Rules (Strict)
Only one classification field is required:
AI_GENERATED → Voice created using AI or synthetic systems
HUMAN → Voice spoken by a real human
👉 voiceSource is removed because it is logically the same as classification.
11. Error Response Example
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
12. Evaluation Process
System sends one Base64 MP3 per request
Language will be one of the 5 supported languages
Your API analyzes the voice
JSON response is returned
Multiple requests are made for evaluation

13. Evaluation Criteria
Participants will be evaluated on:
🎯 Accuracy of AI vs Human detection
🌍 Consistency across all 5 languages
📦 Correct request & response format
⚡ API reliability and response time
🧠 Quality of explanation
14. Rules & Constraints
❌ Hard-coding results is strictly prohibited
❌ Misuse of data leads to disqualification
⚠️ External detection APIs may be restricted
✅ Ethical and transparent AI usage is mandatory
15. One-Line Summary
Build a secure REST API that accepts one Base64-encoded MP3 voice in Tamil, English, Hindi, Malayalam, or Telugu and correctly identifies whether it is AI-generated or Human.
