import os
import torch
import torchaudio
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import RawNet
from config import SAMPLES_PER_TRACK, SAMPLE_RATE, DEVICE, MODEL_DIR

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for the React Frontend

# --- Load Model ---
# We assume the model is saved at the path defined in your main.py/config.py
MODEL_PATH = os.path.join(MODEL_DIR, "raw_tf_net_best.pth")
model = None

def load_model():
    global model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        # Initialize architecture
        model = RawNet(d_args={}).to(DEVICE)
        
        # Load weights
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("Model loaded successfully.")
        else:
            print(f"WARNING: Model file not found at {MODEL_PATH}. Server will fail on inference.")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    model.eval()

# --- Preprocessing Logic ---
# Replicating logic from dataset.py to ensure consistency
# --- Preprocessing Logic ---
def preprocess_audio(audio_file):
    try:
        # 1. Read audio
        # sf.read returns data as (samples, channels) for multi-channel
        audio_np, sr = sf.read(audio_file, dtype='float32')
        
        # 2. Force Stereo to Mono
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)
        
        # Convert to Tensor (shape: [1, Time])
        waveform = torch.from_numpy(audio_np)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # 3. Resample if necessary (CRITICAL for model accuracy)
        if sr != SAMPLE_RATE:
            print(f"Resampling from {sr}Hz to {SAMPLE_RATE}Hz...")
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
            
        # 4. Pad/Truncate
        target_length = SAMPLES_PER_TRACK
        _, w_len = waveform.shape
        
        if w_len < target_length:
            padding = target_length - w_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif w_len > target_length:
            waveform = waveform[:, :target_length]
            
        return waveform.unsqueeze(0) # Add batch dimension -> (1, 1, 64000)

    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 1. Preprocess
    input_tensor = preprocess_audio(file)
    if input_tensor is None:
        return jsonify({"error": "Failed to process audio file"}), 400

    # 2. Inference
    try:
        with torch.no_grad():
            input_tensor = input_tensor.to(DEVICE)
            outputs = model(input_tensor)
            
            # Apply Softmax
            probs = torch.softmax(outputs, dim=1)
            
            # Your dataset.py has LABEL_MAP = {"bonafide": 0, "spoof": 1}
            # So index 0 is Real, index 1 is Fake.
            bonafide_score = probs[0, 0].item()
            spoof_score = probs[0, 1].item()

            # Decision Logic
            # Using a high confidence threshold for security (e.g., 0.5 or 0.9)
            # You can tune this based on the EER threshold found in evaluation.py
            THRESHOLD = 0.50 
            
            is_bonafide = bonafide_score > THRESHOLD
            
            result = {
                "access_granted": bool(is_bonafide),
                "label": "Bonafide" if is_bonafide else "Spoof",
                "confidence": float(bonafide_score),
                "spoof_probability": float(spoof_score)
            }
            
            return jsonify(result)

    except Exception as e:
        print(f"Inference error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running", "device": str(DEVICE)})

if __name__ == '__main__':
    load_model()
    # Run on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)