import os
import torch
import soundfile as sf
import numpy as np
import librosa
import pyloudnorm as pyln
import warnings

from torch.utils.data import Dataset
from config import DATASET_ROOT

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)

class ASVspoofDataset(Dataset):
    def __init__(self, df, target_length=64000):
        self.df = df
        self.target_length = target_length
        self.labels = torch.tensor(df['label'].values, dtype=torch.long)
        self.filenames = df['filename'].values
        self.subsets = df['subset'].values

        # --- LOAD SILERO VAD (Stage 1) ---
        # Loading from torchhub once to avoid overhead in __getitem__
        # Using onnx=False to minimize dependency issues, though onnx is faster.
        try:
            print("Loading Silero VAD for robust silence removal...")
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            (self.get_speech_timestamps, _, _, _, _) = utils
            self.vad_enabled = True
        except Exception as e:
            print(f"Warning: Failed to load Silero VAD ({e}). Fallback to Librosa trim.")
            self.vad_enabled = False

    def __len__(self):
        return len(self.df)

    def _apply_mu_law(self, x, mu=255):
        """
        Stage 3: Mu-Law Companding [cite: 50, 51]
        F(x) = sgn(x) * ln(1 + mu*|x|) / ln(1 + mu)
        Keeps data as float32 (continuous) rather than 8-bit int.
        """
        x_tensor = torch.as_tensor(x)
        return torch.sign(x_tensor) * torch.log1p(mu * torch.abs(x_tensor)) / np.log1p(mu)

    def _standardize_loudness(self, audio, sr):
        """
        Stage 2: EBU R128 Normalization [cite: 30]
        """
        try:
            # Create a meter for the specific sample rate
            meter = pyln.Meter(sr) 
            loudness = meter.integrated_loudness(audio)
            
            # Normalize to -23 LUFS (Broadcast standard)
            # If audio is effectively silent (-inf), skip normalization to avoid NaNs
            if not np.isinf(loudness):
                audio = pyln.normalize.loudness(audio, loudness, -23.0)
        except Exception:
            # Fallback if signal is too short for meter or other error
            pass
        return audio

    def _process_temporal_structure(self, audio, sr):
        """
        Stage 1: VAD and Audio Folding 
        """
        # 1. Voice Activity Detection (Silero)
        if self.vad_enabled and len(audio) > 512:
            try:
                # Convert to tensor for VAD
                wav_tensor = torch.from_numpy(audio).float()
                timestamps = self.get_speech_timestamps(wav_tensor, self.vad_model, sampling_rate=sr)
                
                if len(timestamps) > 0:
                    # Concatenate speech segments
                    speech_chunks = [audio[ts['start']:ts['end']] for ts in timestamps]
                    audio = np.concatenate(speech_chunks)
            except Exception:
                pass # Fallback to original audio on VAD failure

        # 2. Audio Folding (Looping) if too short
        if len(audio) < self.target_length:
            repeat_count = (self.target_length // len(audio)) + 1
            audio = np.tile(audio, repeat_count)
        
        # 3. Crop if too long
        if len(audio) > self.target_length:
            start = np.random.randint(0, len(audio) - self.target_length)
            audio = audio[start : start + self.target_length]
            
        return audio

    def extract_robust_features(self, y, sr):
        """
        Stage 4: Advanced Bio-Physical Features [cite: 160]
        1. TEO (Teager Energy Operator) stats
        2. PCEN Spectral Flux
        3. Unvoiced Segment Length (Proxy)
        """
        epsilon = 1e-10
        
        # --- Feature 1: Teager Energy Operator (TEO) [cite: 106] ---
        # TEO[n] = x[n]^2 - x[n-1]*x[n+1]
        # We compute mean and variance of Log-TEO
        teo = y[1:-1]**2 - y[:-2] * y[2:]
        # Add epsilon to avoid log(0) or negative inputs in log
        teo_log = np.log(np.abs(teo) + epsilon)
        
        teo_mean = np.mean(teo_log)
        teo_std = np.std(teo_log)

        # --- Feature 2: PCEN Spectral Flux [cite: 119, 163] ---
        # Compute Mel Spectrogram -> PCEN -> Onset Strength
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
        # Apply PCEN (Per-Channel Energy Normalization)
        S_pcen = librosa.pcen(S * (2**31)) # Scaling often helps PCEN stability
        # Calculate Spectral Flux (rate of change) on PCEN
        flux = librosa.onset.onset_strength(S=S_pcen, sr=sr)
        pcen_flux_mean = np.mean(flux)

        # --- Feature 3: Unvoiced Segment Length (Proxy) [cite: 165] ---
        # Detecting "unnatural" silence/unvoiced patterns.
        # Simple proxy: Frames with High ZCR but Low Energy.
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        rms = librosa.feature.rms(y=y)[0]
        
        # Normalize
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + epsilon)
        
        # Definition: Unvoiced = High ZCR (>0.1) AND Low Energy (<0.2)
        unvoiced_mask = (zcr > 0.1) & (rms_norm < 0.2)
        
        # Calculate average length of consecutive unvoiced segments
        # (This is a heuristic proxy for opensmile's MeanUnvoicedSegmentLength)
        padded = np.pad(unvoiced_mask, (1, 1), 'constant')
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        lengths = ends - starts
        
        unvoiced_len = np.mean(lengths) if len(lengths) > 0 else 0.0

        # Stack into tensor
        features = np.array([teo_mean, pcen_flux_mean, unvoiced_len], dtype=np.float32)
        
        # Simple normalization (instance-level) to keep ranges sane for the MLP
        features = (features - np.mean(features)) / (np.std(features) + epsilon)
        
        return torch.from_numpy(features)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        subset = self.subsets[idx]
        label = self.labels[idx]

        folder_name = f"ASVspoof2019_LA_{subset}"
        file_path = os.path.join(DATASET_ROOT, folder_name, "flac", f"{filename}.flac")

        try:
            # 1. Load Audio [cite: 63]
            audio_np, sr = sf.read(file_path, dtype='float32')

            # 2. VAD & Audio Folding [cite: 94, 98]
            # Remove silence and ensure fixed length via looping
            audio_np = self._process_temporal_structure(audio_np, sr)

            # 3. Signal Standardization (EBU R128) 
            # Crucial: Normalize loudness BEFORE feature extraction
            audio_linear = self._standardize_loudness(audio_np, sr)
            
            # 4. Feature Extraction (x_feat) [cite: 160]
            # Extracted from the standardized LINEAR audio
            extra_features = self.extract_robust_features(audio_linear, sr)

            # 5. Waveform Preparation (x_raw) [cite: 44, 157]
            # Apply Mu-Law Companding for the neural network input
            waveform = self._apply_mu_law(audio_linear, mu=255)

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()

        except Exception as e:
            # Robust fallback
            # print(f"Error processing {filename}: {e}")
            waveform = torch.zeros(1, self.target_length)
            extra_features = torch.zeros(3, dtype=torch.float32)

        # Final Length Check (Paranoia)
        _, w_len = waveform.shape
        if w_len < self.target_length:
            padding = self.target_length - w_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif w_len > self.target_length:
            waveform = waveform[:, :self.target_length]

        return waveform, label, extra_features