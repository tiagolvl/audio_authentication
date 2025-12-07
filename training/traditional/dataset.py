import os
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from config import DATASET_ROOT  # Import the correct local path root

class ASVspoofDataset(Dataset):
    def __init__(self, df, target_length=64000):
        self.df = df
        self.target_length = target_length
        # Pre-convert labels to tensor
        self.labels = torch.tensor(df['label'].values, dtype=torch.long)
        
        # We grab filenames and subsets to reconstruct the path correctly
        self.filenames = df['filename'].values
        self.subsets = df['subset'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        subset = self.subsets[idx]
        label = self.labels[idx]

        # --- PATH RECONSTRUCTION ---
        # Instead of using the broken path from the CSV, we build it here.
        # Structure: data/01_raw/LA/ASVspoof2019_LA_{subset}/flac/{filename}.flac
        folder_name = f"ASVspoof2019_LA_{subset}"
        file_path = os.path.join(DATASET_ROOT, folder_name, "flac", f"{filename}.flac")

        # --- LOAD AUDIO ---
        try:
            # sf.read returns (data, samplerate)
            audio_np, sr = sf.read(file_path, dtype='float32')
            
            # Convert to Tensor
            waveform = torch.from_numpy(audio_np)

            # Ensure shape is (Channels, Time)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()

        except Exception as e:
            # If explicit open fails, try normalized path just in case
            # This helps if there are subtle slash issues on Windows
            try:
                norm_path = os.path.normpath(file_path)
                audio_np, sr = sf.read(norm_path, dtype='float32')
                waveform = torch.from_numpy(audio_np)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.t()
            except:
                print(f"FAILED to load: {file_path}")
                # Return a silent tensor to prevent crashing the whole training loop
                waveform = torch.zeros(1, self.target_length)

        # --- Fast Pad/Truncate ---
        _, w_len = waveform.shape
        if w_len < self.target_length:
            padding = self.target_length - w_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif w_len > self.target_length:
            waveform = waveform[:, :self.target_length]

        return waveform, label