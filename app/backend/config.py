import os
import torch

# --- Paths ---
BASE_DATA_DIR = "data"
MODEL_DIR = os.path.join(BASE_DATA_DIR, "model")

# --- Audio Parameters ---
SAMPLE_RATE = 16000
DURATION = 4
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
LABEL_MAP = {"bonafide": 0, "spoof": 1}

# --- Training Hyperparameters ---
# Physical batch size (what fits in VRAM)
BATCH_SIZE = 16

# Gradient Accumulation Steps
# Effective Batch Size = BATCH_SIZE * GRAD_ACCUM_STEPS = 8 * 4 = 32
GRAD_ACCUM_STEPS = 2

EPOCHS = 50 # Increased to allow convergence
N_TRIALS = 3 # Keep low for optimization speed

# --- GPU Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Subsetting ---
# Set to None to use the FULL dataset (Crucial for fixing underfitting)
SAMPLES_PER_EPOCH = None 

# Evaluation subset
EVAL_SAMPLES = None