import os
import torch

# --- Paths ---
BASE_DATA_DIR = "data"
RAW_DIR = os.path.join(BASE_DATA_DIR, "01_raw")
INTERMEDIARY_DIR = os.path.join(BASE_DATA_DIR, "02_intermediary")
MODEL_DIR = os.path.join(BASE_DATA_DIR, "04_model")
EVALUATION_DIR = os.path.join(BASE_DATA_DIR, "05_evaluation")

# Zip file configuration
ZIP_FILENAME = "LA.zip" 
ZIP_PATH = os.path.join(RAW_DIR, ZIP_FILENAME)
DATASET_ROOT = os.path.join(RAW_DIR, "LA")

# --- Protocol Paths ---
PATHS = {
    "train": {
        "audio": os.path.join(DATASET_ROOT, "ASVspoof2019_LA_train/flac"),
        "protocol": os.path.join(DATASET_ROOT, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    },
    "dev": {
        "audio": os.path.join(DATASET_ROOT, "ASVspoof2019_LA_dev/flac"),
        "protocol": os.path.join(DATASET_ROOT, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
    },
    "eval": {
        "audio": os.path.join(DATASET_ROOT, "ASVspoof2019_LA_eval/flac"),
        "protocol": os.path.join(DATASET_ROOT, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
    }
}

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

EPOCHS = 5 # Increased to allow convergence
N_TRIALS = 3 # Keep low for optimization speed

FINAL_TRAINING_EPOCHS = 50
PATIENCE = 10
# --- GPU Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Subsetting ---
# Set to None to use the FULL dataset (Crucial for fixing underfitting)
SAMPLES_PER_EPOCH = None 

# Evaluation subset
EVAL_SAMPLES = None