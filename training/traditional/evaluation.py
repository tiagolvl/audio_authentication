import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, det_curve
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import (
    MODEL_DIR, SAMPLES_PER_TRACK, DEVICE, 
    BATCH_SIZE, EVALUATION_DIR, BASE_DATA_DIR, EVAL_SAMPLES
)
from dataset import ASVspoofDataset
from model import RawNet

# Create evaluation directory
os.makedirs(os.path.join(BASE_DATA_DIR, "05_evaluation"), exist_ok=True)

def compute_eer(bonafide_scores, spoof_scores):
    """
    Calculates EER, thresholds, and minDCF-like metrics.
    """
    # Concatenate scores and labels
    # Label 1 = Bonafide (Positive), Label 0 = Spoof (Negative) for ROC logic
    # In our dataset: 0=Bonafide, 1=Spoof. 
    # We invert labels here: 1 for Bonafide to treat it as the "Target" class for detection.
    
    y_true = np.concatenate([np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))])
    y_scores = np.concatenate([bonafide_scores, spoof_scores])

    # Calculate ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # Find EER point where FPR (False Acceptance of Spoof) approx equals FNR (False Rejection of Bonafide)
    eer_threshold_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_threshold_idx]
    threshold = thresholds[eer_threshold_idx]

    return eer, threshold, fpr, fnr

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    # Re-order to match report logic if necessary, but standard is:
    # 0: Bonafide, 1: Spoof
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bonafide (0)', 'Spoof (1)'], 
                yticklabels=['Bonafide (0)', 'Spoof (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - ASVspoof Evaluation')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def evaluate_model():
    print(f"--- Starting Evaluation on {DEVICE} ---")
    
    # 1. Load Data
    try:
        df_eval = pd.read_csv(os.path.join(BASE_DATA_DIR, "02_intermediary", "eval.csv"))
    except FileNotFoundError:
        print("Eval data not found. Run ingestion first.")
        return

    full_eval_dataset = ASVspoofDataset(df_eval, target_length=SAMPLES_PER_TRACK)
    
    # --- SUBSET LOGIC ---
    if EVAL_SAMPLES is not None and EVAL_SAMPLES < len(full_eval_dataset):
        print(f"Subsetting evaluation to {EVAL_SAMPLES} samples.")
        # Create a subset using random indices or sequential
        # Using sequential for reproducibility unless shuffle is needed
        indices = list(range(EVAL_SAMPLES))
        eval_dataset = Subset(full_eval_dataset, indices)
    else:
        print(f"Using full evaluation set ({len(full_eval_dataset)} samples).")
        eval_dataset = full_eval_dataset

    # Batch size 1 is useful for accurate inference timing per sample
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 2. Load Model
    model = RawNet(d_args={}).to(DEVICE)
    model_path = os.path.join(MODEL_DIR, "raw_tf_net_best.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")
    else:
        print("Model file not found. Using untrained model (Results will be random).")

    model.eval()

    # 3. Inference Loop
    bonafide_scores = []
    spoof_scores = []
    inference_times = []
    
    all_labels = []

    print(f"Evaluating {len(eval_dataset)} samples...")
    
    # Wrap loader with tqdm for progress bar
    pbar = tqdm(eval_loader, desc="Evaluating", unit="sample")
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(DEVICE)
            
            # --- Metric: Inference Time (Section 4.1 item 5) ---
            start_time = time.time()
            outputs = model(inputs) # Logits
            end_time = time.time()
            
            # Store inference time in ms
            inference_times.append((end_time - start_time) * 1000)

            # Apply Softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Score for class 0 (Bonafide). Higher score = More likely Bonafide.
            bonafide_score = probs[:, 0].item()
            true_label = labels.item() # 0 for Bonafide, 1 for Spoof

            if true_label == 0:
                bonafide_scores.append(bonafide_score)
            else:
                spoof_scores.append(bonafide_score)
            
            all_labels.append(true_label)

    # 4. Calculate Metrics (EER, FAR, FRR)
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        print("Error: Evaluation set must contain both Bonafide and Spoof samples to calculate EER.")
        return

    bonafide_scores = np.array(bonafide_scores)
    spoof_scores = np.array(spoof_scores)
    
    eer, threshold, fpr, fnr = compute_eer(bonafide_scores, spoof_scores)

    # RE-MATCHING LABELS FOR CONFUSION MATRIX
    # We reconstruct y_true and y_scores based on the separated lists
    y_true_cm = np.concatenate([np.zeros(len(bonafide_scores)), np.ones(len(spoof_scores))])
    y_scores_cm = np.concatenate([bonafide_scores, spoof_scores])
    
    # Prediction: If score >= threshold, it is Bonafide (0). Else Spoof (1).
    # Note logic: High score means Bonafide.
    y_pred_cm = np.where(y_scores_cm >= threshold, 0, 1)

    # 5. Report Results
    avg_inference_time = np.mean(inference_times)
    
    print("\n" + "="*30)
    print("   EVALUATION REPORT   ")
    print("="*30)
    
    print(f"Total Samples: {len(eval_dataset)}")
    print(f"Bonafide Samples: {len(bonafide_scores)}")
    print(f"Spoof Samples: {len(spoof_scores)}")
    print("-" * 20)
    
    # Metrics from Report Table 1
    print(f"[Metric] EER (Target <= 5%): {eer:.4%}")
    
    tn, fp, fn, tp = confusion_matrix(y_true_cm, y_pred_cm).ravel()
    
    # Map to Biometric Terms (0=Bonafide, 1=Spoof):
    # TN = Bonafide correctly identified as Bonafide
    # FP = Bonafide identified as Spoof (False Rejection)
    # FN = Spoof identified as Bonafide (False Acceptance)
    # TP = Spoof correctly identified as Spoof
    
    actual_frr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    actual_far = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"[Metric] FAR - Security (Target < 1%): {actual_far:.4%}")
    print(f"[Metric] FRR - Usability: {actual_frr:.4%}")
    print(f"[Metric] Inference Time (Target < 100ms): {avg_inference_time:.2f} ms")
    print(f"[Info] Optimal Threshold: {threshold:.4f}")
    
    print("-" * 20)
    print("Validation against Project Targets:")
    print(f"[*] EER <= 5%? {'PASSED' if eer <= 0.05 else 'FAILED'}")
    print(f"[*] FAR < 1%? {'PASSED' if actual_far < 0.01 else 'FAILED'}")
    print(f"[*] Latency < 100ms? {'PASSED' if avg_inference_time < 100 else 'FAILED'}")

    # 6. Save Confusion Matrix
    cm_path = os.path.join(BASE_DATA_DIR, "05_evaluation", "confusion_matrix.png")
    plot_confusion_matrix(y_true_cm, y_pred_cm, cm_path)

if __name__ == "__main__":
    evaluate_model()