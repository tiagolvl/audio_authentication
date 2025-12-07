import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import (
    MODEL_DIR, SAMPLES_PER_TRACK, DEVICE, 
    BATCH_SIZE, EVALUATION_DIR, BASE_DATA_DIR
)
from dataset import ASVspoofDataset
from model import RawNetOCL, OCSoftmax # New model

EVAL_SAMPLES = None  
os.makedirs(os.path.join(BASE_DATA_DIR, "05_evaluation"), exist_ok=True)

def compute_eer(bonafide_scores, spoof_scores):
    y_true = np.concatenate([np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))])
    y_scores = np.concatenate([bonafide_scores, spoof_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_threshold_idx]
    threshold = thresholds[eer_threshold_idx]
    return eer, threshold

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bonafide (0)', 'Spoof (1)'], 
                yticklabels=['Bonafide (0)', 'Spoof (1)'])
    plt.title('Confusion Matrix - ASVspoof OCL Evaluation')
    plt.savefig(save_path)
    plt.close()

def evaluate_model():
    print(f"--- Starting Evaluation on {DEVICE} (OCL Mode) ---")
    
    try:
        df_eval = pd.read_csv(os.path.join(BASE_DATA_DIR, "02_intermediary", "eval.csv"))
    except FileNotFoundError:
        print("Eval data not found.")
        return

    full_eval_dataset = ASVspoofDataset(df_eval, target_length=SAMPLES_PER_TRACK)
    
    if EVAL_SAMPLES is not None and EVAL_SAMPLES < len(full_eval_dataset):
        indices = list(range(EVAL_SAMPLES))
        eval_dataset = Subset(full_eval_dataset, indices)
    else:
        eval_dataset = full_eval_dataset

    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load Model AND Loss (for centers)
    model = RawNetOCL(d_args={}).to(DEVICE)
    loss_fn = OCSoftmax(feat_dim=64).to(DEVICE)
    
    model_path = os.path.join(MODEL_DIR, "raw_tf_net_best.pth")
    loss_path = os.path.join(MODEL_DIR, "oc_loss_best.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        # Load centers if available (crucial for accurate OCL scoring)
        if os.path.exists(loss_path):
            loss_fn.load_state_dict(torch.load(loss_path, map_location=DEVICE))
            print("Loaded Model AND Loss Centers.")
        else:
            print("Loaded Model, but NO Loss Centers found. Results might be inaccurate.")
    else:
        print("Model file not found.")
        return

    model.eval()
    
    bonafide_scores = []
    spoof_scores = []
    inference_times = []

    pbar = tqdm(eval_loader, desc="Evaluating", unit="sample")
    
    with torch.no_grad():
        for inputs, labels, extra_feats in pbar:
            inputs = inputs.to(DEVICE)
            extra_feats = extra_feats.to(DEVICE)
            
            start_time = time.time()
            embeddings = model(inputs, extra_feats)
            
            # --- OCL Scoring ---
            # In OCL, the "Score" is the similarity to the Real Center.
            # High similarity = Real. Low similarity = Spoof.
            _, scores = loss_fn(embeddings, labels.to(DEVICE))
            
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)

            score = scores.item() # This is cosine similarity
            true_label = labels.item()

            if true_label == 0:
                bonafide_scores.append(score)
            else:
                spoof_scores.append(score)

    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        print("Error: Dataset missing classes.")
        return

    bonafide_scores = np.array(bonafide_scores)
    spoof_scores = np.array(spoof_scores)
    
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)

    # CM Logic
    y_true_cm = np.concatenate([np.zeros(len(bonafide_scores)), np.ones(len(spoof_scores))])
    y_scores_cm = np.concatenate([bonafide_scores, spoof_scores])
    
    # If score > threshold -> Bonafide (0). Else Spoof (1).
    y_pred_cm = np.where(y_scores_cm >= threshold, 0, 1)

    avg_inference_time = np.mean(inference_times)
    
    # --- REPORT GENERATION ---
    # We collect all lines into a list first, then print and save them.
    
    report_lines = []
    report_lines.append("\n" + "="*30)
    report_lines.append("   EVALUATION REPORT   ")
    report_lines.append("="*30)
    
    report_lines.append(f"Total Samples: {len(eval_dataset)}")
    report_lines.append(f"Bonafide Samples: {len(bonafide_scores)}")
    report_lines.append(f"Spoof Samples: {len(spoof_scores)}")
    report_lines.append("-" * 20)

    tn, fp, fn, tp = confusion_matrix(y_true_cm, y_pred_cm).ravel()
    
    # Map to Biometric Terms (0=Bonafide, 1=Spoof):
    actual_frr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    actual_far = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    report_lines.append(f"[Metric] FAR - Security (Target < 1%): {actual_far:.4%}")
    report_lines.append(f"[Metric] FRR - Usability: {actual_frr:.4%}")
    report_lines.append(f"[Metric] Inference Time (Target < 100ms): {avg_inference_time:.2f} ms")
    report_lines.append(f"[Info] Optimal Threshold: {threshold:.4f}")
    
    report_lines.append("-" * 20)
    report_lines.append("Validation against Project Targets:")
    report_lines.append(f"[*] EER <= 5%? {'PASSED' if eer <= 0.05 else 'FAILED'}")
    report_lines.append(f"[*] FAR < 1%? {'PASSED' if actual_far < 0.01 else 'FAILED'}")
    report_lines.append(f"[*] Latency < 100ms? {'PASSED' if avg_inference_time < 100 else 'FAILED'}")

    # Combine into one string
    final_report = "\n".join(report_lines)

    # 1. Print to Console
    print(final_report)

    # 2. Save to .txt file
    report_path = os.path.join(BASE_DATA_DIR, "05_evaluation", "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(final_report)
    
    print(f"\n[Saved] Report saved to: {report_path}")

    # 3. Generate Confusion Matrix Image
    cm_path = os.path.join(BASE_DATA_DIR, "05_evaluation", "confusion_matrix.png")
    plot_confusion_matrix(y_true_cm, y_pred_cm, cm_path)

if __name__ == "__main__":
    evaluate_model()