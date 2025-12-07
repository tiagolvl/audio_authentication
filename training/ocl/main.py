import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import optuna
import gc
from tqdm import tqdm

from config import (
    INTERMEDIARY_DIR, MODEL_DIR, BATCH_SIZE, 
    EPOCHS, N_TRIALS, SAMPLES_PER_TRACK, DEVICE,
    SAMPLES_PER_EPOCH, GRAD_ACCUM_STEPS, FINAL_TRAINING_EPOCHS, PATIENCE
)
from ingestion import ingest_datasets
from dataset import ASVspoofDataset
from model import RawNetOCL, OCSoftmax # Changed imports

BEST_PARAMS_PATH = os.path.join(MODEL_DIR, "best_params.json")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "raw_tf_net_best.pth")

def get_balanced_loader(dataset, batch_size):
    targets = dataset.labels
    class_counts = torch.bincount(targets)
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=0, pin_memory=True)

def train_epoch(model, loader, optimizer, criterion, device, scaler, accum_steps):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for i, (inputs, labels, extra_feats) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        extra_feats = extra_feats.to(device, non_blocking=True) # Send features to GPU
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            # Forward pass now takes two inputs
            embeddings = model(inputs, extra_feats)
            
            # OCL Loss returns (loss, scores)
            loss, _ = criterion(embeddings, labels)
            
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_val = loss.item() * accum_steps
        running_loss += loss_val
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    if (i + 1) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Validating", unit="batch", leave=False)

    with torch.no_grad():
        for inputs, labels, extra_feats in pbar:
            inputs = inputs.to(device, non_blocking=True)
            extra_feats = extra_feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                embeddings = model(inputs, extra_feats)
                loss, _ = criterion(embeddings, labels)

            loss_val = loss.item()
            running_loss += loss_val
            pbar.set_postfix(loss=f"{loss_val:.4f}")

    return running_loss / len(loader)

def run_final_training(train_dataset, dev_dataset, params):
    print("\n" + "="*40)
    print(f"FINAL TRAINING STARTED (OCL + Features)")
    print(f"Params: {params}")
    print("="*40)

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    model = RawNetOCL(d_args={}).to(DEVICE)
    # OC-Softmax has trainble parameters (centers), add them to optimizer!
    criterion = OCSoftmax(feat_dim=64).to(DEVICE)
    
    # Combine parameters from model AND loss function
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = optim.Adam(all_params, lr=params['lr'], weight_decay=params['weight_decay'])
    
    scaler = torch.amp.GradScaler('cuda')

    train_loader = get_balanced_loader(train_dataset, BATCH_SIZE)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = PATIENCE

    for epoch in range(FINAL_TRAINING_EPOCHS):
        start = time.time()
        t_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler, GRAD_ACCUM_STEPS)
        v_loss = validate_epoch(model, dev_loader, criterion, DEVICE)
        duration = time.time() - start
        
        print(f"Final Ep {epoch+1}/{FINAL_TRAINING_EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Time: {duration:.1f}s")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            # Also save the loss function state (it has learned centers!)
            torch.save(criterion.state_dict(), os.path.join(MODEL_DIR, "oc_loss_best.pth"))
            print(f"  -> Model & Centers Saved")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping.")
                break

    print("Final training complete.")

def main():
    print(f"Running on: {DEVICE}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        df_train, df_dev, df_eval = ingest_datasets()
    except Exception as e:
        print(f"Ingestion failed: {e}")
        return

    print("Initializing Datasets (with Feature Extraction)...")
    train_dataset = ASVspoofDataset(df_train, target_length=SAMPLES_PER_TRACK)
    dev_dataset = ASVspoofDataset(df_dev, target_length=SAMPLES_PER_TRACK)

    best_params = None
    if os.path.exists(BEST_PARAMS_PATH):
        print(f"\n--- Found existing best params ---")
        with open(BEST_PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
    else:
        print("\n--- Starting Optuna Optimization ---")
        def objective(trial):
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

            model = RawNetOCL(d_args={}).to(DEVICE)
            criterion = OCSoftmax(feat_dim=64).to(DEVICE)
            
            all_params = list(model.parameters()) + list(criterion.parameters())
            optimizer = optim.Adam(all_params, lr=lr, weight_decay=weight_decay)
            scaler = torch.amp.GradScaler('cuda')

            train_loader = get_balanced_loader(train_dataset, BATCH_SIZE)
            dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            for epoch in range(EPOCHS): 
                t_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler, GRAD_ACCUM_STEPS)
                v_loss = validate_epoch(model, dev_loader, criterion, DEVICE)
                trial.report(v_loss, epoch)
                if trial.should_prune(): raise optuna.TrialPruned()
            return v_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS)
        best_params = study.best_params
        with open(BEST_PARAMS_PATH, 'w') as f: json.dump(best_params, f)

    if best_params:
        run_final_training(train_dataset, dev_dataset, best_params)

if __name__ == "__main__":
    main()