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
    SAMPLES_PER_EPOCH, GRAD_ACCUM_STEPS
)
from ingestion import ingest_datasets
from dataset import ASVspoofDataset
from model import RawNet, FocalLoss

# Define path for saving parameters
BEST_PARAMS_PATH = os.path.join(MODEL_DIR, "best_params.json")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "raw_tf_net_best.pth")

def get_balanced_loader(dataset, batch_size):
    """
    Creates a DataLoader with WeightedRandomSampler to handle class imbalance.
    This ensures each batch has roughly 50/50 Real/Spoof data.
    """
    targets = dataset.labels
    class_counts = torch.bincount(targets)
    
    # Calculate weights: inverse of frequency
    class_weights = 1. / class_counts.float()
    
    # Assign weight to each sample based on its class
    sample_weights = class_weights[targets]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # When using a sampler, shuffle must be False
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=sampler, 
        num_workers=0, 
        pin_memory=True
    )

def train_epoch(model, loader, optimizer, criterion, device, scaler, accum_steps):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)
    
    optimizer.zero_grad(set_to_none=True)

    for i, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # --- Mixed Precision Context ---
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # --- Gradient Accumulation ---
            # Normalize loss to account for accumulation
            loss = loss / accum_steps

        # Scale loss and backward
        scaler.scale(loss).backward()

        # Step only after accumulating enough gradients
        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_val = loss.item() * accum_steps # Scale back for logging
        running_loss += loss_val
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    # Handle remaining gradients
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
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss_val = loss.item()
            running_loss += loss_val
            pbar.set_postfix(loss=f"{loss_val:.4f}")

    return running_loss / len(loader)

def run_final_training(train_dataset, dev_dataset, params):
    print("\n" + "="*40)
    print(f"FINAL TRAINING STARTED")
    print(f"Params: {params}")
    print("="*40)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = RawNet(d_args={}).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
    
    # Use Focal Loss to focus on hard misclassifications
    criterion = FocalLoss(gamma=2.0).to(DEVICE)

    # Use Balanced Loader (Full Dataset, No Subsetting)
    train_loader = get_balanced_loader(train_dataset, BATCH_SIZE)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 5

    for epoch in range(EPOCHS):
        start = time.time()

        t_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler, GRAD_ACCUM_STEPS)
        v_loss = validate_epoch(model, dev_loader, criterion, DEVICE)

        duration = time.time() - start
        print(f"Final Ep {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Time: {duration:.1f}s")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> Model Saved to {BEST_MODEL_PATH}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Final training complete.")

def main():
    print(f"Running on: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"Batch Size: {BATCH_SIZE} (Accumulated to {BATCH_SIZE * GRAD_ACCUM_STEPS})")
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        df_train, df_dev, df_eval = ingest_datasets()
    except Exception as e:
        print(f"Ingestion failed: {e}")
        return

    print("Initializing Datasets...")
    # NOTE: Using FULL datasets now to fix underfitting
    train_dataset = ASVspoofDataset(df_train, target_length=SAMPLES_PER_TRACK)
    dev_dataset = ASVspoofDataset(df_dev, target_length=SAMPLES_PER_TRACK)

    best_params = None

    if os.path.exists(BEST_PARAMS_PATH):
        print(f"\n--- Found existing best params at {BEST_PARAMS_PATH} ---")
        with open(BEST_PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
    else:
        print("\n--- Starting Optuna Optimization ---")
        
        def objective(trial):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

            model = RawNet(d_args={}).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scaler = torch.amp.GradScaler('cuda')
            criterion = FocalLoss(gamma=2.0).to(DEVICE)

            # Use Balanced Loader for Optuna too
            train_loader = get_balanced_loader(train_dataset, BATCH_SIZE)
            dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            print(f"\n--- Trial {trial.number} (LR: {lr:.1e}) ---")

            # Shorter run for Optuna
            for epoch in range(3): 
                t_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler, GRAD_ACCUM_STEPS)
                v_loss = validate_epoch(model, dev_loader, criterion, DEVICE)
                
                trial.report(v_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return v_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS)
        
        best_params = study.best_params
        print("Best Params:", best_params)
        with open(BEST_PARAMS_PATH, 'w') as f:
            json.dump(best_params, f)

    if best_params:
        run_final_training(train_dataset, dev_dataset, best_params)

if __name__ == "__main__":
    main()