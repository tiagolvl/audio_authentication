import os
import pandas as pd
import zipfile
from config import PATHS, DATASET_ROOT, ZIP_PATH, RAW_DIR, INTERMEDIARY_DIR, LABEL_MAP

def prepare_dataset_files():
    """Checks for dataset existence and unzips if necessary."""
    if not os.path.exists(PATHS['train']['protocol']):
        print(f"Dataset not found at {DATASET_ROOT}")
        
        if os.path.exists(ZIP_PATH):
            print(f"Found zip file at: {ZIP_PATH}")
            print(f"Extracting to: {RAW_DIR} ...")
            
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(RAW_DIR)
            print("Extraction complete.")
        else:
            raise FileNotFoundError(f"Zip file not found at {ZIP_PATH}. Please place 'LA.zip' in {RAW_DIR}")
    else:
        print(f"Raw dataset files found at {DATASET_ROOT}.")

def parse_asvspoof_protocol(protocol_path, audio_dir, subset_name):
    """Parses ASVspoof protocol text files."""
    if not os.path.exists(protocol_path):
        print(f"Warning: Protocol file not found: {protocol_path}")
        return pd.DataFrame()

    data = []
    with open(protocol_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) < 5: continue
        filename = parts[1]
        label_str = parts[4]
        file_path = os.path.join(audio_dir, f"{filename}.flac")

        data.append({
            "path": file_path,
            "filename": filename,
            "label_str": label_str,
            "label": LABEL_MAP.get(label_str, -1),
            "subset": subset_name
        })
    return pd.DataFrame(data)

def ingest_datasets():
    # Define cache paths
    os.makedirs(INTERMEDIARY_DIR, exist_ok=True)
    cache_train = os.path.join(INTERMEDIARY_DIR, "train.csv")
    cache_dev = os.path.join(INTERMEDIARY_DIR, "dev.csv")
    cache_eval = os.path.join(INTERMEDIARY_DIR, "eval.csv")

    # CHECK: If cached files exist, load them and skip processing
    if os.path.exists(cache_train) and os.path.exists(cache_dev) and os.path.exists(cache_eval):
        print(f"--- Found cached data in {INTERMEDIARY_DIR} ---")
        print("Loading from CSVs...")
        train_df = pd.read_csv(cache_train)
        dev_df = pd.read_csv(cache_dev)
        eval_df = pd.read_csv(cache_eval)
        print("Loaded successfully.")
        return train_df, dev_df, eval_df

    # PROCESS: If no cache, run full ingestion
    print("--- No cache found. Starting raw ingestion ---")

    # 1. Unzip if needed
    prepare_dataset_files()

    # 2. Parse Protocols
    print(f"Parsing Train set...")
    train_df = parse_asvspoof_protocol(PATHS["train"]["protocol"], PATHS["train"]["audio"], "train")

    print(f"Parsing Dev set...")
    dev_df = parse_asvspoof_protocol(PATHS["dev"]["protocol"], PATHS["dev"]["audio"], "dev")

    print(f"Parsing Eval set...")
    eval_df = parse_asvspoof_protocol(PATHS["eval"]["protocol"], PATHS["eval"]["audio"], "eval")

    # 3. Save to Intermediary Folder
    print(f"--- Saving intermediary files to {INTERMEDIARY_DIR} ---")
    
    train_df.to_csv(cache_train, index=False)
    dev_df.to_csv(cache_dev, index=False)
    eval_df.to_csv(cache_eval, index=False)
    print("Saved train.csv, dev.csv, and eval.csv")

    return train_df, dev_df, eval_df

if __name__ == "__main__":
    ingest_datasets()