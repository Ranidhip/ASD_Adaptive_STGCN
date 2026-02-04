import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

# --- CONFIG ---
SOURCE_DIR = "./data/processed/raw_csv/"
OUTPUT_FILE = "mmasd_v1_frozen.npz"
MAX_FRAMES = 300 

def normalize_skeleton(data):
    """
    1. Centers the skeleton to (0,0,0).
    2. SCALES it so every child is mathematically the same size.
    """
    # Center to body mean
    center = np.mean(data[0], axis=0) 
    data = data - center
    
    # Scale Normalization (The Fix)
    max_dist = np.max(np.linalg.norm(data[0], axis=1))
    if max_dist > 0:
        data = data / max_dist
        
    return data

def process_and_save():
    print(f"🔄 Starting Preprocessing: Scale-Invariant Mode...")
    
    data_list = []
    label_list = []
    group_list = [] 
    
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ ERROR: Source directory '{SOURCE_DIR}' not found.")
        return

    for root, dirs, files in os.walk(SOURCE_DIR):
        for filename in tqdm(files, desc="Processing"):
            if filename.endswith(".csv"):
                try:
                    clean_name = filename.replace(".csv", "")
                    parts = clean_name.split("_")
                    label_str = parts[-1].split(" ")[0]
                    if not label_str.isdigit(): continue
                    label = int(label_str)
                    if label not in [0, 1]: continue
                    patient_id = "_".join(parts[:-1]) 

                    df = pd.read_csv(os.path.join(root, filename), header=None, dtype=str)
                    df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').dropna(axis=0, how='any')
                    raw = df.values
                    
                    if raw.shape[1] > 75: raw = raw[:, -75:]
                    elif raw.shape[1] < 75: continue 
                    if raw.shape[0] > MAX_FRAMES: raw = raw[:MAX_FRAMES, :]
                    else: raw = np.vstack((raw, np.zeros((MAX_FRAMES - raw.shape[0], 75))))
                    
                    skel_data = raw.reshape(MAX_FRAMES, 25, 3)
                    
                    # --- APPLY SCALING ---
                    skel_data = normalize_skeleton(skel_data)
                    
                    data = skel_data.transpose(2, 0, 1)
                    
                    data_list.append(data)
                    label_list.append(label)
                    group_list.append(patient_id)
                except: continue

    if not data_list: return
    X = np.stack(data_list)
    Y = np.array(label_list)
    groups = np.array(group_list)
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, Y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    np.savez(OUTPUT_FILE, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    print(f"✅ Saved SCALED dataset: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_and_save()