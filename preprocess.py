import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
SOURCE_DIR = "./data/processed/raw_csv/"
OUTPUT_FILE = "mmasd_v1_frozen.npz"
MAX_FRAMES = 300  # Fixed time length

def process_and_save():
    print(f"🔄 Starting Preprocessing: Filename Parsing Mode...")
    
    data_list = []
    label_list = []
    
    # 1. Walk through all files
    for root, dirs, files in os.walk(SOURCE_DIR):
        for filename in tqdm(files, desc="Processing"):
            if filename.endswith(".csv"):
                file_path = os.path.join(root, filename)
                
                try:
                    # --- PARSE FILENAME FOR LABEL ---
                    clean_name = filename.replace(".csv", "")
                    parts = clean_name.split("_")
                    last_part = parts[-1]
                    
                    # Handle cases like "0 (1)"
                    if " " in last_part: last_part = last_part.split(" ")[0]

                    if not last_part.isdigit(): continue 

                    label = int(last_part)

                    # --- FILTER: BINARY CLASSIFICATION (0 vs 1) ---
                    if label not in [0, 1]: continue

                    # --- READ DATA ---
                    df = pd.read_csv(file_path, header=None, dtype=str)
                    df = df.apply(pd.to_numeric, errors='coerce')
                    df = df.dropna(axis=1, how='all').dropna(axis=0, how='any')
                    raw_data = df.values
                    
                    # Ensure 75 columns
                    if raw_data.shape[1] > 75: raw_data = raw_data[:, -75:]
                    elif raw_data.shape[1] < 75: continue 

                    # Resize Time (T)
                    T = raw_data.shape[0]
                    if T > MAX_FRAMES:
                        raw_data = raw_data[:MAX_FRAMES, :]
                    elif T < MAX_FRAMES:
                        padding = np.zeros((MAX_FRAMES - T, 75))
                        raw_data = np.vstack((raw_data, padding))
                    
                    # Reshape: (T, 25, 3) -> (3, T, 25)
                    data = raw_data.reshape(MAX_FRAMES, 25, 3) 
                    data = data.transpose(2, 0, 1)
                    
                    data_list.append(data)
                    label_list.append(label)

                except Exception:
                    continue

    # 2. Convert & Save
    if len(data_list) == 0:
        print("❌ ERROR: No files matched filters 0/1.")
        return

    X = np.stack(data_list) 
    Y = np.array(label_list)

    print(f"📊 DATASET READY | Total: {X.shape[0]} | Typical: {np.sum(Y == 0)} | ASD: {np.sum(Y == 1)}")
    
    # Stratified Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    np.savez(OUTPUT_FILE, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    print(f"✅ Success! Saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    process_and_save()