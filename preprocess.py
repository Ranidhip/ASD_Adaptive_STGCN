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
    print(f"🔄 Starting Preprocessing: CSV -> Combined Dataset ({OUTPUT_FILE})...")
    
    data_list = []
    label_list = []
    
    # 1. Walk through all files
    for root, dirs, files in os.walk(SOURCE_DIR):
        for filename in tqdm(files, desc="Reading CSVs"):
            if filename.endswith(".csv"):
                file_path = os.path.join(root, filename)
                
                try:
                    # Read CSV safely
                    df = pd.read_csv(file_path, header=None, dtype=str)
                    df = df.apply(pd.to_numeric, errors='coerce')
                    df = df.dropna(axis=1, how='all').dropna(axis=0, how='any')
                    
                    raw_data = df.values
                    
                    # Ensure 75 columns (25 joints * 3 coords)
                    if raw_data.shape[1] > 75: 
                        raw_data = raw_data[:, -75:]
                    elif raw_data.shape[1] < 75:
                        continue # Skip bad files

                    # --- RESIZING (CRITICAL) ---
                    T = raw_data.shape[0]
                    if T > MAX_FRAMES:
                        raw_data = raw_data[:MAX_FRAMES, :] # Crop
                    elif T < MAX_FRAMES:
                        # Pad with zeros
                        padding = np.zeros((MAX_FRAMES - T, 75))
                        raw_data = np.vstack((raw_data, padding))
                    
                    # --- RESHAPE TO (Channels, Time, Joints) ---
                    # 1. Reshape flat 75 -> (T, 25, 3)
                    data = raw_data.reshape(MAX_FRAMES, 25, 3) 
                    # 2. Transpose to (3, T, 25) which is (C, T, V)
                    data = data.transpose(2, 0, 1)
                    
                    # NOTE: We do NOT add the 5th 'Person' dimension here to avoid the crash
                    data_list.append(data)

                    # --- LABELING LOGIC (BROAD DETECTION) ---
                    path_upper = file_path.upper()
                    # Checks for "ASD", "AUTISM", or "PATIENT" (often used for positive class)
                    if "ASD" in path_upper or "AUTISM" in path_upper or "PATIENT" in path_upper:
                        label_list.append(1) # ASD
                    else:
                        label_list.append(0) # Non-ASD

                except Exception as e:
                    continue

    # 2. Stack and Save
    if len(data_list) == 0:
        print("❌ ERROR: No valid CSV files found/converted!")
        return

    X = np.stack(data_list) # Shape: (N, 3, T, 25)
    Y = np.array(label_list)

    print(f"📊 Total Samples: {X.shape[0]}")
    print(f"   - ASD Samples: {np.sum(Y == 1)}")
    print(f"   - Non-ASD Samples: {np.sum(Y == 0)}")

    if np.sum(Y==1) == 0:
        print("⚠️ CRITICAL WARNING: Still finding 0 ASD samples. Check your folder names in Drive!")

    # 3. Split Train/Test
    # Handle case where we might have very few samples during testing
    stratify_labels = Y if np.sum(Y==1) > 1 else None
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=stratify_labels)

    # 4. Save Final File
    np.savez(OUTPUT_FILE, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    print(f"✅ Success! Saved dataset to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    process_and_save()