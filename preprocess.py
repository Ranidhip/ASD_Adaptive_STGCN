import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
SOURCE_DIR = "./data/processed/raw_csv/"
OUTPUT_FILE = "mmasd_v1_frozen.npz"
MAX_FRAMES = 300  # Fixed time length (pads if shorter, cuts if longer)

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

                    # --- RESIZING (CRITICAL for ST-GCN) ---
                    # We need fixed time length (T) for all samples
                    T = raw_data.shape[0]
                    if T > MAX_FRAMES:
                        raw_data = raw_data[:MAX_FRAMES, :] # Crop
                    elif T < MAX_FRAMES:
                        # Pad with zeros
                        padding = np.zeros((MAX_FRAMES - T, 75))
                        raw_data = np.vstack((raw_data, padding))
                    
                    # Reshape to (C, T, V, M) format
                    # Current: (T, 75) -> (T, 25, 3)
                    data = raw_data.reshape(MAX_FRAMES, 25, 3) 
                    # Transpose to (Channels, Time, Joints) -> (3, T, 25)
                    data = data.transpose(2, 0, 1)
                    # Add Person dimension -> (3, T, 25, 1)
                    data = data[:, :, :, np.newaxis]

                    data_list.append(data)

                    # --- LABELING LOGIC ---
                    # Assumes "ASD" is in the folder name or filename for positive class
                    if "ASD" in file_path.upper():
                        label_list.append(1) # ASD
                    else:
                        label_list.append(0) # Non-ASD

                except Exception as e:
                    # print(f"Skipping {filename}: {e}")
                    continue

    # 2. Convert to Numpy Arrays
    if len(data_list) == 0:
        print("❌ ERROR: No valid CSV files found/converted!")
        return

    X = np.stack(data_list) # Shape: (N, 3, T, 25, 1)
    Y = np.array(label_list)

    print(f"📊 Total Samples: {X.shape[0]}")
    print(f"   - ASD Samples: {np.sum(Y == 1)}")
    print(f"   - Non-ASD Samples: {np.sum(Y == 0)}")

    # 3. Split Train/Test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # 4. Save Final File
    np.savez(OUTPUT_FILE, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    print(f"✅ Success! Saved dataset to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    process_and_save()