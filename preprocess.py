import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def convert_csv_to_npy():
    # --- CONFIG ---
    # Where the raw CSVs from Drive will be
    source_dir = "./data/processed/raw_csv/" 
    # Where the clean NPYs will go (and where train_runner looks)
    target_dir = "./data/processed/3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit/"

    print(f"🔄 Starting Preprocessing (CSV -> NPY)...")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files_converted = 0
    
    for root, dirs, files in os.walk(source_dir):
        for filename in tqdm(files, desc="Converting"):
            if filename.endswith(".csv"):
                try:
                    file_path = os.path.join(root, filename)
                    
                    # Robust CSV reading
                    df = pd.read_csv(file_path, header=None, dtype=str)
                    df = df.apply(pd.to_numeric, errors='coerce')
                    df = df.dropna(axis=1, how='all')
                    df = df.dropna(axis=0, how='any')
                    
                    data = df.values
                    # Ensure 75 columns
                    if data.shape[1] > 75: data = data[:, -75:]
                    
                    # Reshape
                    T = data.shape[0]
                    data = data.reshape(T, 25, 3) # (Time, Joints, Channels)
                    data = data.transpose(2, 0, 1) # (Channels, Time, Joints)
                    data = data[:, :, :, np.newaxis] # (Channels, Time, Joints, Person)

                    # Save
                    relative_folder = os.path.relpath(root, source_dir)
                    save_folder = os.path.join(target_dir, relative_folder)
                    
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    
                    np.save(os.path.join(save_folder, filename.replace(".csv", ".npy")), data)
                    files_converted += 1
                    
                except Exception:
                    continue

    print(f"✅ Preprocessing Done! Converted {files_converted} files.")

if __name__ == "__main__":
    convert_csv_to_npy()