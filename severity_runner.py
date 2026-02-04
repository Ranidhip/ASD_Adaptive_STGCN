import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import AdaptiveSTGCN

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "mmasd_v1_frozen.npz"
MODEL_PATH = "trained_models/best_model.pth"
BATCH_SIZE = 32
OUTPUT_CSV = "asd_severity_report.csv"

def generate_severity():
    print(f"🚀 Starting Severity Analysis...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print("❌ Error: Dataset not found.")
        return
    
    data = np.load(DATA_PATH)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32).to(DEVICE)
    Y_test = torch.tensor(data['Y_test'], dtype=torch.long).to(DEVICE)
    
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Model
    model = AdaptiveSTGCN().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Model loaded.")
    else:
        print("⚠️ Warning: Model not found. Using random weights.")

    model.eval()
    
    # 3. Get Probability Scores
    all_probs = []
    all_labels = []
    
    print("📊 Calculating ASD Risk Scores...")
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            outputs = model(X_batch)
            # Apply Softmax to get probabilities (0.0 to 1.0)
            probs = F.softmax(outputs, dim=1)
            # Get probability of Class 1 (ASD)
            asd_probs = probs[:, 1].cpu().numpy()
            all_probs.extend(asd_probs)
            all_labels.extend(Y_batch.cpu().numpy())

    all_probs = np.array(all_probs)
    
    # 4. Clustering (The Severity Logic)
    # We cluster the probabilities into 3 groups: Low, Mid, High
    print("🧠 Clustering into Severity Groups...")
    
    # Reshape for KMeans
    X_scores = all_probs.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scores)
    
    # Map clusters to "Low", "Medium", "High" based on their average score
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(cluster_centers) # Index 0 = Lowest Score, Index 2 = Highest
    
    # Create a mapping dict: {0: 'Low', 1: 'High', ...}
    severity_map = {
        sorted_indices[0]: "Low Risk",
        sorted_indices[1]: "Moderate Risk",
        sorted_indices[2]: "High Risk"
    }
    
    severity_labels = [severity_map[c] for c in clusters]
    
    # 5. Save Report
    df = pd.DataFrame({
        "Sample_ID": [f"Test_Child_{i}" for i in range(len(all_probs))],
        "True_Label": ["ASD" if y==1 else "Typical" for y in all_labels],
        "ASD_Probability": all_probs,
        "Severity_Group": severity_labels
    })
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Report saved to: {OUTPUT_CSV}")
    
    # 6. Visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="ASD_Probability", hue="Severity_Group", bins=30, palette="viridis", multiple="stack")
    plt.title("Distribution of ASD Severity Scores")
    plt.xlabel("Model Probability (0=Typical, 1=ASD)")
    plt.ylabel("Number of Children")
    plt.axvline(x=0.5, color='red', linestyle='--', label="Decision Boundary")
    plt.savefig("severity_distribution.png")
    print("✅ Plot saved to: severity_distribution.png")
    # plt.show() # Commented out for headless server

if __name__ == "__main__":
    generate_severity()