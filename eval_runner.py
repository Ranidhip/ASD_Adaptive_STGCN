import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from model import AdaptiveSTGCN

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "mmasd_v1_frozen.npz"
MODEL_PATH = "trained_models/best_model.pth"
BATCH_SIZE = 32

def evaluate():
    print(f"📊 Loading Test Data...")
    if not os.path.exists(DATA_PATH):
        print("❌ Error: Dataset not found.")
        return

    # Load Data
    data = np.load(DATA_PATH)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32).to(DEVICE)
    Y_test = torch.tensor(data['Y_test'], dtype=torch.long).to(DEVICE)
    
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

    # Load Model
    print(f"🔄 Loading Model: {MODEL_PATH}")
    model = AdaptiveSTGCN().to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("⚠️ Warning: Best model not found. Using random weights (Just for demo).")

    model.eval()
    
    # Run Inference
    all_preds = []
    all_labels = []
    
    print("🚀 Running Inference...")
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(Y_batch.cpu().numpy())

    # --- METRICS ---
    print("\n" + "="*40)
    print("       FINAL CLASSIFICATION REPORT")
    print("="*40)
    print(classification_report(all_labels, all_preds, target_names=["Typical", "ASD"]))

    # --- PLOT CONFUSION MATRIX ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Typical", "ASD"], yticklabels=["Typical", "ASD"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Baseline Model)')
    plt.savefig('confusion_matrix.png')
    print("✅ Saved plot to 'confusion_matrix.png'")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    evaluate()