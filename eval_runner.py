import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import AdaptiveSTGCN  # <--- IMPORTING!

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "mmasd_v1_frozen.npz"
MODEL_PATH = "adaptive_stgcn_best.pth"

if __name__ == "__main__":
    # Load Data
    data = np.load(DATA_PATH)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32).to(DEVICE)
    Y_test = torch.tensor(data['Y_test'], dtype=torch.long).to(DEVICE)

    # Load Model
    model = AdaptiveSTGCN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("📊 Evaluating Model...")
    y_true = []
    y_pred = []

    with torch.no_grad():
        out = model(X_test)
        _, preds = torch.max(out, 1)
        y_true.extend(Y_test.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    # Report
    print(classification_report(y_true, y_pred, target_names=["Typical", "ASD"]))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Typical", "ASD"], yticklabels=["Typical", "ASD"])
    plt.title("Confusion Matrix")
    plt.show()