import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import AdaptiveSTGCN

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "mmasd_v1_frozen.npz"
MODEL_PATH = "trained_models/best_model.pth"

# 25 Standard Skeletal Joints
JOINTS = [
    "Base", "Spine", "Neck", "Head", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand", "L_Hip", "L_Knee", "L_Ankle",
    "L_Foot", "R_Hip", "R_Knee", "R_Ankle", "R_Foot", "Spine_Chest", "Spine_Mid",
    "L_HandTip", "L_Thumb", "R_HandTip"
]

def generate_explanation():
    print(f"🧠 Starting Explainability (XAI) Analysis...")
    
    if not os.path.exists(DATA_PATH):
        print("❌ Dataset not found.")
        return
    
    # 1. Load Data
    data = np.load(DATA_PATH)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32).to(DEVICE)
    
    # 2. Load Model
    model = AdaptiveSTGCN().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Model loaded.")
    else:
        print("⚠️ Model not found, using random weights.")
    
    model.eval()

    # 3. Saliency Map Logic
    # We find a sample where the model is confident it's ASD
    print("🔍 Calculating Gradient Saliency...")
    
    X_test.requires_grad = True
    outputs = model(X_test)
    probs = torch.softmax(outputs, dim=1)
    
    # Pick the sample with highest ASD probability
    asd_probs = probs[:, 1]
    target_idx = torch.argmax(asd_probs).item()
    
    print(f"✅ Explaining Subject #{target_idx} (ASD Confidence: {asd_probs[target_idx]:.4f})")
    
    # Backpropagate to get gradients
    score = outputs[target_idx, 1]
    score.backward()
    
    # Gradient shape: (3, 300, 25) -> (Channels, Time, Joints)
    grads = X_test.grad[target_idx].cpu().numpy()
    
    # Calculate Importance: Sum of absolute gradients across time and channels
    # Shape becomes (25,) -> One score per joint
    importance = np.sum(np.abs(grads), axis=(0, 1))
    
    # Normalize 0-1
    importance = (importance - importance.min()) / (importance.max() - importance.min())

    # 4. Plotting
    plt.figure(figsize=(14, 6))
    colors = ['red' if x > 0.6 else 'blue' for x in importance]
    sns.barplot(x=np.arange(25), y=importance, palette="viridis")
    plt.xticks(np.arange(25), JOINTS, rotation=45)
    plt.title(f"Explainability: Key Joints Contributing to ASD Prediction")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    
    plt.savefig("xai_joint_importance.png")
    print("✅ Heatmap saved to: xai_joint_importance.png")
    # plt.show() # Commented out for local execution without display

if __name__ == "__main__":
    generate_explanation()