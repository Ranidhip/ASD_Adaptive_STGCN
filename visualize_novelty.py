import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. MODEL DEFINITION ---
class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.B = nn.Parameter(torch.randn(num_nodes, num_nodes) * 1e-5, requires_grad=True)
    def forward(self, x):
        return self.conv(x) # Dummy forward for this script

class AdaptiveSTGCN(nn.Module):
    def __init__(self, num_classes=2, num_joints=25):
        super().__init__()
        self.gcn1 = AdaptiveGraphConv(2, 64, num_joints)
        # We only need the first layer to visualize features
    def forward(self, x):
        pass

# --- 2. VISUALIZATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_novelty():
    model = AdaptiveSTGCN().to(DEVICE)
    # Load partially (strict=False allowed because we only defined part of the model here for simplicity)
    try:
        full_state = torch.load("adaptive_stgcn_best.pth", map_location=DEVICE)
        model.gcn1.B.data = full_state['gcn1.B']
        print("✅ Weights loaded for Layer 1.")
    except:
        print("❌ Could not load weights. Running with random init (Demo only).")

    weights = model.gcn1.B.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, cmap="coolwarm", center=0, square=True)
    plt.title("Learned Adaptive Connections (The Novelty)")
    plt.xlabel("Joint Index")
    plt.ylabel("Joint Index")
    plt.show()

if __name__ == "__main__":
    plot_novelty()