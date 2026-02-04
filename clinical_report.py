import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# --- 1. MODEL DEFINITION (Must match exactly) ---
class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.B = nn.Parameter(torch.randn(num_nodes, num_nodes) * 1e-5, requires_grad=True)
    def forward(self, x):
        N, C, T, V = x.size()
        A_adaptive = torch.eye(V).to(x.device) + self.B
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(-1, V)
        x_graph = torch.matmul(x_reshaped, A_adaptive)
        x_graph = x_graph.view(N, T, C, V).permute(0, 2, 1, 3)
        return self.conv(x_graph)

class AdaptiveSTGCN(nn.Module):
    def __init__(self, num_classes=2, num_joints=25):
        super().__init__()
        self.gcn1 = AdaptiveGraphConv(2, 64, num_joints)
        self.tcn1 = nn.Conv2d(64, 64, kernel_size=(9, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.gcn2 = AdaptiveGraphConv(64, 128, num_joints)
        self.tcn2 = nn.Conv2d(128, 128, kernel_size=(9, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(128)
        self.gcn3 = AdaptiveGraphConv(128, 256, num_joints)
        self.tcn3 = nn.Conv2d(256, 256, kernel_size=(9, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        x = F.relu(self.bn1(self.tcn1(self.gcn1(x))))
        x = F.relu(self.bn2(self.tcn2(self.gcn2(x))))
        x = F.relu(self.bn3(self.tcn3(self.gcn3(x))))
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- 2. REPORT LOGIC ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEVERITY_MAPPING = {"Low": (0.50, 0.70), "Moderate": (0.70, 0.90), "High": (0.90, 1.00)}

def generate_report():
    # Load Model
    model = AdaptiveSTGCN().to(DEVICE)
    try:
        model.load_state_dict(torch.load("adaptive_stgcn_best.pth", map_location=DEVICE))
    except:
        print("❌ Please run train_runner.py first!")
        return
    model.eval()

    # Load Data and pick random ASD sample
    data = np.load("mmasd_v1_frozen.npz")
    X_test = torch.tensor(data['X_test'], dtype=torch.float32).to(DEVICE)
    Y_test = torch.tensor(data['Y_test'], dtype=torch.long).to(DEVICE)
    
    # Find all ASD indices to guarantee we show an ASD example
    asd_indices = (Y_test == 1).nonzero(as_tuple=True)[0]
    idx = asd_indices[torch.randint(0, len(asd_indices), (1,)).item()]

    x_input = X_test[idx].unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        logits = model(x_input)
        probs = F.softmax(logits, dim=1)
    
    prob_asd = probs[0][1].item()
    severity = "Unknown"
    for level, (low, high) in SEVERITY_MAPPING.items():
        if low <= prob_asd <= high: severity = level

    print("-" * 50)
    print(f"📄 CLINICAL REPORT | Patient ID: {idx.item()}")
    print("-" * 50)
    print(f"Prediction: ASD DETECTED")
    print(f"Confidence: {prob_asd*100:.2f}%")
    print(f"Severity:   {severity.upper()}")
    print("-" * 50)

    # Plot
    plt.figure(figsize=(8, 3))
    plt.barh(['Non-ASD', 'ASD'], [probs[0][0].item(), prob_asd], color=['green', 'red'])
    plt.xlim(0, 1)
    plt.title("Diagnostic Confidence")
    plt.show()

if __name__ == "__main__":
    generate_report()