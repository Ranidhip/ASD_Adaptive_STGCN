import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# --- RE-DEFINE MODEL (Required to load weights) ---
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

# --- EXECUTION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
data = np.load("mmasd_v1_frozen.npz")
X_test  = torch.tensor(data['X_test'], dtype=torch.float32).to(DEVICE)
Y_test  = torch.tensor(data['Y_test'], dtype=torch.long).to(DEVICE)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=32, shuffle=False)

# Load Model
model = AdaptiveSTGCN().to(DEVICE)
model.load_state_dict(torch.load("adaptive_stgcn_best.pth", map_location=DEVICE))
model.eval()

# Get Predictions
y_true = []
y_pred = []
print("📊 Generating Evaluation Metrics...")

with torch.no_grad():
    for vx, vy in test_loader:
        out = model(vx)
        _, pred = torch.max(out, 1)
        y_true.extend(vy.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

# Plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-ASD', 'ASD'], yticklabels=['Non-ASD', 'ASD'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\n📝 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Non-ASD', 'ASD']))