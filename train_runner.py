import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 30
SAVE_PATH = "adaptive_stgcn_best.pth"
DATA_PATH = "mmasd_v1_frozen.npz"

print(f"🚀 Training on: {DEVICE}")

# --- 1. LOAD DATA ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"🚨 ERROR: '{DATA_PATH}' not found. Please upload it to the Colab/Repo folder.")

print("📦 Loading dataset...")
data = np.load(DATA_PATH)
X_train = torch.tensor(data['X_train'], dtype=torch.float32).to(DEVICE)
Y_train = torch.tensor(data['Y_train'], dtype=torch.long).to(DEVICE)
X_test  = torch.tensor(data['X_test'], dtype=torch.float32).to(DEVICE)
Y_test  = torch.tensor(data['Y_test'], dtype=torch.long).to(DEVICE)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

# --- 2. ADAPTIVE GRAPH CONVOLUTION ---
class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        # The Learnable Matrix 'B' (This is your novelty!)
        self.B = nn.Parameter(torch.randn(num_nodes, num_nodes) * 1e-5, requires_grad=True)

    def forward(self, x):
        N, C, T, V = x.size()
        A_adaptive = torch.eye(V).to(x.device) + self.B
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(-1, V)
        x_graph = torch.matmul(x_reshaped, A_adaptive)
        x_graph = x_graph.view(N, T, C, V).permute(0, 2, 1, 3)
        return self.conv(x_graph)

# --- 3. MODEL ARCHITECTURE ---
class AdaptiveSTGCN(nn.Module):
    def __init__(self, num_classes=2, num_joints=25):
        super().__init__()
        # Layer 1
        self.gcn1 = AdaptiveGraphConv(2, 64, num_joints)
        self.tcn1 = nn.Conv2d(64, 64, kernel_size=(9, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(64)
        # Layer 2
        self.gcn2 = AdaptiveGraphConv(64, 128, num_joints)
        self.tcn2 = nn.Conv2d(128, 128, kernel_size=(9, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(128)
        # Layer 3
        self.gcn3 = AdaptiveGraphConv(128, 256, num_joints)
        self.tcn3 = nn.Conv2d(256, 256, kernel_size=(9, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.tcn1(self.gcn1(x))))
        x = F.dropout(x, 0.3, training=self.training)
        x = F.relu(self.bn2(self.tcn2(self.gcn2(x))))
        x = F.dropout(x, 0.3, training=self.training)
        x = F.relu(self.bn3(self.tcn3(self.gcn3(x))))
        
        # Global Pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- 4. TRAINING LOOP ---
if __name__ == "__main__":
    model = AdaptiveSTGCN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    print("\n🏁 Starting Training...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == by).sum().item()
            total += by.size(0)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for vx, vy in test_loader:
                out = model(vx)
                _, pred = torch.max(out, 1)
                val_correct += (pred == vy).sum().item()
                val_total += vy.size(0)

        val_acc = 100 * val_correct / val_total
        train_acc = 100 * correct / total

        print(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"\n✅ Training Finished. Best Validation Accuracy: {best_acc:.2f}%")
    print(f"💾 Model saved as '{SAVE_PATH}'")