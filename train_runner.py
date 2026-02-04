import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from model import AdaptiveSTGCN  # <--- IMPORTING THE CLASS!

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
DATA_PATH = "mmasd_v1_frozen.npz"
SAVE_PATH = "adaptive_stgcn_best.pth"

print(f"🚀 Training on: {DEVICE}")

# --- LOAD DATA ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("🚨 Run preprocess.py first!")

data = np.load(DATA_PATH)
X_train = torch.tensor(data['X_train'], dtype=torch.float32).to(DEVICE)
Y_train = torch.tensor(data['Y_train'], dtype=torch.long).to(DEVICE)
X_test  = torch.tensor(data['X_test'], dtype=torch.float32).to(DEVICE)
Y_test  = torch.tensor(data['Y_test'], dtype=torch.long).to(DEVICE)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

# --- RUNNER ---
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

    print(f"\n✅ Finished. Best Acc: {best_acc:.2f}%")