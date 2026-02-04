import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from model import AdaptiveSTGCN

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
DATA_PATH = "mmasd_v1_frozen.npz"
CHECKPOINT_DIR = "trained_models"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if __name__ == "__main__":
    print(f"🚀 Training on: {DEVICE}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"🚨 {DATA_PATH} not found. Run preprocess.py first!")

    # Load Data
    data = np.load(DATA_PATH)
    X_train = torch.tensor(data['X_train'], dtype=torch.float32).to(DEVICE)
    Y_train = torch.tensor(data['Y_train'], dtype=torch.long).to(DEVICE)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32).to(DEVICE)
    Y_test = torch.tensor(data['Y_test'], dtype=torch.long).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model
    model = AdaptiveSTGCN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    print("🏁 Starting Training...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += Y_batch.size(0)
            correct += (predicted == Y_batch).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total_val += Y_batch.size(0)
                correct_val += (predicted == Y_batch).sum().item()
        
        val_acc = 100 * correct_val / total_val
        
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))

    print(f"✅ Finished. Best Model Saved to '{CHECKPOINT_DIR}/best_model.pth' | Best Acc: {best_acc:.2f}%")