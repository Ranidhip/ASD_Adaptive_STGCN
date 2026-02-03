import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# Import your modules
from src.dataset import ASD_Dataset
from src.model import AdaptiveSTGCN

def train_pipeline():
    # --- CONFIGURATION ---
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.001
    # Check for GPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Path to your Converted NPY Data
    DATA_PATH = "./data/processed/3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit/"
    
    print(f"🚀 Setting up Training on {DEVICE}...")

    # 1. Load Data
    try:
        # Check if data exists
        if not os.path.exists(DATA_PATH):
            print(f"❌ Error: Data not found at {DATA_PATH}")
            print("   Did you run preprocess.py first?")
            return

        train_data = ASD_Dataset(DATA_PATH, mode='train')
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        print(f"✅ Data Loaded. Found {len(train_data)} samples.")
    except Exception as e:
        print(f"❌ Error Loading Data: {e}")
        return

    # 2. Initialize Model
    model = AdaptiveSTGCN(num_classes=2, num_joints=25, T=100).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    print("\n🔥 STARTING TRAINING LOOP...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            # --- THE FIX: Handle Dimensions ---
            # Remove the extra 'Person' dimension if it exists
            if len(X.shape) == 5:
                X = X.squeeze(-1) 
            
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f}  |  Accuracy: {accuracy:.2f}%")

    print("🎉 Training Complete!")

if __name__ == "__main__":
    train_pipeline()