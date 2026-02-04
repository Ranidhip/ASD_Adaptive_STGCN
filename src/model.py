import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveSTGCN(nn.Module):
    def __init__(self, in_channels=3, num_class=2, num_point=25, num_person=1, graph_args=None):
        super(AdaptiveSTGCN, self).__init__()
        
        # --- 1. SPATIAL GRAPH CONVOLUTION (SGC) ---
        # A simple adaptive adjacency matrix
        self.num_point = num_point
        self.A = nn.Parameter(torch.rand(num_point, num_point), requires_grad=True) # Learnable Graph
        
        # --- 2. LAYERS ---
        # Temporal Conv 1
        self.tcn1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 64, kernel_size=(9,1), padding=(4,0)),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5)
        )
        
        # Graph Conv 1 (Adaptive)
        self.gcn1 = nn.Conv2d(64, 64, kernel_size=(1,1))

        # Temporal Conv 2
        self.tcn2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(9,1), padding=(4,0)),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5)
        )
        
        # Graph Conv 2
        self.gcn2 = nn.Conv2d(128, 128, kernel_size=(1,1))
        
        # Temporal Conv 3 (Bottleneck)
        self.tcn3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(9,1), padding=(4,0)),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5)
        )
        
        # --- 3. CLASSIFIER ---
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        # Input Shape: (N, C, T, V) -> (Batch, 3, 300, 25)
        N, C, T, V = x.size()
        
        # --- LAYER 1 ---
        x = self.tcn1(x) 
        # Adaptive Graph Conv: X = A * X * W
        x = self.gcn1(x)
        x = torch.einsum('vw,nctw->nctv', self.A, x) # Apply Adjacency Matrix
        
        # --- LAYER 2 ---
        x = self.tcn2(x)
        x = self.gcn2(x)
        x = torch.einsum('vw,nctw->nctv', self.A, x)
        
        # --- LAYER 3 ---
        x = self.tcn3(x)
        
        # --- POOLING & OUTPUT ---
        # Global Average Pooling over Time (T) and Joints (V)
        x = F.avg_pool2d(x, kernel_size=(T, V)) # (N, 256, 1, 1)
        x = x.view(N, -1) # Flatten -> (N, 256)
        
        x = self.fc(x)
        return x