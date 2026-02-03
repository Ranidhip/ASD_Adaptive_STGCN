import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ASD_Dataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.files = []
        # 'os.walk' will find your folder "3_75 ELEMENTS..." automatically
        for root, dirs, filenames in os.walk(data_dir):
            for f in filenames:
                if f.endswith('.npy'):
                    self.files.append(os.path.join(root, f))
        
        # Split data (80% train / 20% test)
        split_idx = int(len(self.files) * 0.8)
        if mode == 'train':
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        # Load the file
        data = np.load(path)
        data = torch.FloatTensor(data)
        
        # Fix Dimensions: (Frames, Joints, 3) -> (3, Frames, Joints)
        # check the last dimension. If it's 3 (x,y,z), we move it to the front.
        if data.shape[-1] == 3: 
            data = data.permute(2, 0, 1) 
            
        # Create a label (Checking filename for "ASD" or "Typical")
        # Adjust logic if labels are different
        label = 1 if 'ASD' in path else 0 
        
        return data, label