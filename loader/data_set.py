
import torch
from torch.utils.data import Dataset


class TrajDataset(Dataset):
    def __init__(self, dates, features, targets, window_size):
        self.dates = dates
        self.features = features
        self.targets = targets
        self.window_size = window_size
    
    def __len__(self):
        return len(self.dates) - (self.window_size + 1)
    
    def __getitem__(self, idx):
        t = self.targets[:, idx+self.window_size-1].clone()
        f = self.features[:, idx:idx+self.window_size, :].clone()
        return f, t


class TrajNormDataset(Dataset):
    def __init__(self, dates, features, targets, window_size):
        self.dates = dates
        self.features = features
        self.targets = targets
        self.window_size = window_size
    
    def __len__(self):
        return len(self.dates) - (self.window_size + 1)
    
    def __getitem__(self, idx):
        t = self.targets[:, idx+self.window_size-1].clone()
        f = self.features[:, idx:idx+self.window_size, :].clone()
        final_close = f[:, -1, 2].reshape(-1, 1, 1).expand(-1, f.size(1), 3)
        final_volume = f[:, -1, 3].reshape(-1, 1, 1).expand(-1, f.size(1), 1)
        divisor = torch.cat([final_close, final_volume], dim=-1)
        f[:, :, :4] = f[:, :, :4] / divisor
        return f, t