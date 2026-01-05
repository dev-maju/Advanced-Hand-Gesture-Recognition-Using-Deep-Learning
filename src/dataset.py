import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config import GESTURE_LABELS

class GestureDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.labels = []

        for label, gesture in GESTURE_LABELS.items():
            gesture_path = os.path.join(data_dir, gesture)
            for file in os.listdir(gesture_path):
                self.samples.append(os.path.join(gesture_path, file))
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = np.load(self.samples[idx])
        y = self.labels[idx]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y
