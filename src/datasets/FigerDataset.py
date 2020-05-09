import pickle
import numpy as np

from torch.utils.data import Dataset


class FigerDataset(Dataset):
    def __init__(self, output_dir, ds="train"):
        with open(f"{output_dir}/{ds}_entities.bin", 'rb') as f:
            self.entities = pickle.load(f)
        with open(f"{output_dir}/{ds}_features.bin", 'rb') as f:
            self.features = pickle.load(f)
        with open(f"{output_dir}/{ds}_targets.bin", 'rb') as f:
            self.targets = pickle.load(f)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        entity = self.entities[idx]
        feature = self.features[idx]
        target = self.targets[idx]
        return entity, feature, target
