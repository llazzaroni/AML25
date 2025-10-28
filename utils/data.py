import numpy as np
from torch.utils.data import TensorDataset, random_split
import torch

def get_subsplits(X, y, train_ratio=0.8, seed=42):

    # First turn X and y into np object
    X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().reshape(-1,1).astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    #generator = torch.Generator().manual_seed(seed)

    #train_subsplit, val_subsplit = random_split(dataset, [train_size, test_size], generator=generator)
    train_subsplit, val_subsplit = random_split(dataset, [train_size, test_size])

    return train_subsplit, val_subsplit

