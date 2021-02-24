from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class YourDataset(Dataset):
    def __init__(self, path: str):
        super(self, YourDataset).__init__()

    def __len__(self) -> int:
        return

    def __getitem__(self, idx):
        return


def get_dataloaders(path: str, batch_size: int) -> Tuple[DataLoader]:
    """
    TODO: add collate function or transformation

    :return: train_loader, val_loader, test_loader
    """
    train_dset = None
    val_dset = None
    test_dset = None

    train_loader = DataLoader()
    val_loader = DataLoader()
    test_loader = DataLoader()

    return train_loader, val_loader, test_loader
