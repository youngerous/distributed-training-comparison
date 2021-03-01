"""
Ref: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
Create train, valid, test iterators for CIFAR-100 [1].
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


class DistributedIndicesWrapper(torch.utils.data.Dataset):
    """
    Utility wrapper so that torch.utils.data.distributed.DistributedSampler can work with train test splits

    Ref: https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143
    """

    def __init__(self, dataset: torch.utils.data.Dataset, indices: list):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]
        return self.dataset[idx]


def get_trn_val_loader(
    data_dir: str,
    batch_size: int,
    valid_size: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 1,
    pin_memory: bool = True,
):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-100 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    :param data_dir: path directory to the dataset.
    :param batch_size: how many samples per batch to load.
    :param valid_size: percentage split of the training set used for
                        the validation set. Should be a float in the range [0, 1].
    :param shuffle: whether to shuffle the train/validation indices.
    :param num_workers: number of subprocesses to use when loading the dataset.
    :param pin_memory: whether to copy tensors into CUDA pinned memory.
                        Set it to True if using GPU.
    :param ddgsd: whether to apply ddgsd.
    :return train_loader: training set iterator.
    :return valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    # augmentation
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # load the dataset
    ## Actually, download should be set to be False, because it is not multiprocess safe.
    ## So you should prefetch dataset.
    ## Ref: https://leimao.github.io/blog/PyTorch-Distributed-Training/
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,  ##
        transform=train_transform,
    )

    valid_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,  ##
        transform=valid_transform,
    )

    # train/valid split
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = DistributedSampler(
        DistributedIndicesWrapper(train_dataset, train_idx),
    )
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        shuffle=False,  # False if DistributedSampler exists
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, train_sampler, valid_loader


def get_tst_loader(
    data_dir: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = False,
):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-100 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    :param data_dir: path directory to the dataset.
    :param batch_size: how many samples per batch to load.
    :param shuffle: whether to shuffle the dataset after every epoch.
    :param num_workers: number of subprocesses to use when loading the dataset.
    :param pin_memory: whether to copy tensors into CUDA pinned memory.
                        Set it to True if using GPU.
    :return data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    sampler = DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader
