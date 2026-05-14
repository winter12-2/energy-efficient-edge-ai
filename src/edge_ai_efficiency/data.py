from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_mnist_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:

    # load the MNIST dataset and train and test data loaders
    # the data loader sends images to the model in batches insted of all at once

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    
    # converting images to tensors and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    root = Path(data_dir)

    # download and load the training and testing datasets
    train_dataset = datasets.MNIST(
        root=root, 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root=root, 
        train=False, 
        download=True, 
        transform=transform
    )

    # traing data is shuffled so that the model sees the data in random order
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    # testing data is not shuffled because we only use it for evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader
