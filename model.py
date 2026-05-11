from __future__ import annotations

import torch
from torch import nn


class MnistCnn(nn.Module):
    """Small CNN suitable for fast MNIST edge-AI experiments."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module, only_nonzero: bool = False) -> int:
    total = 0
    for parameter in model.parameters():
        if only_nonzero:
            total += int(torch.count_nonzero(parameter).item())
        else:
            total += parameter.numel()
    return total
