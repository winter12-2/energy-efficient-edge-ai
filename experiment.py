from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from edge_ai_efficiency.communication import estimate_model_update_cost
from edge_ai_efficiency.metrics import model_size_bytes, save_checkpoint
from edge_ai_efficiency.model import count_parameters


@dataclass
class ExperimentResult:
    name: str
    accuracy: float
    train_seconds: float
    eval_seconds: float
    model_size_bytes: int
    parameter_count: int
    nonzero_parameter_count: int
    communication_mb: float
    checkpoint_path: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    limit_train_batches: int | None = None,
) -> float:
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    started = time.perf_counter()
    for _epoch in range(epochs):
        for batch_index, (features, labels) in enumerate(train_loader):
            if limit_train_batches is not None and batch_index >= limit_train_batches:
                break
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
    return time.perf_counter() - started


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    limit_test_batches: int | None = None,
) -> tuple[float, float]:
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    started = time.perf_counter()
    for batch_index, (features, labels) in enumerate(test_loader):
        if limit_test_batches is not None and batch_index >= limit_test_batches:
            break
        features = features.to(device)
        labels = labels.to(device)
        predictions = model(features).argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.numel()

    elapsed = time.perf_counter() - started
    accuracy = correct / total if total else 0.0
    return accuracy, elapsed


def apply_global_pruning(model: nn.Module, amount: float) -> nn.Module:
    pruned_model = copy.deepcopy(model)
    parameters_to_prune = []
    for module in pruned_model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, parameter_name in parameters_to_prune:
        prune.remove(module, parameter_name)
    return pruned_model


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    cpu_model = copy.deepcopy(model).to("cpu")
    cpu_model.eval()
    return torch.quantization.quantize_dynamic(
        cpu_model,
        {nn.Linear},
        dtype=torch.qint8,
    )


def build_result(
    name: str,
    model: nn.Module,
    accuracy: float,
    train_seconds: float,
    eval_seconds: float,
    output_dir: Path,
    clients: int,
    rounds: int,
    bytes_per_parameter: int,
) -> ExperimentResult:
    parameter_count = count_parameters(model)
    nonzero_parameter_count = count_parameters(model, only_nonzero=True)
    communication = estimate_model_update_cost(
        parameter_count=nonzero_parameter_count,
        clients=clients,
        rounds=rounds,
        bytes_per_parameter=bytes_per_parameter,
    )
    checkpoint_path = output_dir / "checkpoints" / f"{name}.pt"
    checkpoint_size = save_checkpoint(model, checkpoint_path)
    return ExperimentResult(
        name=name,
        accuracy=accuracy,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        model_size_bytes=checkpoint_size or model_size_bytes(model),
        parameter_count=parameter_count,
        nonzero_parameter_count=nonzero_parameter_count,
        communication_mb=communication.total_megabytes,
        checkpoint_path=str(checkpoint_path),
    )
