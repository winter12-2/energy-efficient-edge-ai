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

    # stores the results for 1 model version. 
    # Each row represents baseline mdeol, or prunded model or quantized model.

    name: str
    accuracy: float
    train_seconds: float
    eval_seconds: float
    model_size_bytes: int
    parameter_count: int
    nonzero_parameter_count: int
    communication_mb: float
    checkpoint_path: str

    #convert the result to dict for saving in csv/json format
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
    
    # train one model and return the training time

    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if limit_train_batches is not None and limit_train_batches <= 0:
        raise ValueError("limit_train_batches must be greater than 0")
    
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

# test the model and return accuracy and evaluation time. 
# torch.no_grad() because we are only testing here

@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    limit_test_batches: int | None = None,
) -> tuple[float, float]:
    
    if limit_test_batches is not None and limit_test_batches <= 0:
        raise ValueError("limit_test_batches must be greater than 0")
    
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


# Here we apply pruning to remove less important weights from the model. 
# this helps lower the no. of active parameters which can lower the communication cost
def apply_global_pruning(model: nn.Module, amount: float) -> nn.Module:

    if amount < 0 or amount > 1:
        raise ValueError("prune amount must be between 0 and 1")

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

    # removing pruning wrappers so that the model can be saved normally
    for module, parameter_name in parameters_to_prune:
        prune.remove(module, parameter_name)
    return pruned_model


# apply dynamic INT8 quantization. this helps reduce model size for edge devices.
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
