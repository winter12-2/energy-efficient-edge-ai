from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn


def model_size_bytes(model: nn.Module) -> int:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as handle:
        torch.save(model.state_dict(), handle.name)
        return Path(handle.name).stat().st_size


def save_checkpoint(model: nn.Module, path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target)
    return target.stat().st_size


def write_metrics_json(rows: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_metrics_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
