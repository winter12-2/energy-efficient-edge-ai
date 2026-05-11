from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edge_ai_efficiency.data import build_mnist_loaders
from edge_ai_efficiency.experiment import (
    apply_dynamic_quantization,
    apply_global_pruning,
    build_result,
    evaluate,
    train_one_model,
)
from edge_ai_efficiency.metrics import write_metrics_csv, write_metrics_json
from edge_ai_efficiency.model import MnistCnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MNIST edge-AI efficiency experiments.")
    parser.add_argument("--data-dir", default="data", help="Directory for downloaded datasets.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for metrics and checkpoints.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--prune-amount", type=float, default=0.30)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def save_plot(rows: list[dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = [str(row["name"]) for row in rows]
    accuracies = [float(row["accuracy"]) for row in rows]
    sizes_mb = [int(row["model_size_bytes"]) / (1024 * 1024) for row in rows]

    fig, axis_accuracy = plt.subplots(figsize=(8, 4.5))
    axis_size = axis_accuracy.twinx()
    axis_accuracy.bar(names, accuracies, color="#3874cb", label="Accuracy")
    axis_size.plot(names, sizes_mb, color="#d1495b", marker="o", label="Model size")
    axis_accuracy.set_ylim(0, 1)
    axis_accuracy.set_ylabel("Accuracy")
    axis_size.set_ylabel("Model size (MB)")
    axis_accuracy.set_title("MNIST Baseline vs Optimized Models")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_size.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    baseline = MnistCnn()
    train_seconds = train_one_model(
        model=baseline,
        train_loader=train_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        limit_train_batches=args.limit_train_batches,
    )
    baseline_accuracy, baseline_eval_seconds = evaluate(
        baseline,
        test_loader,
        device,
        limit_test_batches=args.limit_test_batches,
    )

    rows = [
        build_result(
            name="baseline",
            model=baseline,
            accuracy=baseline_accuracy,
            train_seconds=train_seconds,
            eval_seconds=baseline_eval_seconds,
            output_dir=output_dir,
            clients=args.clients,
            rounds=args.rounds,
            bytes_per_parameter=4,
        ).to_dict()
    ]

    pruned = apply_global_pruning(baseline, amount=args.prune_amount)
    pruned_accuracy, pruned_eval_seconds = evaluate(
        pruned,
        test_loader,
        device,
        limit_test_batches=args.limit_test_batches,
    )
    rows.append(
        build_result(
            name=f"pruned_{int(args.prune_amount * 100)}pct",
            model=pruned,
            accuracy=pruned_accuracy,
            train_seconds=0.0,
            eval_seconds=pruned_eval_seconds,
            output_dir=output_dir,
            clients=args.clients,
            rounds=args.rounds,
            bytes_per_parameter=4,
        ).to_dict()
    )

    quantized = apply_dynamic_quantization(baseline)
    quantized_accuracy, quantized_eval_seconds = evaluate(
        quantized,
        test_loader,
        torch.device("cpu"),
        limit_test_batches=args.limit_test_batches,
    )
    rows.append(
        build_result(
            name="dynamic_int8_quantized",
            model=quantized,
            accuracy=quantized_accuracy,
            train_seconds=0.0,
            eval_seconds=quantized_eval_seconds,
            output_dir=output_dir,
            clients=args.clients,
            rounds=args.rounds,
            bytes_per_parameter=1,
        ).to_dict()
    )

    write_metrics_json(rows, output_dir / "metrics.json")
    write_metrics_csv(rows, output_dir / "metrics.csv")
    save_plot(rows, output_dir)

    for row in rows:
        print(
            f"{row['name']}: accuracy={float(row['accuracy']):.4f}, "
            f"size={int(row['model_size_bytes']) / 1024:.1f} KB, "
            f"communication={float(row['communication_mb']):.2f} MB"
        )


if __name__ == "__main__":
    main()
