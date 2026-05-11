from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    metrics_path = Path("outputs/metrics.csv")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(
            "outputs/metrics.csv not found. Run scripts/run_mnist_experiment.py first."
        )

    data = pd.read_csv(metrics_path)
    data["model_size_kb"] = data["model_size_bytes"] / 1024

    # model size comparison
    plt.figure(figsize=(8, 5))
    plt.bar(data["name"], data["model_size_kb"])
    plt.ylabel("Model Size (KB)")
    plt.title("Model Size Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "model_size_comparison.png")
    plt.close()

    # accuracy comparison
    plt.figure(figsize=(8, 5))
    plt.bar(data["name"], data["accuracy"])
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png")
    plt.close()

    # communication cost comparison
    plt.figure(figsize=(8, 5))
    plt.bar(data["name"], data["communication_mb"])
    plt.ylabel("Estimated Communication Cost (MB)")
    plt.title("Communication Cost Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "communication_cost_comparison.png")
    plt.close()

    print("Visualizations saved in outputs/")
    print("- model_size_comparison.png")
    print("- accuracy_comparison.png")
    print("- communication_cost_comparison.png")


if __name__ == "__main__":
    main()