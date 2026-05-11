# Energy-Efficient Edge AI Experiments

This repository starts the implementation phase from the research progress report. It trains a baseline CNN on MNIST, records efficiency metrics, then compares pruning and quantization as lightweight edge-AI optimization methods.

## Project Goals

- Build and train a simple MNIST CNN in PyTorch.
- Measure accuracy, execution time, model size, parameter count, and estimated communication cost.
- Apply model compression through pruning.
- Apply quantization for a smaller inference model.
- Save tables and plots for the final analysis report.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the First Experiment

```powershell
python scripts/run_mnist_experiment.py --epochs 3 --batch-size 64
```

Outputs are written to `outputs/`:

- `metrics.json`
- `metrics.csv`
- `accuracy_vs_size.png`
- saved model checkpoints

For a quick smoke test:

```powershell
python scripts/run_mnist_experiment.py --epochs 1 --limit-train-batches 20 --limit-test-batches 10
```

## Metrics

The initial implementation records:

- Accuracy
- Training time
- Evaluation time
- Model size on disk
- Parameter count
- Estimated communication bytes for federated-learning style updates

The communication estimate is intentionally simple. It gives a first comparison point for baseline, pruned, and quantized models before adding a full ns-3 or federated learning simulator.
