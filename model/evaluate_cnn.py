"""
Temporal CNN evaluation.

Project:
    Motor Execution-to-Imagery Transfer Learning

Purpose:
    Evaluate the pretrained Temporal CNN on:
        1. Held-out motor execution EEG
        2. Motor imagery EEG

Important:
    No training or fine-tuning occurs during evaluation.

Inputs:
    results/cnn/best_model.pt
    outputs_3s/execution_test.npz
    outputs_3s/imagery_test.npz

Outputs:
    results/cnn/evaluation_metrics.json
    results/cnn/execution_predictions.csv
    results/cnn/imagery_predictions.csv
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from model.data_loader import create_dataloaders
from model.temporal_cnn import TemporalCNN


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = Path("outputs_3s")

RESULTS_DIR = Path("results/cnn")

CHECKPOINT_PATH = RESULTS_DIR / "best_model.pt"

METRICS_PATH = RESULTS_DIR / "evaluation_metrics.json"

BATCH_SIZE = 16

DEVICE = torch.device("cpu")

CLASS_NAMES = ["Left fist", "Right fist"]


# ============================================================
# 2. METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    """
    Compute binary classification metrics.

    Class 0: Left fist
    Class 1: Right fist
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError("Prediction and label shapes differ.")

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Unexpected ground-truth labels.")

    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Unexpected predicted labels.")

    # Rows: true labels
    # Columns: predicted labels
    confusion_matrix = np.zeros((2, 2), dtype=np.int64)

    for true_label, prediction in zip(y_true, y_pred):
        confusion_matrix[true_label, prediction] += 1

    accuracy = float(np.mean(y_true == y_pred))

    support = confusion_matrix.sum(axis=1)

    recalls = np.divide(
        np.diag(confusion_matrix),
        support,
        out=np.full(2, np.nan, dtype=float),
        where=support != 0,
    )

    if np.isnan(recalls).any():
        raise ValueError(
            "Both classes must be represented for balanced accuracy."
        )

    balanced_accuracy = float(np.mean(recalls))

    return {
        "n_trials": int(len(y_true)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "left_recall": float(recalls[0]),
        "right_recall": float(recalls[1]),
        "confusion_matrix": confusion_matrix.tolist(),
    }


# ============================================================
# 3. EVALUATE ONE DATASET
# ============================================================

@torch.no_grad()
def evaluate_dataset(model, dataloader):
    """
    Evaluate a pretrained model without updating its weights.
    """

    model.eval()

    all_true = []
    all_pred = []

    for X, y in dataloader:

        X = X.to(DEVICE)

        logits = model(X)

        predictions = logits.argmax(dim=1)

        all_true.extend(y.numpy().tolist())

        all_pred.extend(
            predictions.cpu().numpy().tolist()
        )

    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)

    metrics = compute_metrics(y_true, y_pred)

    return metrics, y_true, y_pred


# ============================================================
# 4. SAVE TRIAL-LEVEL PREDICTIONS
# ============================================================

def save_predictions(output_path, y_true, y_pred):

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "trial",
            "true_label",
            "predicted_label",
            "correct",
        ])

        for i, (true_label, prediction) in enumerate(
            zip(y_true, y_pred),
            start=1,
        ):

            writer.writerow([
                i,
                int(true_label),
                int(prediction),
                int(true_label == prediction),
            ])


# ============================================================
# 5. MAIN EVALUATION
# ============================================================

def main():

    print("=" * 60)
    print("TEMPORAL CNN EVALUATION")
    print("Execution-to-Imagery Generalization")
    print("=" * 60)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Load the best saved model
    # --------------------------------------------------------

    model = TemporalCNN().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"\nLoaded checkpoint from epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint validation loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    loaders = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
    )

    results = {
        "model": "TemporalCNN",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "class_names": CLASS_NAMES,
    }

    # --------------------------------------------------------
    # Evaluate execution and imagery
    # --------------------------------------------------------

    test_conditions = {
        "execution": "execution_test",
        "imagery": "imagery_test",
    }

    for condition, loader_name in test_conditions.items():

        print(f"\nEvaluating {condition} EEG...")

        metrics, y_true, y_pred = evaluate_dataset(
            model,
            loaders[loader_name],
        )

        results[condition] = metrics

        save_predictions(
            RESULTS_DIR / f"{condition}_predictions.csv",
            y_true,
            y_pred,
        )

        print(f"Trials: {metrics['n_trials']}")

        print(
            f"Accuracy: "
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Balanced accuracy: "
            f"{metrics['balanced_accuracy'] * 100:.2f}%"
        )

        print("Confusion matrix:")

        print(
            np.asarray(metrics["confusion_matrix"])
        )

    # --------------------------------------------------------
    # Compute transfer gap
    # --------------------------------------------------------

    execution_ba = results["execution"]["balanced_accuracy"]

    imagery_ba = results["imagery"]["balanced_accuracy"]

    transfer_gap = execution_ba - imagery_ba

    results["transfer_gap"] = float(transfer_gap)

    print("\n" + "=" * 60)
    print("CROSS-CONDITION GENERALIZATION")
    print("=" * 60)

    print(
        f"Execution balanced accuracy: "
        f"{execution_ba * 100:.2f}%"
    )

    print(
        f"Imagery balanced accuracy: "
        f"{imagery_ba * 100:.2f}%"
    )

    print(
        f"Transfer gap: "
        f"{transfer_gap * 100:.2f} percentage points"
    )

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print(f"\nSaved evaluation metrics: {METRICS_PATH}")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()