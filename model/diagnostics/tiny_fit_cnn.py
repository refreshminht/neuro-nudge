"""
NEURO 240 - Temporal CNN Small-Subset Overfitting Test

Purpose:
    Determine whether our existing TemporalCNN can memorize
    a small balanced set of motor execution EEG trials.

Dataset:
    outputs_3s/train.npz

Classes:
    0 = Left fist
    1 = Right fist

Experiment:
    8 left-fist trials
    8 right-fist trials

Important:
    This is a diagnostic experiment, not a generalization test.
    Validation, execution-test, and imagery-test data are not used.

Run:
    python -m model.diagnostics.tiny_fit_cnn --data-dir outputs_3s
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model.temporal_cnn import TemporalCNN


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

N_PER_CLASS = 8

LEARNING_RATE = 0.001

MAX_EPOCHS = 150

DEVICE = torch.device("cpu")

RESULTS_DIR = Path("results/cnn/diagnostics")


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Random seed: {seed}")


# ============================================================
# 3. LOAD BALANCED TRAINING SUBSET
# ============================================================

def load_small_subset(data_dir):

    file_path = Path(data_dir) / "train.npz"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {file_path}"
        )

    with np.load(file_path, allow_pickle=False) as data:

        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)

    # Check EEG dimensions
    if X.ndim != 3 or X.shape[1:] != (64, 480):

        raise ValueError(
            f"Expected (N, 64, 480), got {X.shape}"
        )

    if len(X) != len(y):
        raise ValueError("EEG arrays and labels do not match.")

    if not np.isfinite(X).all():
        raise ValueError("Non-finite EEG values detected.")

    # Identify left/right trials
    left_indices = np.where(y == 0)[0]
    right_indices = np.where(y == 1)[0]

    if (
        len(left_indices) < N_PER_CLASS
        or len(right_indices) < N_PER_CLASS
    ):
        raise ValueError("Not enough trials per class.")

    # Select a reproducible balanced subset
    rng = np.random.default_rng(SEED)

    selected_left = rng.choice(
        left_indices,
        size=N_PER_CLASS,
        replace=False
    )

    selected_right = rng.choice(
        right_indices,
        size=N_PER_CLASS,
        replace=False
    )

    selected_indices = np.concatenate([
        selected_left,
        selected_right
    ])

    rng.shuffle(selected_indices)

    X_small = X[selected_indices]
    y_small = y[selected_indices]

    print("\nDataset information")
    print("-" * 50)

    print("Original training shape:", X.shape)
    print("Selected subset shape:", X_small.shape)

    print("Left trials:", np.sum(y_small == 0))
    print("Right trials:", np.sum(y_small == 1))

    print("Selected trial indices:", selected_indices.tolist())

    X_tensor = torch.from_numpy(X_small)
    y_tensor = torch.from_numpy(y_small)

    return X_tensor, y_tensor


# ============================================================
# 4. TRAINING DIAGNOSTIC
# ============================================================

def run_diagnostic(X, y):

    print("\nInitializing TemporalCNN...")

    # Disable dropout for this memorization diagnostic.
    # The original architecture remains unchanged.
    model = TemporalCNN(dropout=0.0).to(DEVICE)

    n_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Trainable parameters:", n_parameters)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    X = X.to(DEVICE)
    y = y.to(DEVICE)

    history = []

    print("\nStarting small-subset training...")
    print("=" * 70)

    for epoch in range(1, MAX_EPOCHS + 1):

        # Training mode
        model.train()

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        # Evaluate memorization on the same training examples.
        # We deliberately do not use held-out data.
        model.eval()

        with torch.no_grad():

            logits_eval = model(X)

            predictions = logits_eval.argmax(dim=1)

            accuracy = (
                predictions == y
            ).float().mean().item()

            eval_loss = criterion(
                logits_eval,
                y
            ).item()

            predicted_left = int(
                (predictions == 0).sum().item()
            )

            predicted_right = int(
                (predictions == 1).sum().item()
            )

        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "eval_loss": float(eval_loss),
            "train_accuracy": float(accuracy),
            "predicted_left": predicted_left,
            "predicted_right": predicted_right
        })

        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"Loss: {eval_loss:.4f} | "
                f"Accuracy: {accuracy * 100:.1f}% | "
                f"Predicted left: {predicted_left} | "
                f"Predicted right: {predicted_right}"
            )

        # Stop once all 16 examples are memorized.
        if accuracy == 1.0:

            print("\nAll 16 training examples memorized.")

            break

    return model, history


# ============================================================
# 5. SAVE RESULTS
# ============================================================

def save_results(history):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULTS_DIR / "tiny_fit_seed42.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys()
        )

        writer.writeheader()
        writer.writerows(history)

    print("\nSaved diagnostic results:")
    print(output_path)


# ============================================================
# 6. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="outputs_3s",
        help="Directory containing train.npz"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NEURO 240 - CNN SMALL-SUBSET DIAGNOSTIC")
    print("=" * 70)

    set_seed(SEED)

    X, y = load_small_subset(args.data_dir)

    model, history = run_diagnostic(X, y)

    save_results(history)

    final_result = history[-1]

    print("\n" + "=" * 70)
    print("FINAL DIAGNOSTIC RESULTS")
    print("=" * 70)

    print("Final epoch:", final_result["epoch"])

    print(
        "Final accuracy:",
        f'{final_result["train_accuracy"] * 100:.2f}%'
    )

    print(
        "Final loss:",
        f'{final_result["eval_loss"]:.4f}'
    )

    print(
        "Predicted left:",
        final_result["predicted_left"]
    )

    print(
        "Predicted right:",
        final_result["predicted_right"]
    )

    if final_result["train_accuracy"] >= 0.95:

        print(
            "\nRESULT: CNN successfully memorized "
            "the small training subset."
        )

    else:

        print(
            "\nRESULT: CNN did not fully memorize "
            "the small training subset."
        )

        print(
            "Further training and architecture "
            "diagnostics may be necessary."
        )


if __name__ == "__main__":
    main()