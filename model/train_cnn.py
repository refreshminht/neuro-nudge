"""
Temporal CNN training for motor execution EEG classification.

NEURO 240: Motor Execution-to-Imagery Transfer Learning

Training:
    outputs_3s/train.npz

Validation:
    outputs_3s/val.npz

Model:
    TemporalCNN

Outputs:
    results/cnn/best_model.pt
    results/cnn/training_history.csv

Important:
    Motor imagery data are NOT used during training or
    model selection.
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model.data_loader import create_dataloaders
from model.temporal_cnn import TemporalCNN


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 16
LEARNING_RATE = 0.001

MAX_EPOCHS = 50
PATIENCE = 10

DEVICE = torch.device("cpu")

DATA_DIR = Path("outputs_3s")
RESULTS_DIR = Path("results/cnn")

MODEL_PATH = RESULTS_DIR / "best_model.pt"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Random seed: {seed}")


# ============================================================
# 3. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in dataloader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * X.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == y).sum().item()

        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 4. VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in dataloader:

        X = X.to(device)
        y = y.to(device)

        outputs = model(X)

        loss = criterion(outputs, y)

        running_loss += loss.item() * X.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == y).sum().item()

        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 5. SAVE TRAINING HISTORY
# ============================================================

def save_history(history, output_path):

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)


# ============================================================
# 6. MAIN TRAINING PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("TEMPORAL CNN TRAINING")
    print("Motor Execution EEG: Left vs Right Fist")
    print("=" * 60)

    # Set random seed
    set_seed(SEED)

    # Create results directory
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load datasets
    loaders = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    print("\nDataset information:")
    print(f"Training trials: {len(train_loader.dataset)}")
    print(f"Validation trials: {len(val_loader.dataset)}")
    print(f"Device: {DEVICE}")

    # Initialize CNN
    model = TemporalCNN().to(DEVICE)

    n_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {n_parameters}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Early stopping variables
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    print("\nStarting training...\n")

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(1, MAX_EPOCHS + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        # Record epoch results
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "seed": SEED,
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "model_name": "TemporalCNN",
                    "class_names": ["left", "right"],
                },
                MODEL_PATH,
            )

            print("  -> Best model saved")

        else:

            epochs_without_improvement += 1

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if epochs_without_improvement >= PATIENCE:

            print("\nEarly stopping triggered.")
            break

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    save_history(
        history,
        HISTORY_PATH,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved history: {HISTORY_PATH}")

    print("\nImagery data were not used during training.")


# ============================================================
# 7. RUN
# ============================================================

if __name__ == "__main__":
    main()