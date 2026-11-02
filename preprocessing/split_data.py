from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

# Load the signals and their matching metadata.
with np.load(DATA_DIR / "S001_trials.npz") as dataset:
    X = dataset["X"]
    y = dataset["y"]
    channel_names = dataset["channel_names"]
    sfreq = dataset["sfreq"]

metadata = pd.read_csv(DATA_DIR / "S001_metadata.csv")

# Check that signals, labels, and metadata match.
assert len(X) == len(y) == len(metadata)
assert np.array_equal(y, metadata["label"].to_numpy())

# Assign each trial to one partition.
execution = metadata["condition"].eq("execution")

masks = {
    "train": (execution & metadata["run"].eq(3)).to_numpy(),
    "validation": (execution & metadata["run"].eq(7)).to_numpy(),
    "execution_test": (execution & metadata["run"].eq(11)).to_numpy(),
    "imagery_test": metadata["condition"].eq("imagery").to_numpy(),
}

# Every trial must belong to exactly one partition.
membership = np.stack(list(masks.values()))
assert np.all(membership.sum(axis=0) == 1)

# Calculate one mean and standard deviation per electrode,
# using ONLY training trials and their time samples.
X_train = X[masks["train"]].astype(np.float64)

mean = X_train.mean(axis=(0, 2), keepdims=True)
std = X_train.std(axis=(0, 2), keepdims=True)

if np.any(std == 0):
    raise ValueError("A training electrode has zero variation.")

# Save these values so both models use identical normalization.
np.savez(
    DATA_DIR / "S001_normalization.npz",
    mean=mean,
    std=std,
    channel_names=channel_names
)

metadata["split"] = ""

for name, mask in masks.items():
    # Apply TRAINING statistics to every partition.
    X_normalized = ((X[mask] - mean) / std).astype(np.float32)
    assert np.isfinite(X_normalized).all()

    np.savez_compressed(
        DATA_DIR / f"S001_{name}.npz",
        X=X_normalized,
        y=y[mask],
        trial_indices=np.flatnonzero(mask),
        channel_names=channel_names,
        sfreq=sfreq
    )

    metadata.loc[mask, "split"] = name
    print(f"{name}: {X_normalized.shape}")

metadata.to_csv(DATA_DIR / "S001_split_metadata.csv", index=False)