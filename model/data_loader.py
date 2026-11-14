from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


class EEGDataset(Dataset):
    """Preprocessed EEG dataset for binary classification."""

    def __init__(self, file_path):

        with np.load(file_path, allow_pickle=False) as data:
            X = data["X"].astype(np.float32)
            y = data["y"].astype(np.int64)

        if X.ndim != 3 or X.shape[1:] != (64, 480):
            raise ValueError(
                f"Expected EEG shape (N,64,480), got {X.shape}"
            )

        if y.shape != (X.shape[0],):
            raise ValueError("Incorrect label dimensions.")

        if not np.isfinite(X).all():
            raise ValueError("EEG contains invalid numerical values.")

        if not np.isin(y, [0, 1]).all():
            raise ValueError("Expected binary labels 0 and 1.")

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def create_dataloaders(
    data_dir="outputs_3s",
    batch_size=16,
    seed=42,
):

    data_dir = Path(data_dir)

    names = [
        "train",
        "val",
        "execution_test",
        "imagery_test",
    ]

    datasets = {
        name: EEGDataset(data_dir / f"{name}.npz")
        for name in names
    }

    generator = torch.Generator().manual_seed(seed)

    loaders = {}

    for name, dataset in datasets.items():

        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=0,
            generator=generator if name == "train" else None,
        )

    return loaders


if __name__ == "__main__":

    loaders = create_dataloaders()

    for name, loader in loaders.items():

        X, y = next(iter(loader))

        print(f"\n{name}")
        print(f"Number of trials: {len(loader.dataset)}")
        print(f"Batch shape: {tuple(X.shape)}")
        print(f"Labels: {torch.unique(y).tolist()}")