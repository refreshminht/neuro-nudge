from pathlib import Path
import numpy as np


DATA_DIR = Path("outputs_3s")

SPLITS = [
    "train",
    "val",
    "execution_test",
    "imagery_test",
]


def verify_dataset():

    reference_channels = None

    for split in SPLITS:

        file_path = DATA_DIR / f"{split}.npz"

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        with np.load(file_path, allow_pickle=False) as data:

            X = data["X"]
            y = data["y"]
            channels = data["channels"]
            sfreq = int(data["sfreq"])
            start = float(data["start"])
            end = float(data["end"])

        # Verify EEG dimensions
        assert X.ndim == 3
        assert X.shape[1:] == (64, 480)

        # Verify labels
        assert y.shape == (X.shape[0],)
        assert np.isin(y, [0, 1]).all()

        # Verify numerical values
        assert np.isfinite(X).all()

        # Verify sampling configuration
        assert sfreq == 160
        assert np.isclose(start, 0.5)
        assert np.isclose(end, 3.5)

        # Verify channel consistency
        assert len(channels) == 64

        if reference_channels is None:
            reference_channels = channels.copy()
        else:
            assert np.array_equal(
                reference_channels, channels
            )

        counts = np.bincount(y, minlength=2)

        print(f"\n{split}")
        print(f"  Shape: {X.shape}")
        print(f"  Data type: {X.dtype}")
        print(f"  Left trials: {counts[0]}")
        print(f"  Right trials: {counts[1]}")

    print("\nAll dataset checks passed.")


if __name__ == "__main__":
    verify_dataset()