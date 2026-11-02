from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci

# Find the project folder from this script's location.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "files"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# Start with participant 1.
PARTICIPANT = 1

RUNS = {
    3: "execution",
    7: "execution",
    11: "execution",
    4: "imagery",
    8: "imagery",
    12: "imagery",
}


def preprocess_run(participant, run, condition):
    """Load, filter, and extract left/right trials from one run."""

    file_path = (
        DATA_DIR
        / f"S{participant:03d}"
        / f"S{participant:03d}R{run:02d}.edf"
    )

    raw = mne.io.read_raw_edf(file_path, preload=True)
    eegbci.standardize(raw)

    if raw.info["sfreq"] != 160:
        raise ValueError(f"Unexpected sampling rate in {file_path}")

    raw.filter(l_freq=8, h_freq=30)

    events, _ = mne.events_from_annotations(
        raw,
        event_id={"T1": 1, "T2": 2}
    )

    for annotation in raw.annotations:
        if (
            annotation["description"] in ("T1", "T2")
            and annotation["duration"] < 3.5
        ):
            raise ValueError(f"Task interval too short in {file_path}")

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"left": 1, "right": 2},
        tmin=0.5,
        tmax=3.5 - 1 / raw.info["sfreq"],
        baseline=None,
        picks="eeg",
        preload=True
    )

    X = epochs.get_data().astype(np.float32)
    y = epochs.events[:, 2] - 1

    metadata = pd.DataFrame({
        "participant": participant,
        "run": run,
        "condition": condition,
        "trial_in_run": epochs.selection,
        "cue_sample": epochs.events[:, 0],
        "label": y,
        "hand": np.where(y == 0, "left", "right")
    })

    return X, y, metadata, epochs.ch_names


all_X = []
all_y = []
all_metadata = []
channel_names = None

for run, condition in RUNS.items():
    X, y, metadata, channels = preprocess_run(
        PARTICIPANT, run, condition
    )

    # Both models need the same electrodes in the same order.
    if channel_names is None:
        channel_names = channels
    elif channels != channel_names:
        raise ValueError(f"Channel order differs in run {run}")

    all_X.append(X)
    all_y.append(y)
    all_metadata.append(metadata)

# Join runs while keeping signals, labels, and metadata aligned.
X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
metadata = pd.concat(all_metadata, ignore_index=True)

assert len(X) == len(y) == len(metadata)
assert X.shape[1:] == (64, 480)
assert np.isfinite(X).all()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT_DIR / "S001_trials.npz",
    X=X,
    y=y,
    channel_names=np.array(channel_names),
    sfreq=160
)

metadata.to_csv(OUTPUT_DIR / "S001_metadata.csv", index=False)

print("Combined shape:", X.shape)
print(metadata.groupby(["condition", "run", "hand"]).size())
print("Saved to:", OUTPUT_DIR)