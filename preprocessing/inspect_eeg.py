from pathlib import Path
import mne
from mne.datasets import eegbci
import numpy as np
import pandas as pd

# Replace this with your actual dataset folder.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "files"

file_path = DATA_DIR / "S001" / "S001R03.edf"

if not file_path.exists():
    raise FileNotFoundError(f"Check your dataset path: {file_path}")

# Load participant 1's first left/right execution run.
raw = mne.io.read_raw_edf(file_path, preload=True)

# Clean up this dataset's electrode names and assign positions.
eegbci.standardize(raw)
raw.set_montage("colin27_1005")

print("Sampling rate:", raw.info["sfreq"])
print("Number of channels:", len(raw.ch_names))
print("Electrode names:", raw.ch_names)

# Inspect the task annotations.
print(raw.annotations.to_data_frame().head(12))

# 1. Filter a copy, keeping the original recording unchanged.
# Use 8–30 Hz as an initial preprocessing choice...talk to ozgur about freq adjustment
filtered = raw.copy().filter(l_freq=8, h_freq=30)

# 2. Convert the task annotations into numbered events.
# For run 3: T1 = left hand, T2 = right hand.
events, event_id = mne.events_from_annotations(
    filtered,
    event_id={"T1": 1, "T2": 2}
)

# Confirm that the chosen window fits inside each task interval.
task_annotations = [
    annotation for annotation in filtered.annotations
    if annotation["description"] in ("T1", "T2")
]

if any(annotation["duration"] < 3.5 for annotation in task_annotations):
    raise ValueError("A task interval is too short for this trial window.")

# 3. Extract 3-second trials, beginning 0.5 seconds after each cue.
sfreq = filtered.info["sfreq"]

epochs = mne.Epochs(
    filtered,
    events,
    event_id={"left": 1, "right": 2},
    tmin=0.5,
    tmax=3.5 - 1 / sfreq,
    baseline=None,
    picks="eeg",
    preload=True
)

# 4. Get the signal arrays and matching labels.
X = epochs.get_data()
y = epochs.events[:, 2] - 1  # 0 = left, 1 = right

#keeping track of every example
metadata = pd.DataFrame({
    "participant": [1] * len(y),
    "run": [3] * len(y),
    "condition": ["execution"] * len(y),
    "trial_in_run": epochs.selection,
    "cue_sample": epochs.events[:, 0],
    "label": y,
    "hand": np.where(y == 0, "left", "right")
})

print(metadata.head())

print("Data shape:", X.shape)
print("Left trials:", np.sum(y == 0))
print("Right trials:", np.sum(y == 1))
print("Dropped trials:", epochs.drop_log)

# View a short section of the recordings.
filtered.plot(
    picks=["C3", "Cz", "C4"],
    start=4.5,
    duration=2,
    scalings={"eeg": 100e-6},
    block=True
)