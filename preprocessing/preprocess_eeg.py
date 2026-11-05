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
PARTICIPANTS =[1, 2, 3, 4, 5]

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


    # Collect the left/right task annotations.
    task_annotations = [
        annotation
        for annotation in raw.annotations
        if annotation["description"] in ("T1", "T2")
    ]

    assert len(task_annotations) == len(events)

    sfreq = raw.info["sfreq"]
    rows = []

    for annotation, event in zip(task_annotations, events):
        # Sample positions relative to the start of the recording.
        cue_sample = int(event[0])
        cue_index = cue_sample - raw.first_samp

        # Match the sample window used by our epoch extraction.
        start_sample = cue_index + round(0.5 * sfreq)
        last_sample = cue_index + round(3.5 * sfreq) - 1

        duration = float(annotation["duration"])
        within_task = duration >= 3.5
        within_recording = (
            start_sample >= 0
            and last_sample < raw.n_times
        )

        rows.append({
            "participant": participant,
            "run": run,
            "condition": condition,
            "cue_sample": cue_sample,
            "hand": "left" if event[2] == 1 else "right",
            "task_duration_s": duration,
            "within_task": within_task,
            "within_recording": within_recording,
            "window_fits": within_task and within_recording,
        })

    window_report = pd.DataFrame(rows)

    report_dir = PROJECT_DIR / "results" / "quality"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / (
        f"S{participant:03d}R{run:02d}_window_checks.csv"
    )

    window_report.to_csv(report_path, index=False)

    print("Window report saved to:", report_path)
    print("All windows fit:", window_report["window_fits"].all())

    if not window_report["window_fits"].all():
        raise ValueError(f"A trial window does not fit in {file_path}")

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


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for participant in PARTICIPANTS:
    all_X = []
    all_y = []
    all_metadata = []
    channel_names = None

    for run, condition in RUNS.items():
        X, y, metadata, channels = preprocess_run(
            participant, run, condition
        )

        if channel_names is None:
            channel_names = channels
        elif channels != channel_names:
            raise ValueError(
                f"Channel order differs for participant {participant}, run {run}"
            )

        all_X.append(X)
        all_y.append(y)
        all_metadata.append(metadata)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    metadata = pd.concat(all_metadata, ignore_index=True)

    assert len(X) == len(y) == len(metadata)
    assert X.shape[1:] == (64, 480)
    assert np.isfinite(X).all()

    participant_id = f"S{participant:03d}"

    np.savez_compressed(
        OUTPUT_DIR / f"{participant_id}_trials.npz",
        X=X,
        y=y,
        channel_names=np.array(channel_names),
        sfreq=160
    )

    metadata.to_csv(
        OUTPUT_DIR / f"{participant_id}_metadata.csv",
        index=False
    )

    print(f"\n{participant_id}: {X.shape}")
    print(metadata.groupby(["condition", "run", "hand"]).size())