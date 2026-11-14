from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "results" / "quality"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def check_participant(trials_path):
    """Check one participant and save their quality reports."""

    participant_id = trials_path.name.removesuffix("_trials.npz")
    participant_number = int(participant_id[1:])

    metadata_path = DATA_DIR / f"{participant_id}_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    # Load filtered trials BEFORE normalization.
    with np.load(trials_path) as dataset:
        X = dataset["X"].astype(np.float64)
        y = dataset["y"]
        channels = dataset["channel_names"].astype(str)

    metadata = pd.read_csv(metadata_path)

    # Check that the files describe the same trials and participant.
    if X.ndim != 3 or X.shape[0] == 0 or X.shape[2] == 0:
        raise ValueError(f"{participant_id}: invalid trial array shape.")

    if not (len(X) == len(metadata) == len(y)):
        raise ValueError(f"{participant_id}: trial counts do not match.")

    if not np.array_equal(y, metadata["label"].to_numpy()):
        raise ValueError(f"{participant_id}: labels do not match.")

    if X.shape[1] != len(channels):
        raise ValueError(f"{participant_id}: channel counts do not match.")

    if not metadata["participant"].eq(participant_number).all():
        raise ValueError(f"{participant_id}: participant IDs do not match.")

    # Convert volts to microvolts.
    X_uv = X * 1e6

    # Check each electrode within each trial.
    finite = np.isfinite(X_uv).all(axis=2)
    safe_X = np.where(finite[:, :, None], X_uv, np.nan)

    std_uv = np.std(safe_X, axis=2)
    ptp_uv = np.ptp(safe_X, axis=2)

    # Use only THIS participant's execution training run.
    train_mask = (
        metadata["condition"].eq("execution")
        & metadata["run"].eq(3)
    ).to_numpy()

    if not train_mask.any():
        raise ValueError(
            f"{participant_id}: no execution training trials found."
        )

    if not finite[train_mask].any(axis=0).all():
        raise ValueError(
            f"{participant_id}: an electrode has no valid training trials."
        )

    # Exploratory upper limit per electrode: Q3 + 3 × IQR.
    q1, q3 = np.nanquantile(
        ptp_uv[train_mask],
        [0.25, 0.75],
        axis=0
    )
    ptp_limit = q3 + 3 * (q3 - q1)

    nonfinite_flag = ~finite
    flat_flag = finite & (std_uv == 0)
    high_amplitude_flag = finite & (
        ptp_uv > ptp_limit[None, :]
    )

    # One report row per trial/electrode combination.
    n_trials, n_channels, _ = X.shape

    report = metadata.loc[
        metadata.index.repeat(n_channels)
    ].reset_index(drop=True)

    report["trial_index"] = np.repeat(
        np.arange(n_trials), n_channels
    )
    report["channel"] = np.tile(channels, n_trials)
    report["std_uv"] = std_uv.ravel()
    report["peak_to_peak_uv"] = ptp_uv.ravel()
    report["reference_limit_uv"] = np.tile(ptp_limit, n_trials)
    report["nonfinite"] = nonfinite_flag.ravel()
    report["flat"] = flat_flag.ravel()
    report["high_amplitude"] = high_amplitude_flag.ravel()

    report["needs_review"] = report[
        ["nonfinite", "flat", "high_amplitude"]
    ].any(axis=1)

    # One report row per trial.
    trial_report = metadata.copy()
    trial_report["trial_index"] = np.arange(n_trials)
    trial_report["flagged_channels"] = (
        nonfinite_flag | flat_flag | high_amplitude_flag
    ).sum(axis=1)
    trial_report["needs_review"] = (
        trial_report["flagged_channels"] > 0
    )

    summary = trial_report.groupby(["condition", "run"]).agg(
        total_trials=("needs_review", "size"),
        trials_to_review=("needs_review", "sum")
    )

    # Save separate files for this participant.
    report.to_csv(
        REPORT_DIR / f"{participant_id}_channel_checks.csv",
        index=False
    )
    trial_report.to_csv(
        REPORT_DIR / f"{participant_id}_trial_checks.csv",
        index=False
    )
    summary.to_csv(
        REPORT_DIR / f"{participant_id}_quality_summary.csv"
    )

    print(f"\n{participant_id}")
    print(summary.to_string())
    print(
        "Flagged trials:",
        int(trial_report["needs_review"].sum()),
        "of",
        n_trials
    )

    # Include participant identity in the combined summary.
    summary = summary.reset_index()
    summary.insert(0, "participant", participant_number)

    return summary


def main():
    # Find all preprocessed participant files automatically.
    participant_files = sorted(
        DATA_DIR.glob("S[0-9][0-9][0-9]_trials.npz")
    )

    if not participant_files:
        raise FileNotFoundError(
            f"No participant trial files found in {DATA_DIR}. "
            "Run preprocess_eeg.py first."
        )

    all_summaries = []

    for trials_path in participant_files:
        summary = check_participant(trials_path)
        all_summaries.append(summary)

    # Combine the summaries from this run of the script.
    combined_summary = pd.concat(
        all_summaries,
        ignore_index=True
    )

    combined_path = REPORT_DIR / "all_participants_quality_summary.csv"
    combined_summary.to_csv(combined_path, index=False)

    print(f"\nChecked {len(participant_files)} participants.")
    print("Reports saved to:", REPORT_DIR)
    print("Combined summary:", combined_path)


if __name__ == "__main__":
    main()