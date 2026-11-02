from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "results" / "quality"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load filtered trials BEFORE normalization.
with np.load(DATA_DIR / "S001_trials.npz") as dataset:
    X = dataset["X"].astype(np.float64)
    y = dataset["y"]
    channels = dataset["channel_names"].astype(str)

metadata = pd.read_csv(DATA_DIR / "S001_metadata.csv")

assert len(X) == len(metadata) == len(y)
assert np.array_equal(y, metadata["label"].to_numpy())
assert X.shape[1] == len(channels)

# MNE stores EEG in volts. Convert to microvolts for readable reports.
X_uv = X * 1e6

# Check each electrode within each trial.
# Shape of each result: trials × electrodes.
finite = np.isfinite(X_uv).all(axis=2)

# Mark invalid trial/channel combinations as NaN for calculations.
safe_X = np.where(finite[:, :, None], X_uv, np.nan)

std_uv = np.std(safe_X, axis=2)
ptp_uv = np.ptp(safe_X, axis=2)  # Maximum minus minimum.

# Estimate reference limits using ONLY execution training run 3.
train_mask = (
    metadata["condition"].eq("execution")
    & metadata["run"].eq(3)
).to_numpy()

if not train_mask.any():
    raise ValueError("No execution training trials found.")

if not finite[train_mask].any(axis=0).all():
    raise ValueError("An electrode has no valid training trials.")

# Exploratory upper limit per electrode: Q3 + 3 × IQR.
q1, q3 = np.nanquantile(
    ptp_uv[train_mask], [0.25, 0.75], axis=0
)
ptp_limit = q3 + 3 * (q3 - q1)

nonfinite_flag = ~finite
flat_flag = finite & (std_uv == 0)
high_amplitude_flag = finite & (ptp_uv > ptp_limit[None, :])

# Build one report row per trial/electrode combination.
n_trials, n_channels, _ = X.shape

report = metadata.loc[
    metadata.index.repeat(n_channels)
].reset_index(drop=True)

report["trial_index"] = np.repeat(np.arange(n_trials), n_channels)
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

# Summarize how many trials have at least one flagged channel.
trial_report = metadata.copy()
trial_report["trial_index"] = np.arange(n_trials)
trial_report["flagged_channels"] = (
    nonfinite_flag | flat_flag | high_amplitude_flag
).sum(axis=1)
trial_report["needs_review"] = trial_report["flagged_channels"] > 0

summary = trial_report.groupby(["condition", "run"]).agg(
    total_trials=("needs_review", "size"),
    trials_to_review=("needs_review", "sum")
)

report.to_csv(REPORT_DIR / "S001_channel_checks.csv", index=False)
trial_report.to_csv(REPORT_DIR / "S001_trial_checks.csv", index=False)
summary.to_csv(REPORT_DIR / "S001_quality_summary.csv")

print(summary)
print("\nFlagged examples:")
print(
    report.loc[
        report["needs_review"],
        ["trial_index", "run", "hand", "channel",
         "peak_to_peak_uv", "nonfinite", "flat", "high_amplitude"]
    ].head(15).to_string(index=False)
)
print("\nReports saved to:", REPORT_DIR)