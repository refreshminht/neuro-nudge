from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_DIR / "results" / "quality"
PARTICIPANT = "S001"

# 1. Load the six timing reports.
window_files = sorted(
    REPORT_DIR.glob(f"{PARTICIPANT}R*_window_checks.csv")
)

if not window_files:
    raise FileNotFoundError("Run preprocess_eeg.py first.")

windows = pd.concat(
    [pd.read_csv(path) for path in window_files],
    ignore_index=True
)

# Load trial-level and electrode-level quality reports.
trials = pd.read_csv(
    REPORT_DIR / f"{PARTICIPANT}_trial_checks.csv"
)
channels = pd.read_csv(
    REPORT_DIR / f"{PARTICIPANT}_channel_checks.csv"
)


# Read CSV boolean values explicitly.
def read_boolean(column):
    values = column.astype(str).str.strip().str.lower()

    if not values.isin(["true", "false"]).all():
        raise ValueError(f"Unexpected values in {column.name}")

    return values.eq("true")


windows["window_fits"] = read_boolean(windows["window_fits"])
trials["needs_review"] = read_boolean(trials["needs_review"])

flag_columns = ["nonfinite", "flat", "high_amplitude"]

for column in flag_columns:
    channels[column] = read_boolean(channels[column])

# Identify a trial using its participant, run, condition, and cue.
trial_keys = ["participant", "run", "condition", "cue_sample"]
run_keys = ["participant", "run", "condition"]

if windows.duplicated(trial_keys).any():
    raise ValueError("Duplicate trials in timing reports.")

if trials.duplicated(trial_keys).any():
    raise ValueError("Duplicate trials in trial reports.")

# 2. Count each flag once per trial, even if several electrodes flag.
trial_flags = (
    channels.groupby(trial_keys)[flag_columns]
    .any()
    .reset_index()
)

trials = trials.merge(
    trial_flags,
    on=trial_keys,
    how="left",
    validate="one_to_one"
)

if trials[flag_columns].isna().any().any():
    raise ValueError("Some trials have no matching channel checks.")

# Verify that quality reports agree about which trials need review.
if not trials["needs_review"].eq(
    trials[flag_columns].any(axis=1)
).all():
    raise ValueError("Quality reports disagree. Rerun check_quality.py.")

# Confirm that every extracted trial has a matching timing report.
timing_match = trials[trial_keys].merge(
    windows[trial_keys],
    on=trial_keys,
    how="left",
    indicator=True,
    validate="one_to_one"
)

if not timing_match["_merge"].eq("both").all():
    raise ValueError("An extracted trial has no timing report.")

trials["is_left"] = trials["hand"].eq("left")
trials["is_right"] = trials["hand"].eq("right")

# 3. Summarize timing checks by run.
timing_summary = windows.groupby(run_keys).agg(
    annotated_tasks=("cue_sample", "size"),
    min_task_duration_s=("task_duration_s", "min"),
    max_task_duration_s=("task_duration_s", "max"),
    windows_passed=("window_fits", "sum")
)

timing_summary["windows_failed"] = (
    timing_summary["annotated_tasks"]
    - timing_summary["windows_passed"]
)

# Summarize extracted trials and quality flags by run.
quality_summary = trials.groupby(run_keys).agg(
    extracted_trials=("cue_sample", "size"),
    left_trials=("is_left", "sum"),
    right_trials=("is_right", "sum"),
    nonfinite_trials=("nonfinite", "sum"),
    flat_trials=("flat", "sum"),
    high_amplitude_trials=("high_amplitude", "sum"),
    flagged_trials=("needs_review", "sum")
)

summary = timing_summary.join(
    quality_summary,
    how="outer",
    validate="one_to_one"
)

if summary.isna().any().any():
    raise ValueError("Timing and quality reports cover different runs.")

summary["unflagged_trials"] = (
    summary["extracted_trials"] - summary["flagged_trials"]
)

summary = summary.reset_index().sort_values(run_keys)

# 4. Save the combined table.
csv_path = REPORT_DIR / f"{PARTICIPANT}_combined_summary.csv"
summary.to_csv(csv_path, index=False)

print(summary.to_string(index=False))
print("\nSummary saved to:", csv_path)

# 5. Plot trial counts by run.
plot_data = summary.set_index(
    summary.apply(
        lambda row: f"R{int(row['run']):02d}\n{row['condition']}",
        axis=1
    )
)[["unflagged_trials", "flagged_trials"]]

ax = plot_data.plot.bar(
    stacked=True,
    figsize=(9, 5),
    color=["#4C9FAD", "#E9A34B"],
    rot=0
)

ax.set_title("S001: EEG quality screening by run")
ax.set_xlabel("Recording run")
ax.set_ylabel("Number of extracted trials")
ax.legend(["No flags", "Flagged for review"])

plt.tight_layout()

plot_path = REPORT_DIR / f"{PARTICIPANT}_quality_overview.png"
plt.savefig(plot_path, dpi=200)
plt.close()

print("Plot saved to:", plot_path)