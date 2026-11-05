from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_DIR / "results" / "quality"

RUN_ORDER = [3, 4, 7, 8, 11, 12]
RUN_LABELS = [
    "R03\nExecution",
    "R04\nImagery",
    "R07\nExecution",
    "R08\nImagery",
    "R11\nExecution",
    "R12\nImagery",
]


def read_boolean(column):
    """Read True/False values from a CSV."""
    values = column.astype(str).str.strip().str.lower()

    if not values.isin(["true", "false"]).all():
        raise ValueError(f"Unexpected values in {column.name}")

    return values.eq("true")


def summarize_participant(trial_path):
    """Combine one participant's timing and trial-quality reports."""
    participant_id = trial_path.name.removesuffix("_trial_checks.csv")
    participant_number = int(participant_id[1:])

    trials = pd.read_csv(trial_path)
    trials["needs_review"] = read_boolean(trials["needs_review"])

    window_files = sorted(
        REPORT_DIR.glob(f"{participant_id}R*_window_checks.csv")
    )

    if not window_files:
        raise FileNotFoundError(
            f"No timing reports found for {participant_id}."
        )

    windows = pd.concat(
        [pd.read_csv(path) for path in window_files],
        ignore_index=True,
    )

    for column in ["within_task", "within_recording", "window_fits"]:
        windows[column] = read_boolean(windows[column])

    if not windows["window_fits"].eq(
        windows["within_task"] & windows["within_recording"]
    ).all():
        raise ValueError(
            f"{participant_id}: inconsistent timing checks."
        )

    trial_keys = ["participant", "run", "condition", "cue_sample"]
    run_keys = ["participant", "run", "condition"]

    expected_runs = {
        (3, "execution"),
        (7, "execution"),
        (11, "execution"),
        (4, "imagery"),
        (8, "imagery"),
        (12, "imagery"),
    }

    for table in [trials, windows]:
        if not table["participant"].eq(participant_number).all():
            raise ValueError(
                f"{participant_id}: participant IDs do not match."
            )

        if table.duplicated(trial_keys).any():
            raise ValueError(
                f"{participant_id}: duplicate trial identities."
            )

        actual_runs = set(
            table[["run", "condition"]].itertuples(
                index=False, name=None
            )
        )

        if actual_runs != expected_runs:
            raise ValueError(
                f"{participant_id}: missing or unexpected runs."
            )

    # Every extracted trial must have a matching timing check.
    matched = trials[trial_keys].merge(
        windows[trial_keys],
        on=trial_keys,
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    if not matched["_merge"].eq("both").all():
        raise ValueError(
            f"{participant_id}: an extracted trial has no timing check."
        )

    timing = windows.groupby(run_keys).agg(
        annotated_tasks=("cue_sample", "size"),
        min_task_duration_s=("task_duration_s", "min"),
        max_task_duration_s=("task_duration_s", "max"),
        windows_passed=("window_fits", "sum"),
    )

    timing["windows_failed"] = (
        timing["annotated_tasks"] - timing["windows_passed"]
    )

    if not trials["hand"].isin(["left", "right"]).all():
        raise ValueError(
            f"{participant_id}: unexpected hand labels."
        )

    trials["is_left"] = trials["hand"].eq("left")
    trials["is_right"] = trials["hand"].eq("right")

    quality = trials.groupby(run_keys).agg(
        extracted_trials=("cue_sample", "size"),
        left_trials=("is_left", "sum"),
        right_trials=("is_right", "sum"),
        flagged_trials=("needs_review", "sum"),
    )

    summary = timing.join(
        quality,
        how="outer",
        validate="one_to_one",
    )

    if summary.isna().any().any():
        raise ValueError(
            f"{participant_id}: timing and quality runs do not match."
        )

    summary["unflagged_trials"] = (
        summary["extracted_trials"] - summary["flagged_trials"]
    )
    summary["tasks_not_extracted"] = (
        summary["annotated_tasks"] - summary["extracted_trials"]
    )
    summary["flagged_percent"] = (
        100 * summary["flagged_trials"] / summary["extracted_trials"]
    )

    return summary.reset_index()


def plot_participants(participants):
    """Save one stacked bar per participant."""
    plot_data = participants[
        ["unflagged_percent", "flagged_percent"]
    ].copy()

    plot_data.index = [
        f"S{int(number):03d}" for number in participants.index
    ]

    ax = plot_data.plot.bar(
        stacked=True,
        figsize=(max(8, len(participants) * 0.6), 5),
        color=["#4C9FAD", "#E9A34B"],
        rot=45 if len(participants) > 10 else 0,
    )

    for position, total in enumerate(participants["extracted_trials"]):
        ax.text(
            position,
            102,
            f"n={int(total)}",
            ha="center",
            fontsize=9,
        )

    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Participant")
    ax.set_ylabel("Percentage of extracted trials")
    ax.set_title("EEG quality screening by participant", pad=40)
    ax.legend(
        ["No flags", "Flagged for review"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "quality_by_participant.png", dpi=200)
    plt.close(fig)


def plot_runs(run_summary):
    """Compare quality flags and timing failures across people and runs."""
    flag_grid = run_summary.pivot(
        index="participant",
        columns="run",
        values="flagged_percent",
    ).reindex(columns=RUN_ORDER).sort_index()

    timing_grid = run_summary.pivot(
        index="participant",
        columns="run",
        values="windows_failed",
    ).reindex(index=flag_grid.index, columns=RUN_ORDER)

    labels = [f"S{int(number):03d}" for number in flag_grid.index]
    timing_max = max(1, float(timing_grid.max().max()))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, max(4, len(labels) * 0.4)),
        constrained_layout=True,
    )

    quality_image = axes[0].imshow(
        flag_grid.to_numpy(),
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        aspect="auto",
    )

    timing_image = axes[1].imshow(
        timing_grid.to_numpy(),
        cmap="Reds",
        vmin=0,
        vmax=timing_max,
        aspect="auto",
    )

    axes[0].set_title("Trials flagged for review (%)")
    axes[1].set_title("Trial windows failing timing checks")

    for ax in axes:
        ax.set_xticks(range(len(RUN_ORDER)))
        ax.set_xticklabels(RUN_LABELS)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Recording run")
        ax.set_ylabel("Participant")

    for row in range(len(labels)):
        for column in range(len(RUN_ORDER)):
            percent = flag_grid.iloc[row, column]
            failures = timing_grid.iloc[row, column]

            axes[0].text(
                column,
                row,
                f"{percent:.0f}%",
                ha="center",
                va="center",
                color="white" if percent >= 60 else "black",
                fontsize=8,
            )

            axes[1].text(
                column,
                row,
                str(int(failures)),
                ha="center",
                va="center",
                color="white" if failures / timing_max > 0.5 else "black",
                fontsize=8,
            )

    fig.colorbar(
        quality_image, ax=axes[0], label="Flagged trials (%)"
    )

    # Use integer ticks because timing failures are counts.
    colorbar = fig.colorbar(
        timing_image, ax=axes[1], label="Failed windows"
    )
    colorbar.set_ticks(range(int(timing_max) + 1))

    fig.savefig(
        REPORT_DIR / "all_participants_run_overview.png",
        dpi=200,
    )
    plt.close(fig)


def remove_old_summaries():
    """Remove only obsolete generated summaries and per-person plots."""
    patterns = [
        "S[0-9][0-9][0-9]_quality_summary.csv",
        "S[0-9][0-9][0-9]_combined_summary.csv",
        "S[0-9][0-9][0-9]_quality_overview.png",
    ]

    obsolete_files = set()

    for pattern in patterns:
        obsolete_files.update(REPORT_DIR.glob(pattern))

    obsolete_files.add(
        REPORT_DIR / "all_participants_quality_summary.csv"
    )

    for path in sorted(obsolete_files):
        if path.is_file():
            path.unlink()
            print("Deleted old output:", path.name)


def main():
    trial_files = sorted(
        REPORT_DIR.glob("S[0-9][0-9][0-9]_trial_checks.csv")
    )

    if not trial_files:
        raise FileNotFoundError("Run check_quality.py first.")

    run_summary = pd.concat(
        [summarize_participant(path) for path in trial_files],
        ignore_index=True,
    ).sort_values(["participant", "run"])

    participants = run_summary.groupby("participant").agg(
        annotated_tasks=("annotated_tasks", "sum"),
        extracted_trials=("extracted_trials", "sum"),
        left_trials=("left_trials", "sum"),
        right_trials=("right_trials", "sum"),
        windows_passed=("windows_passed", "sum"),
        windows_failed=("windows_failed", "sum"),
        tasks_not_extracted=("tasks_not_extracted", "sum"),
        min_task_duration_s=("min_task_duration_s", "min"),
        max_task_duration_s=("max_task_duration_s", "max"),
        flagged_trials=("flagged_trials", "sum"),
        unflagged_trials=("unflagged_trials", "sum"),
    )

    participants["all_windows_fit"] = (
        participants["windows_failed"] == 0
    )
    participants["flagged_percent"] = (
        100
        * participants["flagged_trials"]
        / participants["extracted_trials"]
    )
    participants["unflagged_percent"] = (
        100 - participants["flagged_percent"]
    )

    # Save the two final tables.
    run_summary.to_csv(
        REPORT_DIR / "all_participants_run_summary.csv",
        index=False,
    )
    participants.to_csv(
        REPORT_DIR / "quality_by_participant.csv"
    )

    # Save both graphs.
    plot_participants(participants)
    plot_runs(run_summary)

    print("\nRun-by-run summary:")
    print(
        run_summary[
            [
                "participant",
                "run",
                "condition",
                "extracted_trials",
                "left_trials",
                "right_trials",
                "windows_failed",
                "tasks_not_extracted",
                "flagged_trials",
            ]
        ].to_string(index=False)
    )

    print("\nParticipant summary:")
    print(participants.to_string())

    # Clean up only after all new tables and graphs are saved.
    remove_old_summaries()

    print(f"\nSummarized {len(participants)} participants.")
    print("Reports saved to:", REPORT_DIR)


if __name__ == "__main__":
    main()