"""Publication-style temporal CNN figures for the NEURO 240 five-subject pilot.

Run at repository root: python -m model.plot_cnn
Inputs: results/cnn/training_history.csv, results/cnn/evaluation_metrics.json
Outputs: PNG (300 dpi) and vector PDF files in results/cnn/figures/
No network access, retraining, or test-set inference occurs here.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # File generation, even without a GUI
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "cnn"
FIGURES = RESULTS / "figures"
HISTORY = RESULTS / "training_history.csv"
METRICS = RESULTS / "evaluation_metrics.json"

# Quiet, high-contrast scientific palette; consistent across outputs.
NAVY = "#243A53"
BLUE = "#2878A3"
AMBER = "#CE7B28"
PURPLE = "#8067A5"
GRAY = "#737C86"
GRID = "#E5E9ED"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "normal",
    "axes.labelsize": 9.5,
    "axes.edgecolor": NAVY,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def save(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = FIGURES / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
        print(f"Saved: {path.relative_to(ROOT)}")
    plt.close(fig)


def load_history():
    if not HISTORY.exists():
        raise FileNotFoundError(f"Missing {HISTORY}; run CNN training first.")
    with HISTORY.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Training history CSV is empty")
    columns = ("epoch", "train_loss", "val_loss")
    for name in columns:
        if name not in rows[0]:
            raise ValueError(f"Training CSV lacks column: {name}")
    return {name: np.asarray([float(r[name]) for r in rows]) for name in columns}


def load_metrics():
    if not METRICS.exists():
        raise FileNotFoundError(
            f"Missing {METRICS}; first run: python -m model.evaluate_cnn"
        )
    with METRICS.open(encoding="utf-8") as handle:
        data = json.load(handle)
    for name in ("execution", "imagery"):
        d = data[name]
        cm = np.asarray(d["confusion_matrix"], dtype=int)
        if cm.shape != (2, 2) or np.any(cm < 0):
            raise ValueError(f"Invalid {name} confusion matrix")
        if int(cm.sum()) != int(d["n_trials"]):
            raise ValueError(f"{name}: matrix count and n_trials disagree")
        supports = cm.sum(axis=1)
        if np.any(supports == 0):
            raise ValueError(f"{name}: need both true classes for balanced accuracy")
        balanced_acc = float(np.mean(np.diag(cm) / supports))
        if not np.isclose(balanced_acc, float(d["balanced_accuracy"]), atol=1e-5):
            raise ValueError(f"{name}: balanced accuracy does not match confusion matrix")
    return data


def tidy_axis(ax, horizontal_grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if horizontal_grid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(length=3, color=GRAY)


def training_plot(history, metrics):
    epochs = history["epoch"]
    best_epoch = int(metrics["checkpoint_epoch"])
    chosen = np.flatnonzero(np.isclose(epochs, best_epoch))
    if chosen.size != 1:
        raise ValueError(f"Checkpoint epoch {best_epoch} missing from history")
    ix = int(chosen[0])
    loss = float(history["val_loss"][ix])
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    ax.plot(epochs, history["train_loss"], label="Training", color=BLUE,
            linewidth=1.9, marker="o", markersize=3.6)
    ax.plot(epochs, history["val_loss"], label="Validation", color=AMBER,
            linewidth=1.9, marker="s", markersize=3.6)
    ax.axvline(best_epoch, color=GRAY, linestyle=":", linewidth=1.2)
    ax.scatter([best_epoch], [loss], color=AMBER, s=45, zorder=4)
    ax.annotate(f"Best validation epoch: {best_epoch}", xy=(best_epoch, loss),
                xytext=(0.22, 0.90), textcoords="axes fraction",
                arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.8},
                fontsize=9, va="center")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss",
           title="Temporal CNN: training and validation loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(epochs.min() - 0.15, epochs.max() + 0.25)
    ax.set_title("Temporal CNN: training and validation loss", pad=7, fontweight="normal")
    ax.legend(frameon=False, loc="lower left")
    tidy_axis(ax, horizontal_grid=True)
    fig.tight_layout()
    save(fig, "fig1_training_dynamics")


def class_collapse(metrics):
    cms = [np.asarray(metrics[k]["confusion_matrix"], dtype=int)
           for k in ("execution", "imagery")]
    all_right = all(int(cm[:, 0].sum()) == 0 for cm in cms)
    all_left = all(int(cm[:, 1].sum()) == 0 for cm in cms)
    if all_right:
        return "All trials predicted as right fist"
    if all_left:
        return "All trials predicted as left fist"
    return None


def balanced_accuracy_panel(ax, metrics, show_title=True):
    values = [100 * float(metrics[k]["balanced_accuracy"])
              for k in ("execution", "imagery")]
    x = np.array([0, 1])
    ax.bar(x, values, width=0.53, color=(BLUE, PURPLE), alpha=0.9, zorder=2)
    ax.axhline(50, color=GRAY, linewidth=1.1, linestyle="--", label="Chance level (50%)")
    for xi, value in zip(x, values):
        ax.text(xi, min(97, value + 3.2), f"{value:.1f}%", ha="center",
                va="bottom", fontsize=10, fontweight="semibold", color=NAVY)
    ax.set_xticks(x, ["Execution", "Imagery"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Balanced accuracy (%)")
    if show_title:
        ax.set_title("Cross-condition performance", pad=7, fontweight="normal")
    tidy_axis(ax, horizontal_grid=True)
    # The bar colors are directly identified by the x-axis labels.
    # Keep only the chance-level reference in the legend.
    ax.legend(frameon=False, loc="upper right", fontsize=8)


def figure2(metrics):
    """Standalone balanced accuracy plot; methodological caveats belong in caption."""
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    balanced_accuracy_panel(ax, metrics)
    # Do not place the single-class prediction warning inside the plot.
    # The caption should explain it, alongside the confusion matrices.
    fig.tight_layout()
    save(fig, "fig2_cross_condition_generalization")


def confusion_panel(ax, cm, heading, panel_letter=None):
    cm = np.asarray(cm, dtype=int)
    support = cm.sum(axis=1, keepdims=True)
    fractions = np.divide(cm, support, out=np.zeros_like(cm, dtype=float), where=support > 0)
    im = ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues", aspect="equal")
    ax.set_xticks([0, 1], ["Left", "Right"])
    ax.set_yticks([0, 1], ["Left", "Right"])
    ax.set(xlabel="Predicted class", ylabel="True class")
    prefix = f"{panel_letter}  " if panel_letter else ""
    ax.set_title(f"{prefix}{heading} (n = {cm.sum()})", fontsize=10, pad=6,
                 fontweight="normal")
    for i in range(2):
        for j in range(2):
            pct = 100 * fractions[i, j]
            color = "white" if fractions[i, j] >= 0.57 else NAVY
            ax.text(j, i, f"{cm[i, j]}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=9, color=color, fontweight="medium")
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def add_shared_colorbar(fig, axes, shrink=0.8, pad=0.025):
    """One shared scale for both matrices; a second bar would be redundant."""
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="Blues")
    sm.set_array([])
    cb = fig.colorbar(sm, ax=list(axes), location="right",
                      shrink=shrink, fraction=0.042, pad=pad)
    cb.set_label("Proportion within true class", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return cb


def figure3(metrics):
    """Two confusion matrices with one shared normalization/color scale."""
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.85), layout="constrained")
    fig.suptitle("CNN confusion matrices across motor conditions",
                 fontsize=11, fontweight="normal")
    for ax, key, name, panel in zip(axes, ("execution", "imagery"),
                                    ("Execution", "Imagery"), ("A", "B")):
        confusion_panel(ax, metrics[key]["confusion_matrix"], name, panel)
    add_shared_colorbar(fig, axes, shrink=0.80, pad=0.025)
    save(fig, "fig3_confusion_matrices")


def composite_figure(metrics):
    """Three-panel report figure: BA + two confusion matrices."""
    fig = plt.figure(figsize=(11.6, 4.05), layout="constrained")
    fig.suptitle("CNN classification performance across motor conditions",
                 fontsize=11, fontweight="normal")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.10, 1, 1), wspace=0.10)
    ax_ba = fig.add_subplot(grid[0, 0])
    ax_ex = fig.add_subplot(grid[0, 1])
    ax_im = fig.add_subplot(grid[0, 2])
    balanced_accuracy_panel(ax_ba, metrics, show_title=False)
    ax_ba.set_title("A  Balanced accuracy", pad=6, fontsize=10,
                    fontweight="normal")
    confusion_panel(ax_ex, metrics["execution"]["confusion_matrix"], "Execution", "B")
    confusion_panel(ax_im, metrics["imagery"]["confusion_matrix"], "Imagery", "C")
    add_shared_colorbar(fig, [ax_ex, ax_im], shrink=0.79, pad=0.02)
    save(fig, "fig2_3_combined_performance")


def main():
    history = load_history()
    metrics = load_metrics()
    print("Generating figures from existing records; no model training or inference.")
    training_plot(history, metrics)
    figure2(metrics)
    figure3(metrics)
    composite_figure(metrics)
    collapse = class_collapse(metrics)
    if collapse:
        print("Interpretation check:", collapse)
        print("Equal 50% balanced accuracy does NOT demonstrate transfer.")
    print("Done: three individual figures plus one report-ready composite.")


if __name__ == "__main__":
    main()