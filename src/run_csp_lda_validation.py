"""NEURO 240: CSP + LDA baseline on held-out execution validation run.

Run from repository root:
    python -m src.run_csp_lda_validation --data-dir outputs_3s_20

Reads ONLY train.npz (R03) and val.npz (R07). Does not load reserved
execution_test.npz (R11) or imagery_test.npz (R04/R08/R12).
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline


LABEL_NAMES = ["Left fist", "Right fist"]


def load_partition(folder: Path, name: str, expected_run: int):
    file_path = folder / f"{name}.npz"
    metadata_path = folder / f"{name}_metadata.csv"
    if not file_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {file_path} or {metadata_path}. "
            "Run src.split_data for outputs_3s_20 first."
        )
    with np.load(file_path, allow_pickle=False) as data:
        X = data["X"].astype(np.float64)
        y = data["y"].astype(np.int64)
        channels = data["channels"].tolist()
        start, end = float(data["start"]), float(data["end"])
        sfreq = float(data["sfreq"])

    meta = pd.read_csv(metadata_path)
    if X.ndim != 3 or X.shape[1:] != (64, 480):
        raise ValueError(f"{name}: unexpected shape {X.shape}; expected (N, 64, 480)")
    if len(meta) != len(y) or len(X) != len(y):
        raise ValueError(f"{name}: inconsistent number of epochs/metadata rows")
    if not np.array_equal(meta["label"].to_numpy(dtype=np.int64), y):
        raise ValueError(f"{name}: label order differs from metadata")
    if not (meta["run"].to_numpy() == expected_run).all():
        raise ValueError(f"{name}: expected only run R{expected_run:02d}")
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError(f"{name}: nonfinite EEG samples or invalid labels")
    if len(np.unique(y)) != 2:
        raise ValueError(f"{name}: both motor classes must be present")
    if not np.isclose(start, 0.5) or not np.isclose(end, 3.5) or not np.isclose(sfreq, 160):
        raise ValueError(f"{name}: unexpected window/sampling rate")
    return X, y, channels, meta


def save_confusion_plot(cm, path):
    fig, ax = plt.subplots(figsize=(5.3, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(int(val)), ha="center", va="center",
                fontsize=15, color="white" if val > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], LABEL_NAMES)
    ax.set_yticks([0, 1], LABEL_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("CSP + LDA: held-out execution validation (R07)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Trials")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_subject_plot(by_subject, path):
    df = by_subject.sort_values("subject")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(df["subject"], df["balanced_accuracy"] * 100)
    ax.axvline(50, linestyle="--", linewidth=1.5, color="black", label="Chance reference (50%)")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Balanced accuracy (%)")
    ax.set_ylabel("Participant")
    ax.set_title("CSP + LDA: execution validation accuracy by participant")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_feature_plot(model, X_train, y_train, X_val, y_val, path):
    csp = model.named_steps["csp"]
    f_train = csp.transform(X_train)
    f_val = csp.transform(X_val)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for ax, feats, labels, title in [
        (axes[0], f_train, y_train, "Training: R03"),
        (axes[1], f_val, y_val, "Validation: R07"),
    ]:
        for cls, color in [(0, "tab:blue"), (1, "tab:orange")]:
            ax.scatter(feats[labels == cls, 0], feats[labels == cls, 1],
                       alpha=0.45, s=17, label=LABEL_NAMES[cls], color=color)
        ax.set_title(title)
        ax.set_xlabel("CSP feature 1 (log variance)")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("CSP feature 2 (log variance)")
    axes[1].legend(loc="best")
    fig.suptitle("First two CSP features learned using execution training trials only")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("outputs_3s_20"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/csp_lda_20"))
    args = parser.parse_args()

    X_train, y_train, ch_train, train_meta = load_partition(args.data_dir, "train", 3)
    X_val, y_val, ch_val, val_meta = load_partition(args.data_dir, "val", 7)
    if ch_train != ch_val:
        raise ValueError("Training and validation channel orders differ")

    print("=" * 64)
    print("NEURO 240 | CSP + LDA VALIDATION BASELINE")
    print("=" * 64)
    print(f"Training (R03):   {X_train.shape} | classes {np.bincount(y_train)}")
    print(f"Validation (R07): {X_val.shape} | classes {np.bincount(y_val)}")
    print("CSP: 4 components, regularization=0.1, log-variance features")
    print("LDA: least-squares with automatic shrinkage")
    print("No test or imagery files will be loaded.\n")

    model = Pipeline([
        ("csp", CSP(n_components=4, reg=0.1, log=True,
                    norm_trace=False, cov_est="concat")),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    train_bal = balanced_accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    val_bal = balanced_accuracy_score(y_val, val_pred)
    cm = confusion_matrix(y_val, val_pred, labels=[0, 1])

    print(f"TRAIN accuracy: {train_acc:.4f} | balanced accuracy: {train_bal:.4f}")
    print(f"VAL   accuracy: {val_acc:.4f} | balanced accuracy: {val_bal:.4f}")
    print(f"Validation prediction counts [left, right]: {np.bincount(val_pred, minlength=2).tolist()}")
    print("Validation confusion matrix (rows=true, columns=predicted):\n", cm)
    print("\n" + classification_report(y_val, val_pred, target_names=LABEL_NAMES, digits=3, zero_division=0))

    results = args.results_dir
    results.mkdir(parents=True, exist_ok=True)
    pred_frame = val_meta[["subject", "run", "label"]].copy()
    pred_frame = pred_frame.rename(columns={"label": "true_label"})
    pred_frame["predicted_label"] = val_pred
    pred_frame["correct"] = (y_val == val_pred)
    pred_frame.to_csv(results / "validation_predictions.csv", index=False)

    subject_rows = []
    for subject, group in pred_frame.groupby("subject", sort=True):
        yt = group["true_label"].to_numpy()
        yp = group["predicted_label"].to_numpy()
        subject_rows.append({
            "subject": subject,
            "n_trials": len(group),
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        })
    by_subject = pd.DataFrame(subject_rows)
    by_subject.to_csv(results / "validation_by_subject.csv", index=False)

    metrics = {
        "protocol": "R03 train; R07 validation; R11 and R04/R08/R12 reserved",
        "n_subjects": int(pred_frame["subject"].nunique()),
        "n_train": int(len(y_train)), "n_validation": int(len(y_val)),
        "csp_components": 4, "csp_regularization": 0.1,
        "lda_solver": "lsqr", "lda_shrinkage": "auto",
        "train_accuracy": float(train_acc),
        "train_balanced_accuracy": float(train_bal),
        "validation_accuracy": float(val_acc),
        "validation_balanced_accuracy": float(val_bal),
        "validation_confusion_matrix": cm.tolist(),
        "validation_predicted_counts": np.bincount(val_pred, minlength=2).tolist(),
        "validation_true_counts": np.bincount(y_val, minlength=2).tolist(),
    }
    (results / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(model, results / "csp_lda_model.joblib")
    save_confusion_plot(cm, results / "validation_confusion_matrix.png")
    save_subject_plot(by_subject, results / "validation_accuracy_by_subject.png")
    save_feature_plot(model, X_train, y_train, X_val, y_val,
                      results / "csp_feature_scatter.png")
    print(f"\nSaved model, metrics, predictions and 3 figures in: {results}")
    print("These are validation results, not final held-out test results.")


if __name__ == "__main__":
    main()
