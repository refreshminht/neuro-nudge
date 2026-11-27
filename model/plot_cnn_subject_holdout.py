"""Plot learning curves and selected-checkpoint validation confusion matrix.

Run from repository root:
    python -m model.plot_cnn_subject_holdout
Reads only training history, best checkpoint, and validation arrays.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score
from model.temporal_cnn import TemporalCNN

DATA_DIR = Path('outputs_3s_subject_holdout_20')
RESULTS_DIR = Path('results/cnn_subject_holdout_20')


def main():
    hist = pd.read_csv(RESULTS_DIR / 'training_history.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(hist['epoch'], hist['train_loss'], label='Train')
    axes[0].plot(hist['epoch'], hist['val_loss'], label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='CNN learning curves — loss')
    axes[0].legend()
    axes[1].plot(hist['epoch'], 100 * hist['train_accuracy'], label='Train')
    axes[1].plot(hist['epoch'], 100 * hist['val_accuracy'], label='Validation')
    axes[1].axhline(50, color='gray', linestyle='--', label='50% reference')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)', title='CNN learning curves — accuracy')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Exploratory subject-held-out execution experiment')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'training_curves.png', dpi=200)
    plt.close(fig)

    with np.load(DATA_DIR / 'val.npz', allow_pickle=False) as data:
        x = torch.tensor(data['X'], dtype=torch.float32)
        y = data['y'].astype(int)
    ckpt = torch.load(RESULTS_DIR / 'best_model.pt', map_location='cpu', weights_only=False)
    model = TemporalCNN().eval()
    model.load_state_dict(ckpt['model_state_dict'])
    with torch.no_grad():
        pred = np.concatenate([model(batch).argmax(dim=1).numpy()
                               for batch in x.split(32)])
    cm = confusion_matrix(y, pred, labels=[0, 1])
    acc = accuracy_score(y, pred)
    bal = balanced_accuracy_score(y, pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap='Blues')
    for (i, j), count in np.ndenumerate(cm):
        ax.text(j, i, str(int(count)), ha='center', va='center', fontsize=15,
                color='white' if count > cm.max() / 2 else 'black')
    ax.set_xticks([0, 1], ['Left', 'Right'])
    ax.set_yticks([0, 1], ['Left', 'Right'])
    ax.set(xlabel='Predicted class', ylabel='True class',
           title=f'CNN validation (unseen subjects)\nBalanced accuracy: {bal*100:.1f}%')
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'validation_confusion_matrix.png', dpi=200)
    plt.close(fig)

    pd.DataFrame({'true_label': y, 'predicted_label': pred}).to_csv(
        RESULTS_DIR / 'validation_predictions.csv', index=False)
    metrics = {'validation_accuracy': acc, 'validation_balanced_accuracy': bal,
               'validation_confusion_matrix': cm.tolist(),
               'predicted_counts': np.bincount(pred, minlength=2).tolist(),
               'selected_epoch': ckpt['epoch'],
               'status': 'exploratory_follow_up_not_pristine_holdout'}
    (RESULTS_DIR / 'validation_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'Validation accuracy: {acc:.4f}; balanced accuracy: {bal:.4f}')
    print('Predicted left/right:', metrics['predicted_counts'])
    print('Confusion matrix:\n', cm)
    print(f'Saved 2 PNG figures, metrics JSON, and predictions CSV to: {RESULTS_DIR}')
    print('No test or imagery data accessed.')


if __name__ == '__main__':
    main()
