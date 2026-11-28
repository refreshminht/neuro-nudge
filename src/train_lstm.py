"""Train the LSTM on motor execution EEG (left vs. right fist).

Same settings as the temporal CNN (model/train_cnn_expanded.py):
  seed 42, Adam (lr 0.001), batch size 16, cross-entropy loss,
  max 50 epochs, early stopping patience 10, and the checkpoint with the
  lowest execution validation loss is kept.

Only train.npz (R03) and val.npz (R07) are used here. The held-out
execution test and the imagery runs are evaluated separately by
src/evaluate_lstm.py.

Run:  python -m src.train_lstm --data-dir outputs_3s_20 --out-dir results/lstm_20
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model.data_loader import create_dataloaders
from src.lstm_model import EEGLSTM

SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_EPOCHS = 50
PATIENCE = 10


@torch.no_grad()
def predict(model, loader):
    """Return true labels, predicted labels and mean loss for one split."""
    model.eval()
    y_true, y_pred, total_loss = [], [], 0.0
    for X, y in loader:
        logits = model(X.transpose(1, 2))  # (batch, 64, 480) -> (batch, 480, 64)
        total_loss += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        y_true += y.tolist()
        y_pred += logits.argmax(1).tolist()
    return np.array(y_true), np.array(y_pred), total_loss / len(y_true)


def compute_metrics(y_true, y_pred):
    """Same metrics as model/evaluate_cnn.py, plus predicted-class counts."""
    cm = np.zeros((2, 2), dtype=int)  # rows = true, columns = predicted
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    recall = cm.diagonal() / cm.sum(axis=1)
    return {'n_trials': int(len(y_true)),
            'accuracy': float((y_true == y_pred).mean()),
            'balanced_accuracy': float(recall.mean()),
            'left_recall': float(recall[0]),
            'right_recall': float(recall[1]),
            'confusion_matrix': cm.tolist(),
            'predicted_counts': {'left': int((y_pred == 0).sum()), 'right': int((y_pred == 1).sum())}}


def plot_loss_curves(history, best_epoch, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h['train_loss'] for h in history], label='Train (execution R03)')
    ax.plot(epochs, [h['val_loss'] for h in history], label='Validation (execution R07)')
    ax.axvline(best_epoch, color='grey', linestyle=':', label=f'Selected epoch ({best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-entropy loss')
    ax.set_title('LSTM training curves')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='outputs_3s_20')
    p.add_argument('--out-dir', default='results/lstm_20')
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    loaders = create_dataloaders(data_dir=args.data_dir, batch_size=BATCH_SIZE, seed=SEED)
    train_loader, val_loader = loaders['train'], loaders['val']
    print(f'Training trials: {len(train_loader.dataset)}  Validation trials: {len(val_loader.dataset)}')

    model = EEGLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_loss, best_epoch, history = float('inf'), 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            optimizer.zero_grad()
            logits = model(X.transpose(1, 2))
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        y_true, y_pred, val_loss = predict(model, val_loader)
        val_acc = float((y_true == y_pred).mean())
        history.append({'epoch': epoch, 'train_loss': loss_sum / total, 'train_accuracy': correct / total,
                        'val_loss': val_loss, 'val_accuracy': val_acc})
        print(f'Epoch {epoch:02d}/{MAX_EPOCHS} | train loss {loss_sum / total:.4f} | '
              f'train acc {correct / total:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}')

        if val_loss < best_val_loss:  # keep the checkpoint with the lowest validation loss
            best_val_loss, best_epoch = val_loss, epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_loss,
                        'val_accuracy': val_acc, 'seed': SEED, 'learning_rate': LEARNING_RATE,
                        'batch_size': BATCH_SIZE, 'model_name': 'EEGLSTM',
                        'class_names': ['left', 'right']}, out_dir / 'best_model.pt')
        elif epoch - best_epoch >= PATIENCE:
            print('Early stopping triggered.')
            break

    with open(out_dir / 'training_history.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_loss_curves(history, best_epoch, out_dir / 'loss_curves.png')

    # Validation check for the selected checkpoint
    checkpoint = torch.load(out_dir / 'best_model.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    y_true, y_pred, _ = predict(model, val_loader)
    val = compute_metrics(y_true, y_pred)
    with open(out_dir / 'validation_metrics.json', 'w') as f:
        json.dump({'model': 'EEGLSTM', 'data_dir': str(args.data_dir), 'checkpoint_epoch': best_epoch,
                   'epochs_run': len(history), 'val_loss': best_val_loss, 'validation': val}, f, indent=2)

    print(f'\nBest epoch: {best_epoch}  (validation loss {best_val_loss:.4f})')
    print(f'Validation accuracy: {val["accuracy"]:.3f}  balanced accuracy: {val["balanced_accuracy"]:.3f}')
    print(f'Recall  left: {val["left_recall"]:.3f}  right: {val["right_recall"]:.3f}')
    print(f'Predicted classes: {val["predicted_counts"]}  confusion matrix: {val["confusion_matrix"]}')
    print(f'Saved to {out_dir}')


if __name__ == '__main__':
    main()
