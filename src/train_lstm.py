"""Train the LSTM on motor execution, then test on held-out execution and imagery.

  train           -> fit weights
  val             -> early stopping / pick the best epoch
  execution_test  -> evaluate only
  imagery_test    -> evaluate only (never used for training or model selection)

Run:  python -m src.train_lstm --data-dir data/processed
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.lstm_model import EEGLSTM

# split name -> file pattern (matches both train.npz and S001_train.npz)
PATTERNS = {'train': '*train.npz', 'val': '*val*.npz',
            'execution_test': '*execution_test.npz', 'imagery_test': '*imagery_test.npz'}


def load_split(data_dir, pattern):
    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f'no {pattern} files in {data_dir}')
    X = np.concatenate([np.load(f)['X'] for f in files])
    y = np.concatenate([np.load(f)['y'] for f in files])
    # stored as [N, 64, time]; the LSTM wants [N, time, 64]
    return torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32), torch.tensor(y, dtype=torch.long)


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    logits = model(X)
    pred = logits.argmax(1)
    cm = torch.zeros(2, 2, dtype=torch.long)   # rows = true, cols = predicted
    for t, q in zip(y, pred):
        cm[t, q] += 1
    recall = cm.diag() / cm.sum(1).clamp(min=1)
    return {'loss': nn.functional.cross_entropy(logits, y).item(),
            'accuracy': (pred == y).float().mean().item(),
            'balanced_accuracy': recall.mean().item(),
            'confusion_matrix': cm.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/processed')
    p.add_argument('--out-dir', default='results/lstm')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    data = {name: load_split(args.data_dir, pat) for name, pat in PATTERNS.items()}
    for name, (X, y) in data.items():
        print(f'{name:15s} {tuple(X.shape)}  left/right = {torch.bincount(y, minlength=2).tolist()}')
    Xtr, ytr = data['train']

    model = EEGLSTM(n_channels=Xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val, best_state, best_epoch, history = float('inf'), None, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), args.batch_size):
            idx = order[i:i + args.batch_size]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

        train, val = evaluate(model, Xtr, ytr), evaluate(model, *data['val'])
        history.append({'epoch': epoch, 'train_loss': train['loss'], 'val_loss': val['loss']})
        print(f'epoch {epoch:3d}  train loss {train["loss"]:.3f}  val loss {val["loss"]:.3f}')
        if val['loss'] < best_val:  # keep the weights with the lowest validation loss
            best_val, best_state, best_epoch = val['loss'], copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= args.patience:  # early stopping
            break

    model.load_state_dict(best_state)
    results = {name: evaluate(model, *data[name]) for name in ('execution_test', 'imagery_test')}
    print(f'\nBest epoch: {best_epoch}')
    for name, r in results.items():
        print(f'{name:15s} acc {r["accuracy"]:.3f}  balanced acc {r["balanced_accuracy"]:.3f}  '
              f'confusion [true x pred] {r["confusion_matrix"]}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / 'lstm.pt')
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({'best_epoch': best_epoch, 'results': results, 'history': history}, f, indent=2)
    print(f'Saved model and metrics to {out_dir}')


if __name__ == '__main__':
    main()
