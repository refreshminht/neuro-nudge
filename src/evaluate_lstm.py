"""Evaluate the selected LSTM checkpoint on held-out execution and imagery.

No training or fine-tuning happens here. The checkpoint saved by
src/train_lstm.py is tested once on:
  execution_test.npz  (execution R11)
  imagery_test.npz    (imagery R04, R08, R12)

Run:  python -m src.evaluate_lstm --data-dir outputs_3s_20 --out-dir results/lstm_20
"""
import argparse
import csv
import json
from pathlib import Path

import torch

from model.data_loader import create_dataloaders
from src.lstm_model import EEGLSTM
from src.train_lstm import BATCH_SIZE, compute_metrics, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='outputs_3s_20')
    p.add_argument('--out-dir', default='results/lstm_20')
    args = p.parse_args()
    out_dir = Path(args.out_dir)

    checkpoint = torch.load(out_dir / 'best_model.pt', weights_only=True)
    model = EEGLSTM()
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]} (validation loss {checkpoint["val_loss"]:.4f})')

    loaders = create_dataloaders(data_dir=args.data_dir, batch_size=BATCH_SIZE)
    results = {'model': 'EEGLSTM', 'data_dir': str(args.data_dir),
               'checkpoint_epoch': int(checkpoint['epoch']), 'class_names': ['Left fist', 'Right fist']}

    for condition, split in (('execution', 'execution_test'), ('imagery', 'imagery_test')):
        y_true, y_pred, _ = predict(model, loaders[split])
        m = results[condition] = compute_metrics(y_true, y_pred)
        with open(out_dir / f'{condition}_predictions.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trial', 'true_label', 'predicted_label', 'correct'])
            for i, (t, q) in enumerate(zip(y_true, y_pred), start=1):
                writer.writerow([i, int(t), int(q), int(t == q)])
        print(f'\n{condition}: {m["n_trials"]} trials')
        print(f'  accuracy {m["accuracy"]:.3f}  balanced accuracy {m["balanced_accuracy"]:.3f}')
        print(f'  recall  left {m["left_recall"]:.3f}  right {m["right_recall"]:.3f}')
        print(f'  predicted classes {m["predicted_counts"]}  confusion matrix {m["confusion_matrix"]}')

    results['transfer_gap'] = results['execution']['balanced_accuracy'] - results['imagery']['balanced_accuracy']
    with open(out_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nTransfer gap (execution - imagery balanced accuracy): {results["transfer_gap"] * 100:.1f} points')
    print(f'Saved to {out_dir}')


if __name__ == '__main__':
    main()
