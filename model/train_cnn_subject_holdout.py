"""Exploratory CNN experiment with disjoint training and validation participants.

Run from neuro-nudge repository root:
  python -m model.train_cnn_subject_holdout

Uses only train.npz and val.npz. No test or imagery data are accessed.
Uses the existing TemporalCNN and the same original training functions/settings.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.data_loader import EEGDataset
from model.temporal_cnn import TemporalCNN
from model.train_cnn import train_one_epoch, validate, save_history

SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_EPOCHS = 50
PATIENCE = 10
DEVICE = torch.device('cpu')
DATA_DIR = Path('outputs_3s_subject_holdout_20')
RESULTS_DIR = Path('results/cnn_subject_holdout_20')


def main():
    if not (DATA_DIR / 'split_manifest.json').is_file():
        raise FileNotFoundError('Split manifest not found. Run src.build_subject_holdout first.')
    with (DATA_DIR / 'split_manifest.json').open(encoding='utf-8') as file:
        manifest = json.load(file)
    if manifest['status'] != 'exploratory_follow_up_not_pristine_holdout':
        raise ValueError('Unexpected split manifest/status')
    if RESULTS_DIR.exists() and any(RESULTS_DIR.iterdir()):
        raise FileExistsError(f'{RESULTS_DIR} is not empty. Do not overwrite prior results.')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_ds = EEGDataset(DATA_DIR / 'train.npz')
    val_ds = EEGDataset(DATA_DIR / 'val.npz')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TemporalCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print('=' * 65)
    print('CNN | EXPLORATORY SUBJECT-HOLDOUT EXECUTION VALIDATION')
    print('Train S001-S016 R03+R07; validate S017-S020 R03+R07')
    print(f'Training examples: {len(train_ds)}; validation examples: {len(val_ds)}')
    print('No test or imagery files used. CPU training.')
    print('=' * 65)

    RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    history = []
    best_loss, best_epoch, epochs_without_improvement = float('inf'), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        va_loss, va_acc = validate(model, val_loader, criterion, DEVICE)
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_accuracy': tr_acc,
                        'val_loss': va_loss, 'val_accuracy': va_acc})
        print(f'Epoch {epoch:02d}/{MAX_EPOCHS} | Train loss {tr_loss:.4f} acc {tr_acc:.4f} '
              f'| Val loss {va_loss:.4f} acc {va_acc:.4f}')
        if va_loss < best_loss:
            best_loss, best_epoch, epochs_without_improvement = va_loss, epoch, 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': va_loss, 'val_accuracy': va_acc,
                'seed': SEED, 'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE, 'model_name': 'TemporalCNN',
                'class_names': ['left', 'right'],
                'protocol': 'exploratory_subject_holdout_20',
            }, RESULTS_DIR / 'best_model.pt')
            print('  -> Best checkpoint saved')
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print('Early stopping triggered.')
            break
    save_history(history, RESULTS_DIR / 'training_history.csv')
    (RESULTS_DIR / 'experiment_config.json').write_text(json.dumps({
        'seed': SEED, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE,
        'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
        'best_epoch': best_epoch, 'best_val_loss': best_loss,
        'train_subjects': manifest['train_subjects'],
        'validation_subjects': manifest['validation_subjects'],
        'execution_runs': [3, 7], 'status': manifest['status'],
    }, indent=2), encoding='utf-8')
    print(f'\nBest epoch: {best_epoch}; validation loss: {best_loss:.4f}')
    print(f'Results saved to: {RESULTS_DIR}')
    print('No execution-test or imagery-test data were loaded.')


if __name__ == '__main__':
    main()
