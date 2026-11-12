"""Leakage-safe run split, training-only normalization, shared CNN/LSTM arrays."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def build_split(output_dir, subjects):
    buckets = {'train': [], 'val': [], 'execution_test': [], 'imagery_test': []}
    labels = {key: [] for key in buckets}
    provenance = {key: [] for key in buckets}
    metadata_parts = {key: [] for key in buckets}
    expected_input_shape = None
    interval_ref = None
    channels_ref = None
    for s in subjects:
        with np.load(output_dir / f'S{s:03d}_epochs.npz') as z:
            X, y, channels = z['X'], z['y'], z['channels']
            window = (float(z['start']), float(z['end']))
        if interval_ref is None: interval_ref = window
        elif not np.allclose(interval_ref, window): raise ValueError('Epoch window differs across subjects')
        if expected_input_shape is None: expected_input_shape = X.shape[1:]
        elif X.shape[1:] != expected_input_shape: raise ValueError('Input shape differs across subjects')
        if channels_ref is None:
            channels_ref = channels
        elif not np.array_equal(channels_ref, channels):
            raise ValueError('Channel order mismatch across subjects')
        meta = pd.read_csv(output_dir / f'S{s:03d}_metadata.csv')
        meta = meta[meta.retained.astype(bool)].reset_index(drop=True)
        assert len(meta) == len(y) == len(X)
        for run, partition in [(3,'train'),(7,'val'),(11,'execution_test'),
                               (4,'imagery_test'),(8,'imagery_test'),(12,'imagery_test')]:
            ids = np.flatnonzero(meta.run.to_numpy() == run)
            if len(ids):
                buckets[partition].append(X[ids])
                labels[partition].append(y[ids])
                provenance[partition].extend([f'{s}:{run}:{int(meta.iloc[i].annotation_index)}' for i in ids])
                metadata_parts[partition].append(meta.iloc[ids].copy())
    if not buckets['train']:
        raise RuntimeError('No training data')
    train = np.concatenate(buckets['train'])
    # Statistics use execution training run only, pooled across subjects/trials/time.
    mean = train.mean(axis=(0,2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = train.std(axis=(0,2), keepdims=True, dtype=np.float64).astype(np.float32)
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Zero/nonfinite training channel standard deviation; inspect EEG and QC')
    all_keys = [k for keys in provenance.values() for k in keys]
    assert len(all_keys) == len(set(all_keys)), 'Overlapping trial partitions'
    for partition in buckets:
        if not buckets[partition]:
            continue
        X = np.concatenate(buckets[partition]); y = np.concatenate(labels[partition])
        X = ((X - mean) / std).astype(np.float32)
        assert np.isfinite(X).all()
        np.savez_compressed(output_dir / f'{partition}.npz', X=X, y=y,
                            channels=channels_ref, start=interval_ref[0], end=interval_ref[1], sfreq=160)
        pd.concat(metadata_parts[partition], ignore_index=True).to_csv(
            output_dir / f'{partition}_metadata.csv', index=False)
        print(partition, X.shape, 'class counts', np.bincount(y, minlength=2))
    np.savez_compressed(output_dir / 'train_normalization.npz', mean=mean, std=std)
    print('CNN: X [batch,64,time]; LSTM: X.transpose(0,2,1) [batch,time,64]')
    print('Pilot run split only; for pooled multi-subject modeling consider more execution runs for training.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--subjects', type=int, nargs='+', default=[1,2,3,4,5])
    p.add_argument('--output-dir', type=Path, default=Path('outputs'))
    a = p.parse_args()
    build_split(a.output_dir, a.subjects)
