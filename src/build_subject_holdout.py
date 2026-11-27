"""Exploratory subject-held-out, execution-only NEURO 240 experiment.

Run from neuro-nudge repository root:
  python -m src.build_subject_holdout --source-dir outputs_3s_20 --output-dir outputs_3s_subject_holdout_20

Train: S001-S016, execution runs R03 and R07.
Validation: S017-S020, execution runs R03 and R07.
No test or imagery files are opened, created, or evaluated.

IMPORTANT: Participants S017-S020/R07 were used previously for model
validation in the run-held-out pilot. Therefore this is an EXPLORATORY
follow-up, not a pristine, never-inspected validation cohort.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_IDS = tuple(range(1, 17))
VAL_IDS = tuple(range(17, 21))
EXECUTION_RUNS = (3, 7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('outputs_3s_20'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs_3s_subject_holdout_20'))
    args = parser.parse_args()

    src = args.source_dir.resolve()
    out = args.output_dir.resolve()
    if src == out:
        raise ValueError('Output directory must differ from source directory!')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty. Use a fresh output folder to avoid overwriting files.')

    result = {'train': {'X': [], 'y': [], 'meta': []}, 'val': {'X': [], 'y': [], 'meta': []}}
    expected_channels = None
    expected_time = None
    provenance = set()

    for partition, ids in [('train', TRAIN_IDS), ('val', VAL_IDS)]:
        for subject in ids:
            subject_code = f'S{subject:03d}'
            with np.load(src / f'{subject_code}_epochs.npz', allow_pickle=False) as f:
                x = f['X'].astype(np.float32)
                y = f['y'].astype(np.int64)
                channels = f['channels']
                start, end = float(f['start']), float(f['end'])
                sfreq = float(f['sfreq'])
            meta = pd.read_csv(src / f'{subject_code}_metadata.csv')
            meta = meta.loc[meta['retained'].astype(bool)].reset_index(drop=True)
            if x.shape[1:] != (64, 480) or len(x) != len(y) or len(y) != len(meta):
                raise ValueError(f'{subject_code}: unexpected shapes or metadata row count')
            if not (np.isclose(start, .5) and np.isclose(end, 3.5) and np.isclose(sfreq, 160)):
                raise ValueError(f'{subject_code}: unexpected preprocessing parameters')
            if not np.array_equal(meta['label'].to_numpy(dtype=np.int64), y):
                raise ValueError(f'{subject_code}: metadata/label mismatch')
            if not (meta['subject'].astype(str) == subject_code).all():
                raise ValueError(f'{subject_code}: subject mismatch')
            if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
                raise ValueError(f'{subject_code}: invalid signals or labels')
            if expected_channels is None:
                expected_channels = channels.copy()
                expected_time = (start, end, sfreq)
            elif not np.array_equal(channels, expected_channels) or (start, end, sfreq) != expected_time:
                raise ValueError(f'{subject_code}: channel order/window mismatch')

            idx = np.flatnonzero(meta['run'].isin(EXECUTION_RUNS).to_numpy())
            selected_meta = meta.iloc[idx].copy()
            if set(selected_meta['run'].unique()) != set(EXECUTION_RUNS):
                raise ValueError(f'{subject_code}: both execution R03 and R07 required')
            if not (selected_meta['condition'].astype(str) == 'execution').all():
                raise ValueError(f'{subject_code}: unexpected condition')
            for row in selected_meta.itertuples():
                key = (row.subject, int(row.run), int(row.annotation_index))
                if key in provenance:
                    raise ValueError(f'Duplicate epoch: {key}')
                provenance.add(key)

            result[partition]['X'].append(x[idx])
            result[partition]['y'].append(y[idx])
            result[partition]['meta'].append(selected_meta)

    train = np.concatenate(result['train']['X'])
    mean = train.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = train.std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid training-only normalization statistics')

    # Prepare all arrays before writing anything.
    prepared = {}
    for partition, ids in [('train', TRAIN_IDS), ('val', VAL_IDS)]:
        x = np.concatenate(result[partition]['X'])
        y = np.concatenate(result[partition]['y'])
        meta = pd.concat(result[partition]['meta'], ignore_index=True)
        if set(meta['subject'].unique()) != {f'S{i:03d}' for i in ids}:
            raise ValueError(f'{partition}: incorrect subject assignment')
        if not set(meta['run'].unique()) == {3, 7}:
            raise ValueError(f'{partition}: incorrect runs')
        x = ((x - mean) / std).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f'{partition}: nonfinite normalized signals')
        prepared[partition] = (x, y, meta)

    if set(prepared['train'][2]['subject']) & set(prepared['val'][2]['subject']):
        raise ValueError('Subject overlap between train and validation')

    out.mkdir(parents=True, exist_ok=False)
    for partition, (x, y, meta) in prepared.items():
        np.savez_compressed(out / f'{partition}.npz', X=x, y=y,
                            channels=expected_channels, start=.5, end=3.5, sfreq=160)
        meta.to_csv(out / f'{partition}_metadata.csv', index=False)
        print(f'{partition}: {x.shape} | class counts {np.bincount(y, minlength=2).tolist()} '
              f'| participants {meta.subject.nunique()} | runs {sorted(meta.run.unique().tolist())}')

    np.savez_compressed(out / 'train_normalization.npz', mean=mean, std=std)
    manifest = {
        'status': 'exploratory_follow_up_not_pristine_holdout',
        'source': str(src), 'window_seconds': [0.5, 3.5],
        'bandpass_hz': [8, 30], 'sfreq_hz': 160,
        'train_subjects': [f'S{i:03d}' for i in TRAIN_IDS],
        'validation_subjects': [f'S{i:03d}' for i in VAL_IDS],
        'train_runs': [3, 7], 'validation_runs': [3, 7],
        'normalization': 'per-channel mean and std from TRAIN only',
        'test_data_policy': 'No R11 or imagery data were read; no test splits were generated',
        'caveat': 'R07 was previously explored for all 20 subjects, including current validation subjects.',
    }
    (out / 'split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nSaved exploratory training/validation datasets to: {out}')
    print('NO held-out test or imagery arrays were loaded or generated.')


if __name__ == '__main__':
    main()
