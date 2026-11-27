"""NEURO 240: task-period C3/C4 mu and beta power, training + validation only.

Run from repository root:
    python -m src.analyze_c3_c4 --data-dir outputs_3s_20

Does NOT read execution_test.npz or imagery_test.npz.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

BANDS = {'mu': (8., 13.), 'beta': (13., 30.)}
PARTITIONS = ('train', 'val')


def load_split(directory, split):
    """Load original filtered, *unstandardized* epochs for a single run.

    This is essential for physically interpretable C3/C4 power comparisons:
    train.npz/val.npz have channel-specific z-scoring applied.
    """
    target_run = {'train': 3, 'val': 7}[split]
    config = json.loads((directory / 'config.json').read_text())
    subjects = config['subjects']
    arrays, labels, metadata = [], [], []
    channels_ref = None
    for subject_id in subjects:
        name = f'S{int(subject_id):03d}'
        with np.load(directory / f'{name}_epochs.npz', allow_pickle=False) as z:
            x, y = z['X'], z['y']
            channels = [str(c) for c in z['channels']]
            fs = float(z['sfreq'])
        m = pd.read_csv(directory / f'{name}_metadata.csv')
        m = m[m['retained'].astype(bool)].reset_index(drop=True)
        if len(x) != len(y) or len(x) != len(m):
            raise ValueError(f'{name}: metadata/EEG row mismatch')
        if channels_ref is None:
            channels_ref = channels
        elif channels != channels_ref:
            raise ValueError(f'{name}: channel order mismatch')
        idx = np.flatnonzero(m['run'].to_numpy() == target_run)
        arrays.append(x[idx])
        labels.append(y[idx])
        metadata.append(m.iloc[idx].copy())
    x = np.concatenate(arrays).astype(np.float64) * 1e6  # volts -> microvolts
    y = np.concatenate(labels).astype(int)
    meta = pd.concat(metadata, ignore_index=True)
    if not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError(f'{split}: invalid data')
    if not {'C3', 'C4'}.issubset(channels_ref) or fs != 160 or x.shape[1:] != (64, 480):
        raise ValueError(f'{split}: unexpected channels, sampling rate, or shape')
    return x, y, meta, channels_ref, fs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path('outputs_3s_20'))
    p.add_argument('--out-dir', type=Path, default=Path('results/c3_c4_20'))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_trial = []
    spectra = []
    for split in PARTITIONS:
        x, y, meta, ch, fs = load_split(args.data_dir, split)
        print(f'{split}: {x.shape}, C3={ch.index("C3")}, C4={ch.index("C4")}')
        # Input data were bandpass-filtered in src.preprocess and are in microvolts.
        # Do not use split files here: they contain channel-specific standardized data.
        x2 = x[:, [ch.index('C3'), ch.index('C4')], :]
        freqs, psd = welch(x2, fs=fs, nperseg=320, noverlap=160, axis=-1)
        # Trials x channels x frequencies
        for band, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs <= hi)
            if mask.sum() < 2:
                raise ValueError(f'Insufficient Welch frequencies for {band}')
            power = np.trapezoid(psd[:, :, mask], x=freqs[mask], axis=-1)
            # Convert power to dB separately per channel.  Difference is log ratio.
            power_db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny))
            for i in range(len(y)):
                per_trial.append({
                    'partition': split,
                    'subject': str(meta.iloc[i]['subject']),
                    'run': int(meta.iloc[i]['run']),
                    'label': int(y[i]),
                    'hand': 'left' if y[i] == 0 else 'right',
                    'band': band,
                    'C3_db': float(power_db[i, 0]),
                    'C4_db': float(power_db[i, 1]),
                    'C4_minus_C3_db': float(power_db[i, 1] - power_db[i, 0]),
                    'qc_flag': bool(meta.iloc[i]['qc_flag'])
                })
        for i in range(len(y)):
            spectra.append((split, str(meta.iloc[i]['subject']), int(y[i]), psd[i]))

    trials = pd.DataFrame(per_trial)
    trials.to_csv(args.out_dir / 'trial_band_power.csv', index=False)
    # Subject-level averages first, then group averages: participants weighted equally.
    subject = trials.groupby(['partition', 'subject', 'hand', 'band'], as_index=False)[
        ['C3_db', 'C4_db', 'C4_minus_C3_db']].mean()
    subject.to_csv(args.out_dir / 'subject_band_power.csv', index=False)

    # Difference-in-differences: (C4-C3) in left minus (C4-C3) in right,
    # i.e. hand-dependent hemispheric asymmetry (not baseline-referenced ERD).
    wide = subject.pivot(index=['partition', 'subject', 'band'], columns='hand',
                         values='C4_minus_C3_db').dropna()
    wide['left_minus_right_lateralization_db'] = wide['left'] - wide['right']
    contrast = wide.reset_index()
    contrast.to_csv(args.out_dir / 'subject_lateralization_contrast.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, band in zip(axes, BANDS):
        subset = contrast[(contrast['partition'] == 'val') & (contrast['band'] == band)]
        values = subset['left_minus_right_lateralization_db'].to_numpy()
        ax.axhline(0, linestyle='--', linewidth=1)
        ax.scatter(np.arange(len(values)), values, s=24)
        ax.set_title(f'Validation: {band} ({BANDS[band][0]:g}–{BANDS[band][1]:g} Hz)')
        ax.set_xlabel('Participant index')
        ax.set_ylabel('Left − right C4−C3 contrast (dB)')
        ax.text(.02, .96, f'N={len(values)}; mean={np.mean(values):.2f} dB',
                transform=ax.transAxes, va='top')
    fig.tight_layout()
    fig.savefig(args.out_dir / 'validation_subject_lateralization.png', dpi=180)
    plt.close(fig)

    # Equal participant weighting of PSD curves.
    for split in PARTITIONS:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ci, channel in enumerate(('C3', 'C4')):
            ax = axes[ci]
            for label, name in [(0, 'Left fist'), (1, 'Right fist')]:
                rows = [(sub, arr[ci]) for sp, sub, lab, arr in spectra
                        if sp == split and lab == label]
                psd_by_subject = {}
                for sub, arr in rows:
                    psd_by_subject.setdefault(sub, []).append(arr)
                subject_means = np.stack([np.mean(v, axis=0) for v in psd_by_subject.values()])
                group_mean = subject_means.mean(axis=0)
                ax.plot(freqs, 10 * np.log10(np.maximum(group_mean, np.finfo(float).tiny)),
                        label=name)
            ax.set_xlim(8, 30)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_title(f'{split}: {channel}')
            ax.grid(alpha=.2)
        axes[0].set_ylabel('PSD (dB, µV²/Hz)')
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / f'{split}_c3_c4_psd.png', dpi=180)
        plt.close(fig)

    print('\nValidation subject-level lateralization contrasts (dB):')
    print(contrast[contrast.partition == 'val'].groupby('band')[
        'left_minus_right_lateralization_db'].agg(['count', 'mean', 'std']).round(3))
    print(f'\nSaved CSVs and PNGs to: {args.out_dir}')
    print('Interpret as task-period power/asymmetry, NOT baseline-relative ERD.')
    print('PSD is computed from original preprocessed, unstandardized epochs in µV²/Hz.')


if __name__ == '__main__':
    main()
