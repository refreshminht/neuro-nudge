"""Reproducible PhysioNet EEGBCI binary left/right preprocessing.

Annotations: T1=left fist, T2=right fist ONLY in runs 3/4/7/8/11/12.
Time zero is the task annotation onset (not measured movement onset).
"""
import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci

RUNS = (3, 4, 7, 8, 11, 12)
EXECUTION = {3, 7, 11}
LABELS = {'T1': 0, 'T2': 1}


def extract_subject(subject, root, start=0.0, end=2.5, l_freq=8.0, h_freq=30.0, download=False):
    """Return EEG array (trials, channels, samples), labels, and trial metadata."""
    if not 0 <= start < end:
        raise ValueError('Require 0 <= start < end')
    arrays, labels, records = [], [], []
    target_channels = None
    expected_samples = round((end - start) * 160)
    if abs(expected_samples - (end - start) * 160) > 1e-5:
        raise ValueError('Window duration must be an integer number of 160 Hz samples')
    for run in RUNS:
        paths = eegbci.load_data(subject, [run], path=str(root), update_path=False,
                                  force_update=False, verbose=False) if download else [str(Path(root) / f'S{subject:03d}' / f'S{subject:03d}R{run:02d}.edf')]
        for filename in paths:
            if not Path(filename).exists():
                raise FileNotFoundError(f'Missing {filename}; run with --download or supply EDFs')
            raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
            eegbci.standardize(raw)
            raw.pick_types(eeg=True)
            if abs(raw.info['sfreq'] - 160) > 0.01:
                raise ValueError(f'Unexpected sampling rate in {filename}: {raw.info["sfreq"]}')
            channels = list(raw.ch_names)
            if len(channels) != 64:
                raise ValueError(f'Expected 64 EEG channels in {filename}, got {len(channels)}')
            if target_channels is None:
                target_channels = channels
            elif set(channels) != set(target_channels):
                raise ValueError(f'Channel set mismatch in {filename}')
            raw.reorder_channels(target_channels)
            # Filter continuous recording before epoch extraction to limit boundary effects.
            raw.filter(l_freq, h_freq, verbose=False)
            data = raw.get_data().astype(np.float32)
            sfreq = raw.info['sfreq']
            for annotation_index, ann in enumerate(raw.annotations):
                if ann['description'] not in LABELS:
                    continue
                onset = float(ann['onset'])
                duration = float(ann['duration'])
                sample_start = round((onset + start) * sfreq)
                sample_end = sample_start + expected_samples
                fits_annotation = end <= duration + 1e-6
                fits_recording = 0 <= sample_start and sample_end <= data.shape[1]
                record = dict(subject=f'S{subject:03d}', run=run,
                              condition='execution' if run in EXECUTION else 'imagery',
                              annotation_index=annotation_index, event=ann['description'],
                              label=LABELS[ann['description']], onset_s=onset,
                              task_duration_s=duration, timing_valid=bool(fits_annotation and fits_recording),
                              retained=False, reason='', qc_nonfinite=False,
                              qc_flat=False, qc_large_amplitude=False,
                              qc_flag=False, peak_to_peak_uv=np.nan)
                if not record['timing_valid']:
                    record['reason'] = 'window_outside_annotation_or_recording'
                    records.append(record)
                    continue
                epoch = data[:, sample_start:sample_end]
                if epoch.shape != (64, expected_samples):
                    raise AssertionError('Unexpected epoch shape')
                # Conservative QC: no auto-exclusion for high amplitude or flatness.
                record['qc_nonfinite'] = bool(not np.isfinite(epoch).all())
                if record['qc_nonfinite']:
                    record['reason'] = 'nonfinite_samples'
                    records.append(record)
                    continue
                ptp_uv = np.ptp(epoch, axis=1) * 1e6
                record['peak_to_peak_uv'] = float(ptp_uv.max())
                record['qc_flat'] = bool((ptp_uv < 0.5).any())
                record['qc_large_amplitude'] = bool((ptp_uv > 200).any())
                record['qc_flag'] = record['qc_flat'] or record['qc_large_amplitude']
                record['retained'] = True
                record['reason'] = 'review' if record['qc_flag'] else 'ok'
                records.append(record)
                arrays.append(epoch)
                labels.append(record['label'])
    if not arrays:
        raise RuntimeError(f'No valid epochs for S{subject:03d}')
    X = np.stack(arrays).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    meta = pd.DataFrame(records)
    assert len(X) == len(y) == int(meta.retained.sum())
    return X, y, meta, target_channels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subjects', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs'))
    p.add_argument('--start', type=float, default=0.0)
    p.add_argument('--end', type=float, default=2.5)
    p.add_argument('--download', action='store_true', help='Download via MNE (internet needed)')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_meta = []
    for subject in args.subjects:
        X, y, meta, channels = extract_subject(subject, args.data_dir, args.start, args.end,
                                               download=args.download)
        np.savez_compressed(args.output_dir / f'S{subject:03d}_epochs.npz', X=X, y=y,
                            channels=np.asarray(channels), sfreq=np.float32(160),
                            start=np.float32(args.start), end=np.float32(args.end))
        # Each NPZ row corresponds to metadata rows where retained==True, in order.
        meta.to_csv(args.output_dir / f'S{subject:03d}_metadata.csv', index=False)
        all_meta.append(meta)
        print(f'S{subject:03d}: {len(y)} retained, {len(meta)-len(y)} invalid windows/nonfinite')
    pd.concat(all_meta, ignore_index=True).to_csv(args.output_dir / 'all_metadata.csv', index=False)
    (args.output_dir / 'config.json').write_text(json.dumps(dict(subjects=args.subjects,
        start=args.start, end=args.end, bandpass_hz=[8,30], sfreq=160,
        qc_flat_ptp_uv=0.5, qc_large_ptp_uv=200, qc_policy='review_only'), indent=2))

if __name__ == '__main__':
    main()
