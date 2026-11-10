"""Read-only, reproducible audit of processed EEGBCI epochs and trial provenance.

Run from repository root:
python -m src.audit_preprocessing --output-dir outputs_3s --compare-dir outputs

Never changes existing EEG arrays or the training/test split.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

PARTITIONS = ('train', 'val', 'execution_test', 'imagery_test')
EXPECTED_RUNS = {3:'train', 7:'val', 11:'execution_test', 4:'imagery_test', 8:'imagery_test', 12:'imagery_test'}


def retained_metadata(folder: Path) -> pd.DataFrame:
    df = pd.read_csv(folder / 'all_metadata.csv')
    if 'retained' not in df:
        raise ValueError('Missing retained column')
    kept = df[df['retained'].astype(bool)].copy()
    kept['trial_key'] = kept['subject'].astype(str) + ':' + kept['run'].astype(str) + ':' + kept['annotation_index'].astype(str)
    if kept.trial_key.duplicated().any():
        raise ValueError('Duplicate trial identifiers')
    return kept


def audit(folder: Path, out: Path, compare_dir: Path | None = None):
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(folder / 'all_metadata.csv')
    retained = retained_metadata(folder)
    if retained.empty:
        raise ValueError('No retained epochs')
    # Each row is a subject/run/condition label group; compute counts without inventing excluded trials.
    counts = (retained.groupby(['subject','condition','run','label'], as_index=False)
              .agg(n_epochs=('trial_key','size'), n_qc_flagged=('qc_flag','sum')))
    counts.to_csv(out/'trial_counts_by_run.csv', index=False)
    reasons = retained.groupby('subject',as_index=False).agg(
        n_epochs=('trial_key','size'), n_qc_flagged=('qc_flag','sum'),
        n_flat=('qc_flat','sum'),n_large_amplitude=('qc_large_amplitude','sum'),
        n_nonfinite=('qc_nonfinite','sum'))
    reasons['pct_qc_flagged'] = 100 * reasons.n_qc_flagged/reasons.n_epochs
    reasons.to_csv(out/'qc_flag_summary.csv',index=False)
    # QC flag causes at channel level: original pipeline flags a trial if ANY channel exceeds thresholds.
    per_channel=[]
    for subject, m in retained.groupby('subject',sort=True):
        subj_no=int(str(subject).lstrip('S'))
        path=folder/f'S{subj_no:03d}_epochs.npz'
        with np.load(path) as z:
            x=z['X']
            ch=z['channels'].astype(str)
            start=float(z['start']); end=float(z['end'])
        if len(m)!=len(x):
            raise ValueError(f'Mismatch for {subject}: {len(m)} metadata rows versus {len(x)} epochs')
        if len(ch)!=64 or x.shape[1]!=64:
            raise ValueError('Unexpected channel dimension')
        if x.shape[2]!=round((end-start)*160):
            raise ValueError('Unexpected sample dimension')
        if not np.isfinite(x).all():
            raise ValueError(f'Nonfinite values in saved {subject} epochs')
        ptp_uv=np.ptp(x,axis=-1)*1e6
        flat=ptp_uv<0.5
        high=ptp_uv>200.0
        flag=np.any(flat|high,axis=1)
        # Check EXACT agreement with metadata QC flags rather than assume.
        if not np.array_equal(flag,m['qc_flag'].astype(bool).to_numpy()):
            raise ValueError(f'QC flags differ from saved metadata for {subject}')
        for i,name in enumerate(ch):
            per_channel.append({'subject':subject,'channel':name,'n_flat_trials':int(flat[:,i].sum()),
                'n_large_amplitude_trials':int(high[:,i].sum()), 'n_evaluated_trials':len(x)})
    channels=pd.DataFrame(per_channel)
    channels.to_csv(out/'qc_flags_by_channel.csv',index=False)

    # Independently confirm leakage-safe split membership, shapes, and normalization.
    split_details=[]; identifiers=[]
    for partition in PARTITIONS:
        file=folder/f'{partition}.npz'
        metadata_file=folder/f'{partition}_metadata.csv'
        if not file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f'Missing {file} or {metadata_file}; run src.split_data first')
        with np.load(file) as z:
            X=z['X']; y=z['y']; start=float(z['start']); end=float(z['end'])
        mm=pd.read_csv(metadata_file)
        if len(X)!=len(y) or len(y)!=len(mm):
            raise ValueError(f'{partition}: data/label/metadata length mismatch')
        if X.shape[1:]!=(64,round((end-start)*160)):
            raise ValueError(f'{partition}: invalid sample dimensions {X.shape}')
        if not np.isfinite(X).all() or not np.isin(y,[0,1]).all():
            raise ValueError(f'{partition}: invalid signals or labels')
        if not np.array_equal(y,mm.label.to_numpy()):
            raise ValueError(f'{partition}: labels mismatch metadata')
        if not mm.run.map(EXPECTED_RUNS).eq(partition).all():
            raise ValueError(f'{partition}: unexpected run in partition')
        keys=(mm.subject.astype(str)+':'+mm.run.astype(str)+':'+mm.annotation_index.astype(str)).tolist()
        identifiers.extend(keys)
        split_details.append({'partition':partition,'trials':len(y),'left':int((y==0).sum()),
                              'right':int((y==1).sum()),'channels':X.shape[1], 'samples':X.shape[2]})
    if len(identifiers)!=len(set(identifiers)):
        raise ValueError('Trial leakage: overlapping trial IDs across partitions')
    if set(identifiers)!=set(retained.trial_key):
        raise ValueError('Split coverage differs from retained trial IDs')
    splits=pd.DataFrame(split_details)
    splits.to_csv(out/'split_integrity.csv',index=False)

    # Data-only comparison of available windows, not a superiority claim.
    if compare_dir is not None:
        older=retained_metadata(compare_dir)
        merged=older[['trial_key','qc_flag','timing_valid']].merge(
            retained[['trial_key','qc_flag','timing_valid']],on='trial_key',how='outer',suffixes=('_old','_primary'),indicator=True)
        merged.to_csv(out/'window_comparison_by_trial.csv',index=False)
        comparison={'n_old':len(older),'n_primary':len(retained),
                    'n_common':int((merged._merge=='both').sum()),
                    'n_only_old':int((merged._merge=='left_only').sum()),
                    'n_only_primary':int((merged._merge=='right_only').sum()),
                    'n_qc_flags_old':int(older.qc_flag.sum()),
                    'n_qc_flags_primary':int(retained.qc_flag.sum())}
        (out/'window_comparison_summary.json').write_text(json.dumps(comparison,indent=2))
    print('AUDIT PASSED: split membership, sample dimensions, labels, finite signals, and QC agreement')
    print(splits.to_string(index=False)); print(reasons.to_string(index=False))
    return counts,reasons,channels,splits


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,default=Path('outputs_3s'))
    p.add_argument('--audit-dir',type=Path,default=Path('results/preprocessing_audit'))
    p.add_argument('--compare-dir',type=Path)
    a=p.parse_args(); audit(a.output_dir,a.audit_dir,a.compare_dir)

if __name__=='__main__': main()
