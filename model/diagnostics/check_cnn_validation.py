"""Read-only NEURO 240 CNN diagnostic on held-out R07 validation data only.

Run from neuro-nudge repository root:
    python -m model.diagnostics.check_cnn_validation --data-dir outputs_3s_20 --checkpoint results/cnn_20/best_model.pt

Does not access execution_test.npz or imagery_test.npz, and does not retrain.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from model.temporal_cnn import TemporalCNN


def load_state(checkpoint_path):
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt):
        if any(k.startswith('conv') or k.startswith('features') for k in ckpt):
            return ckpt
        for key in ('model_state_dict', 'state_dict', 'model_state', 'model', 'net'):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # Generic plain state dict: all values tensor-like
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError(f'Unrecognized checkpoint format: {type(ckpt)}. Check available keys before proceeding.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path('outputs_3s_20'))
    p.add_argument('--checkpoint', type=Path, default=Path('results/cnn_20/best_model.pt'))
    p.add_argument('--results-dir', type=Path, default=Path('results/cnn_20/validation_diagnostics'))
    a = p.parse_args()

    with np.load(a.data_dir/'val.npz', allow_pickle=False) as d:
        X = d['X'].astype(np.float32)
        y = d['y'].astype(np.int64)
    meta = pd.read_csv(a.data_dir/'val_metadata.csv')
    assert X.shape == (300,64,480), f'Unexpected validation shape {X.shape}'
    assert len(y) == len(meta) and np.array_equal(y,meta['label'].to_numpy(dtype=int))
    assert (meta['run'].to_numpy() == 7).all(), 'This diagnostic only permits R07 validation'

    model = TemporalCNN()
    model.load_state_dict(load_state(a.checkpoint), strict=True)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for i in range(0,len(X),32):
            output = model(torch.from_numpy(X[i:i+32]))
            probabilities.append(torch.softmax(output,dim=1).cpu().numpy())
    proba = np.vstack(probabilities)
    predicted = proba.argmax(axis=1)
    cm = confusion_matrix(y,predicted,labels=[0,1])
    acc = accuracy_score(y,predicted)
    bal = balanced_accuracy_score(y,predicted)
    counts = np.bincount(predicted,minlength=2).tolist()
    print('CNN selected checkpoint — R07 validation ONLY')
    print(f'Accuracy: {acc:.4f} | Balanced accuracy: {bal:.4f}')
    print(f'Predicted [left, right]: {counts}')
    print('Confusion matrix (rows=true, cols=pred):\n', cm)

    out = a.results_dir
    out.mkdir(parents=True,exist_ok=True)
    df = meta[['subject','run','label']].copy().rename(columns={'label':'true_label'})
    df['predicted_label'] = predicted
    df['probability_right'] = proba[:,1]
    df['correct'] = predicted == y
    df.to_csv(out/'validation_predictions.csv',index=False)
    rows=[]
    for subject,grp in df.groupby('subject',sort=True):
        rows.append({'subject':subject,'n_trials':len(grp),'accuracy':accuracy_score(grp.true_label,grp.predicted_label),'balanced_accuracy':balanced_accuracy_score(grp.true_label,grp.predicted_label)})
    pd.DataFrame(rows).to_csv(out/'validation_by_subject.csv',index=False)
    (out/'validation_metrics.json').write_text(json.dumps({'accuracy':acc,'balanced_accuracy':bal,'predicted_counts':counts,'confusion_matrix':cm.tolist()},indent=2))
    fig,ax = plt.subplots(figsize=(5,4.5))
    im=ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=16,color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set_xticks([0,1],['Left fist','Right fist'])
    ax.set_yticks([0,1],['Left fist','Right fist'])
    ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
    ax.set_title('Temporal CNN: R07 validation (selected checkpoint)')
    fig.colorbar(im,ax=ax,label='Trials')
    fig.tight_layout();fig.savefig(out/'validation_confusion_matrix.png',dpi=200);plt.close(fig)
    print('Saved validation diagnostics to:',out)
    print('No test or imagery data accessed.')

if __name__=='__main__':
    main()
