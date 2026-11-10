"""Reproducible, report-ready visualizations of actual EEG preprocessing outputs.

Run from the repository root:
    python -m src.figures --output-dir outputs_3s --subject 1

Requires previously generated S###_epochs.npz, S###_metadata.csv,
all_metadata.csv. Does not modify data, QC flags, or epoch selection.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE = '#235D86'
ORANGE = '#C67C37'
TEAL = '#297D80'
GRAY = '#66717D'


def setup_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9.5,
        'axes.titlesize': 11,
        'axes.labelsize': 9.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
    })


def save(fig, directory: Path, stem: str):
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(directory / f'{stem}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def load_subject(folder: Path, subject: int):
    with np.load(folder / f'S{subject:03d}_epochs.npz') as z:
        X, y = z['X'], z['y']
        channels = [str(x) for x in z['channels']]
        start, end = float(z['start']), float(z['end'])
    meta = pd.read_csv(folder / f'S{subject:03d}_metadata.csv')
    retained = meta['retained'].astype(bool)
    meta = meta.loc[retained].reset_index(drop=True)
    if not (len(meta) == len(X) == len(y)):
        raise ValueError('Mismatch between retained metadata and epoch arrays')
    if X.ndim != 3 or X.shape[1] != 64:
        raise ValueError(f'Unexpected EEG shape {X.shape}; expected (trials, 64, samples)')
    if not np.all(np.isfinite(X)):
        raise ValueError('Nonfinite EEG values in retained epochs')
    # Ensure sample times correspond to stored start and sample count.
    sampling_hz = X.shape[-1] / (end - start)
    if not np.isclose(sampling_hz, 160, atol=.01):
        raise ValueError(f'Unexpected effective sample rate {sampling_hz:.3f} Hz')
    return X, y, channels, start, end, meta


def channel_index(channels, name):
    candidates = [i for i, ch in enumerate(channels) if ch.strip('. ').casefold() == name.casefold()]
    if len(candidates) != 1:
        raise ValueError(f'Expected unique {name} in EEG channel list; matches: {candidates}')
    return candidates[0]


def choose_epoch(X, y, meta, run, label=0):
    indices = np.flatnonzero((meta.run.to_numpy() == run) & (y == label))
    if not len(indices):
        raise ValueError(f'No retained label {label} epochs in run {run}')
    return X[indices[0]] * 1e6


def plot_example(folder, subject, out):
    X, y, channels, start, end, meta = load_subject(folder, subject)
    epoch = choose_epoch(X,y,meta,3)
    time = start + np.arange(X.shape[-1]) / 160.0
    fig, axes = plt.subplots(3,1,figsize=(7.2,5.2),sharex=True,layout='constrained')
    for ax, name in zip(axes, ('C3','Cz','C4')):
        ax.plot(time,epoch[channel_index(channels,name)],color=BLUE,lw=.95)
        ax.set_ylabel(f'{name} (µV)')
        ax.grid(axis='y',alpha=.16)
        ax.set_xlim(start,end)
    axes[-1].set_xlabel('Time relative to task annotation onset (s)')
    fig.suptitle('Motor-execution EEG: representative left-fist epoch',fontweight='semibold')
    fig.text(.5,.935,f'S{subject:03d}  |  Run 03  |  C3, Cz, C4  |  8–30 Hz bandpass',ha='center',fontsize=8,color=GRAY)
    save(fig,out,'figure_1_motor_execution_epoch')


def plot_counts(folder, subject, out):
    _, _, _, _, _, meta = load_subject(folder, subject)
    runs = [3,7,11,4,8,12]
    left = np.array([int(((meta.run==r)&(meta.label==0)).sum()) for r in runs])
    right = np.array([int(((meta.run==r)&(meta.label==1)).sum()) for r in runs])
    fig,ax=plt.subplots(figsize=(7.2,3.65),layout='constrained')
    x=np.arange(6);w=.35
    for positions,vals,color,label in [(x-w/2,left,BLUE,'Left fist (T1)'),(x+w/2,right,ORANGE,'Right fist (T2)')]:
        bars=ax.bar(positions,vals,width=w,color=color,label=label)
        ax.bar_label(bars,padding=2,fontsize=8)
    ax.axvline(2.5,color='#AEB7BE',lw=.8,ls='--')
    ax.set_xticks(x,[f'R{r:02d}' for r in runs])
    ax.set_ylabel('Retained EEG epochs (n)')
    ax.set_ylim(0,max(9,max(left+right)+1.5))
    ax.set_title('Left–right class balance across motor-task runs',fontweight='semibold')
    ax.legend(loc='upper right',frameon=False,ncol=2,fontsize=8)
    ax.text(1,-.2,'Motor execution',ha='center',transform=ax.get_xaxis_transform(),color=GRAY,fontsize=8)
    ax.text(4,-.2,'Motor imagery',ha='center',transform=ax.get_xaxis_transform(),color=GRAY,fontsize=8)
    fig.text(.01,.01,f'S{subject:03d} | Each bar is the number of retained annotated task epochs; QC flags are not exclusions.',fontsize=7.8,color=GRAY)
    save(fig,out,'figure_2_class_balance_by_run')


def plot_comparison(folder, subject, out):
    X,y,channels,start,end,meta=load_subject(folder,subject)
    ex=choose_epoch(X,y,meta,3);im=choose_epoch(X,y,meta,4)
    time=start+np.arange(X.shape[-1])/160
    fig,axes=plt.subplots(3,2,figsize=(9.0,5.5),sharex=True,layout='constrained')
    for row,name in enumerate(('C3','Cz','C4')):
        idx=channel_index(channels,name)
        arrs=(ex[idx],im[idx]); low=min(a.min() for a in arrs);high=max(a.max() for a in arrs)
        pad=max((high-low)*.08,1)
        for col,arr in enumerate(arrs):
            ax=axes[row,col]
            ax.plot(time,arr,lw=.85,color=(BLUE if col==0 else TEAL))
            ax.set_ylim(low-pad,high+pad)
            ax.set_xlim(start,end)
            ax.set_ylabel(f'{name} (µV)')
            ax.grid(axis='y',alpha=.13)
    axes[0,0].set_title('Execution · R03',fontweight='semibold')
    axes[0,1].set_title('Imagery · R04',fontweight='semibold')
    for ax in axes[-1]: ax.set_xlabel('Time relative to task annotation onset (s)')
    fig.suptitle('Representative EEG epochs during motor execution and imagery',fontweight='semibold')
    fig.text(.5,.945,f'S{subject:03d} | Left-fist condition | Unpaired single trials | 8–30 Hz',ha='center',fontsize=8,color=GRAY)
    save(fig,out,'figure_3_execution_imagery_epochs')


def plot_quality(folder, out):
    meta=pd.read_csv(folder/'all_metadata.csv')
    meta=meta.loc[meta.retained.astype(bool)].copy()
    qc=meta.groupby('subject',sort=True)['qc_flag'].agg(['sum','count'])
    if qc.empty: raise ValueError('No retained EEG metadata')
    clean=100*(qc['count']-qc['sum'])/qc['count'];flagged=100*qc['sum']/qc['count']
    fig,ax=plt.subplots(figsize=(7.4,4.1),layout='constrained')
    x=np.arange(len(qc))
    ax.bar(x,clean,color=TEAL,label='No QC flags')
    ax.bar(x,flagged,bottom=clean,color=ORANGE,label='≥1 QC flag')
    ax.set_xticks(x,[str(s) for s in qc.index])
    ax.set_ylim(0,110)
    ax.set_yticks(np.arange(0,101,20))
    ax.set_ylabel('Proportion of retained EEG epochs (%)')
    ax.set_title('Exploratory EEG quality-control flags across participants',fontweight='semibold')
    for i,(cnt,nflag) in enumerate(zip(qc['count'],qc['sum'])):
        ax.text(i,102,f'n={cnt}',ha='center',fontsize=8,color=GRAY)
        ax.text(i,100-float(nflag)/float(cnt)*50,f'{100*nflag/cnt:.1f}%',ha='center',va='center',fontsize=8,
                color='white' if nflag/cnt>.2 else '#263238',fontweight='semibold')
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.16),frameon=False,ncol=2)
    fig.text(.01,.005,'QC criterion: peak-to-peak <0.5 or >200 µV in ≥1 channel; flags trigger review, not automatic exclusion.',fontsize=7.4,color=GRAY)
    save(fig,out,'figure_4_qc_flag_rates')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path('outputs_3s'))
    parser.add_argument('--figure-dir',type=Path,default=None)
    parser.add_argument('--subject',type=int,default=1)
    args=parser.parse_args();setup_style()
    out=args.figure_dir or (args.output_dir/'figures_publication')
    plot_example(args.output_dir,args.subject,out)
    plot_counts(args.output_dir,args.subject,out)
    plot_comparison(args.output_dir,args.subject,out)
    plot_quality(args.output_dir,out)
    print('Saved four report-ready figures (PNG + vector PDF) to:',out)

if __name__=='__main__':main()
