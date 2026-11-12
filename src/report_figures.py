"""Report figures derived strictly from existing auditable preprocessing outputs.

python -m src.report_figures --output-dir outputs_3s --audit-dir results/preprocessing_audit
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig,folder,stem):
    folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/f'{stem}.png',dpi=300,bbox_inches='tight')
    fig.savefig(folder/f'{stem}.pdf',bbox_inches='tight')
    plt.close(fig)


def create_figures(output_dir:Path,audit_dir:Path,figure_dir:Path):
    counts=pd.read_csv(audit_dir/'trial_counts_by_run.csv')
    qc=pd.read_csv(audit_dir/'qc_flag_summary.csv')
    by_channel=pd.read_csv(audit_dir/'qc_flags_by_channel.csv')
    primary=pd.read_csv(output_dir/'all_metadata.csv')
    # Figure A: class balance across runs: aggregate all five subjects, not selected single participant.
    order=[3,7,11,4,8,12]
    left=[int(counts.loc[(counts.run==r)&(counts.label==0),'n_epochs'].sum()) for r in order]
    right=[int(counts.loc[(counts.run==r)&(counts.label==1),'n_epochs'].sum()) for r in order]
    x=np.arange(6); w=.36
    fig,ax=plt.subplots(figsize=(8,3.8),constrained_layout=True)
    ax.bar(x-w/2,left,width=w,label='Left fist (T1)',color='#386C99')
    ax.bar(x+w/2,right,width=w,label='Right fist (T2)',color='#DB8F3D')
    ax.set_xticks(x,[f'R{r:02d}\n'+('Execution' if r in (3,7,11) else 'Imagery') for r in order])
    ax.set_ylabel('Extracted epochs (n)');ax.set_ylim(0,max(left+right)*1.2)
    ax.set_title('A  |  Left/right class balance across recording runs',loc='left',fontsize=11)
    ax.legend(frameon=False,ncol=2,loc='upper left')
    for b in ax.patches: ax.annotate(str(round(b.get_height())),(b.get_x()+b.get_width()/2,b.get_height()),
                                    xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    save(fig,figure_dir,'A_class_balance_all_subjects')

    # Figure B: quality flags, with truthful title and denominator.
    fig,ax=plt.subplots(figsize=(7.6,3.8),constrained_layout=True)
    xs=np.arange(len(qc))
    clean=100*(qc.n_epochs-qc.n_qc_flagged)/qc.n_epochs
    flagged=100*qc.n_qc_flagged/qc.n_epochs
    ax.bar(xs,clean,color='#438C9B',label='No QC flag')
    ax.bar(xs,flagged,bottom=clean,color='#E2A24B',label='≥1 QC flag')
    ax.set_xticks(xs,qc.subject);ax.set_ylim(0,112)
    ax.set_ylabel('Share of extracted epochs (%)')
    ax.set_title('B  |  Exploratory EEG quality-control screening',loc='left',fontsize=11)
    for i,n in enumerate(qc.n_epochs): ax.text(i,102,f'n={n}',ha='center',fontsize=8)
    ax.legend(frameon=False,ncol=2,loc='upper right')
    ax.spines[['top','right']].set_visible(False)
    save(fig,figure_dir,'B_qc_screening_by_subject')

    # Figure C: QC driver breakdown per subject, not mutually exclusive (not stacked).
    fig,ax=plt.subplots(figsize=(7.6,3.8),constrained_layout=True)
    xs=np.arange(len(qc));w=.35
    for off,col,label,color in [(-w/2,'n_flat','Low PTP (<0.5 µV)','#4879A6'),
                                (w/2,'n_large_amplitude','High PTP (>200 µV)','#BF7B42')]:
        pct=100*qc[col]/qc.n_epochs
        ax.bar(xs+off,pct,width=w,label=label,color=color)
    ax.set_xticks(xs,qc.subject)
    ax.set_ylabel('Share of extracted epochs (%)')
    ax.set_title('C  |  Causes of exploratory QC flags',loc='left',fontsize=11)
    ax.legend(frameon=False,fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    save(fig,figure_dir,'C_qc_flag_reasons')

    # Figure D: frequency of high-amplitude flags by electrode summed across participants.
    chan=(by_channel.groupby('channel',as_index=False)
          .agg(high=('n_large_amplitude_trials','sum'),flat=('n_flat_trials','sum')))
    chan['total']=chan.high+chan.flat
    chan=chan.sort_values('total',ascending=False).head(12).iloc[::-1]
    fig,ax=plt.subplots(figsize=(7.6,4),constrained_layout=True)
    ax.barh(chan.channel,chan.total,color='#4879A6')
    ax.set_xlabel('Flagged epoch–channel instances (n)')
    ax.set_title('D  |  Electrodes most frequently triggering QC flags',loc='left',fontsize=11)
    ax.spines[['top','right']].set_visible(False)
    save(fig,figure_dir,'D_qc_drivers_by_channel')
    print(f'Saved report-ready diagnostic figures A–D to: {figure_dir}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,default=Path('outputs_3s'))
    p.add_argument('--audit-dir',type=Path,default=Path('results/preprocessing_audit'))
    p.add_argument('--figure-dir',type=Path,default=Path('results/report_figures'))
    a=p.parse_args();create_figures(a.output_dir,a.audit_dir,a.figure_dir)

if __name__=='__main__':main()
