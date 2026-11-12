# NEURO 240 — audit and report figures add-on

This is an **additive contribution** to `refreshminht/neuro-nudge`, not a replacement for the existing preprocessing pipeline. Copy only `src/audit_preprocessing.py`, `src/report_figures.py`, and `tests/test_analysis_audit.py` into the matching project directories.

## Prerequisites

The existing package's `src.preprocess` and `src.split_data` must have created `outputs_3s/` using `--start 0.5 --end 3.5`, and `outputs/` may contain the older 0–2.5 s pilot. Do not submit raw or normalized EEG arrays to GitHub.

From Windows CMD in the repository root with `.venv` activated:

```cmd
python -m pytest tests/test_pipeline.py tests/test_analysis_audit.py -q
python -m src.preprocess --subjects 1 2 3 4 5 --data-dir data/raw --output-dir outputs_3s --download --start 0.5 --end 3.5
python -m src.split_data --subjects 1 2 3 4 5 --output-dir outputs_3s
python -m src.audit_preprocessing --output-dir outputs_3s --compare-dir outputs
python -m src.report_figures --output-dir outputs_3s
```

Outputs: `results/preprocessing_audit/*.csv`, a window-comparison JSON and `results/report_figures/` PNG/PDF figures. Add `results/preprocessing_audit/` and `results/report_figures/` to `.gitignore` if you do not intend to commit generated results. For review, optionally commit a selected CSV summary or figures after verifying them.

## Interpretation

- The 3-second 0.5–3.5 s epoch is a **provisional primary specification**, not empirically optimal.
- The 2.5 s window is retained as an earlier pilot comparator.
- Flagged epochs are not automatically corrupted or excluded. The 0.5/200 µV PTP limits are exploratory. Figure C shows how many trials triggered each type of flag (categories can overlap); Figure D counts channel–trial flags, not unique trials.
- Splitting into R03 training, R07 validation, R11 execution test and imagery runs R04/R08/R12 tests cross-task transfer within a participant-represented cohort. It is **not** cross-subject generalization. Imagery data must remain out of training and model selection.
- The five-subject pilot has only 75 training epochs. Do not interpret low accuracy from that pilot as evidence that transfer fails; increase training participants before definitive model conclusions.
- Read event-aligned waveform comparisons as illustrative only. **ERD/ERS** needs a pre-task baseline and an appropriate spectral method, and should be deferred rather than inferred from one waveform.

## What to include in a 2-page report

Prefer *Figure A* (class balance; succinct) or *Figure B* (QC screening) with the exact screening method in the caption. Put detailed QC reasons (C) and channels (D) in supplementary materials or repository outputs. Most important is a separate **experimental design diagram** showing inputs, CNN/LSTM, train/validation/test and execution-to-imagery transfer.

## Contribution via GitHub

```cmd
git branch --show-current
git status
git add src/audit_preprocessing.py src/report_figures.py tests/test_analysis_audit.py README-ADDON.md
git diff --cached --stat
git commit -m "Add EEG preprocessing audit and report figures"
git push -u origin feature/eeg-preprocessing-v2
```

Create a pull request targeting `main` via GitHub website; ask teammate to review before merging. Nothing in this package directly pushes or changes the remote repository.
