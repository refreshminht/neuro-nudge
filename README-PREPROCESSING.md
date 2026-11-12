# NEURO 240 — EEGBCI preprocessing contribution

A standalone, reviewable preprocessing pipeline for the PhysioNet EEG Motor Movement/Imagery dataset. It does **not** overwrite the team's existing README or scripts.

## Important scientific decisions
- Binary task: T1=left fist, T2=right fist **only for runs 3, 4, 7, 8, 11, 12**.
- Runs 3/7/11: execution; 4/8/12: imagery.
- **Default** window: 0–2.5 s (400 samples), matching the five-subject pilot described in TA notes.
- **Alternative** window: 0.5–3.5 s (480 samples), matching the proposed model input; use only after comparing annotation duration and window validity. Never mix model shapes.
- Task annotation onset is time zero, **not** measured movement onset.
- 8–30 Hz bandpass is a configurable research choice for sensorimotor rhythms; it can suppress other information, so report it and consider alternatives later.
- QC flags are exploratory (flat channel <0.5 µV p-p; large amplitude >200 µV p-p). Nonfinite epochs are excluded; other flagged epochs are retained for review.
- Run-based split: R03 train, R07 validation, R11 execution test, R04/R08/R12 imagery test. This is a *small pilot split*, not the final recommended pooled-subject evaluation protocol.
- Training-only channel-wise normalization; never fit normalization on imagery test data.

## Setup (VS Code terminal, from repository root)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-preprocessing.txt
```
On macOS/Linux use `source .venv/bin/activate`.

## Run
To download EDFs with MNE (requires internet):
```powershell
python -m src.preprocess --subjects 1 2 3 4 5 --data-dir data/raw --output-dir outputs --download --start 0 --end 2.5
```
**Note:** MNE's download directory layout may differ from the manual `data/raw/S001/S001R03.edf` layout. Use `--download` consistently when downloading through MNE. If using manual EDFs, omit `--download` and place each file at `data/raw/S001/S001R03.edf` etc.

For the alternative 3-second window, use `--start 0.5 --end 3.5` and a different output directory, e.g. `outputs_3s` (and add that directory to `.gitignore`).

```powershell
python -m src.split_data --subjects 1 2 3 4 5 --output-dir outputs
python -m src.figures --output-dir outputs --subject 1
python -m pytest -q
```

## Outputs
- `S###_epochs.npz`: `X` (n,64,400 by default), `y` (n,)
- `S###_metadata.csv`: annotation, window validity, QC flags, retained decision
- `all_metadata.csv`: per-trial audit trail
- `train.npz`, `val.npz`, `execution_test.npz`, `imagery_test.npz`: normalized common model inputs
- `train_normalization.npz`: training-only channel means and standard deviations
- `figure_1_sample_epoch.png`: filtered C3/Cz/C4 example, signal extraction check
- `figure_2_trial_counts.png`: left/right counts across runs, class balance
- `figure_3_execution_vs_imagery.png`: single-trial qualitative comparison; **not evidence of condition effects**
- `figure_4_qc_summary.png`: QC flags by subject; **not artifact rejection rates**

## Collaboration
Create a feature branch, commit only `src/`, `tests/`, `requirements-preprocessing.txt`, `README-PREPROCESSING.md`, and `.gitignore` additions. Do not commit EDF files, large arrays, or generated figures unless the team agrees to version a small selected figure. Open a pull request for teammate review.
