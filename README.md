# Neuro240 Final Project: EEG Motor Execution-to-Imagery Generalization

Final project for Neuro240: Biological and Artificial Intelligence at Harvard, Fall 2026.

## Project Overview

Our project investigates whether neural networks trained on EEG recorded during actual left- and right-hand movements can also classify imagined left- and right-hand movements without additional training.

We will compare CNN and LSTM models using the same data splits and evaluation metrics.

**Research question:** Can neural networks trained on actual motor execution generalize to motor imagery without additional training, and how does this generalization differ between CNN and LSTM architectures?

## Dataset

We use the PhysioNet EEG Motor Movement/Imagery Dataset: https://physionet.org/content/eegmmidb/1.0.0/

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
