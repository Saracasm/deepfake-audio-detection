# Environment

This document records the exact environment used for this project.

## Platform

- **Compute:** Google Colab Pro
- **GPU:** Tesla T4, 15.64 GB
- **CUDA version:** 12.1
- **Python version:** 3.12.13
- **OS:** Linux-6.6.113+-x86_64-with-glibc2.35

## Key Library Versions

(See `requirements.txt` for the full pinned list)

- torch: 2.4.0+cu121
- torchaudio: 2.4.0+cu121
- transformers: 4.44.2

## Storage

- Drive folder: `/content/drive/MyDrive/deepfake_audio/`
  - `data/` — raw and preprocessed datasets
  - `checkpoints/` — saved model weights
  - `logs/` — local log files

## Experiment Tracking

- **Wandb project:** `deepfake-audio-detection`
- **Wandb entity:** `sara-jaffrani17-dlp`

## Reproducibility

- Random seed: `42`
- Sample rate: 16 kHz mono throughout
- Window length: 4 seconds with 50% overlap
- Padding for short clips: zero-pad to 4 seconds

## Phase 1 Verification

Phase 1 smoke test: 2026-04-28
All checks passed: GPU detected, Drive mounted, Wav2Vec 2.0 loaded, forward pass succeeded, wandb run logged.
