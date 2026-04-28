# Environment

This document records the exact environment used for this project. After running the smoke-test notebook, fill in the values below from the printed output.

## Platform

- **Compute:** Google Colab Pro
- **GPU:** _(fill from smoke test, e.g. "Tesla T4, 15 GB")_
- **CUDA version:** _(fill from smoke test)_
- **Python version:** _(fill from smoke test, e.g. "3.11.x")_

## Key Library Versions

(See `requirements.txt` for the full pinned list)

- torch: 2.4.0
- torchaudio: 2.4.0
- transformers: 4.44.2

## Storage

- Drive folder: `/content/drive/MyDrive/deepfake_audio/`
  - `data/` — raw and preprocessed datasets
  - `checkpoints/` — saved model weights
  - `logs/` — local log files

## Experiment Tracking

- **wandb project:** `deepfake-audio-detection`
- **wandb entity:** _(your wandb username)_

## Reproducibility Notes

- Random seed: `42` (set in `src/utils/seed.py`)
- Sample rate: 16 kHz mono throughout
- Window length: 4 seconds with 50% overlap
- Padding for short clips: zero-pad to 4 seconds

## Installation Order Notes

If reinstalling on a fresh Colab session:

1. Mount Drive first
2. Clone repo into `/content/`
3. `pip install -r requirements.txt --quiet`
4. Restart runtime ONCE after install (Runtime → Restart session)
5. Run smoke test to verify
