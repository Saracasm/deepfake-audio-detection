# Deepfake Audio Detection

> AI Voice Cloning Detection via Fine-Tuned Self-Supervised Speech Transformers

A deep learning system that classifies whether a given audio clip contains a real human voice or an AI-synthesized voice. Built by fine-tuning Wav2Vec 2.0 on the ASVspoof 2019 LA benchmark, with cross-dataset evaluation on ASVspoof 2021 LA and WaveFake.

**Status:** 🚧 Work in progress (Phase 1: environment setup)

## Authors

- Areeba Arif (23K-0618)
- Sara Iqbal (23K-0669)

## Overview

Voice cloning technology has advanced to the point where AI systems can mimic a real person's voice from just seconds of reference audio. This has enabled a new class of fraud — voice cloning scams, fabricated audio evidence, and non-consensual voice content. This project builds a binary classifier to detect AI-generated voices.

### Approach

- **Base model:** Wav2Vec 2.0 (Base), pretrained by Meta AI on ~960 hours of unlabeled speech
- **Training data:** ASVspoof 2019 LA (Logical Access — TTS and Voice Conversion attacks)
- **Strategy:** Two-stage transfer learning
  - Stage 1: Frozen backbone + trainable classification head (baseline)
  - Stage 2: Top transformer layers unfrozen for task-specific adaptation
- **Evaluation:** Cross-dataset generalization study across three benchmarks

## Datasets

| Dataset | Role | Description |
|---|---|---|
| ASVspoof 2019 LA | Primary (train + eval) | Standard anti-spoofing benchmark with TTS/VC attacks |
| ASVspoof 2021 LA | Secondary (eval only) | Channel-degraded version of 2019 (telephone codecs) |
| WaveFake | Supplementary (eval only) | Modern neural-vocoder-generated audio (HiFi-GAN, MelGAN, etc.) |

## Project Structure

```
.
├── api/              # FastAPI inference service
├── app/              # Gradio demo UI
├── checkpoints/      # Trained model weights (gitignored)
├── config/           # Hyperparameter configs
├── data/             # Datasets (gitignored)
├── docs/             # Reports and protocol documentation
├── notebooks/        # Colab notebooks for each phase
├── results/          # Evaluation metrics and figures
├── src/              # Reusable Python modules
└── tests/            # Unit tests
```

## Results (To Be Filled)

| Dataset | EER | t-DCF | AUC-ROC |
|---|---|---|---|
| ASVspoof 2019 LA (eval) | TBD | TBD | TBD |
| ASVspoof 2021 LA (eval) | TBD | TBD | TBD |
| WaveFake (sampled) | TBD | — | TBD |

## Demo

🚧 Live demo coming in Phase 6. Will be hosted on Hugging Face Spaces.

## Running Locally

Setup instructions and reproduction guide will be added as the project progresses.

## License

MIT

## References

1. Baevski et al. (2020), *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*
2. Wang et al. (2020), *ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech*
3. Frank & Schönherr (2021), *WaveFake: A Data Set to Facilitate Audio Deepfake Detection*
