---
title: Deepfake Audio Detection
emoji: 🎤
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "5.50.0"
app_file: app/app.py
pinned: false
license: mit
short_description: Detect AI-generated speech using Wav2Vec 2.0
---

# Deepfake Audio Detection — Wav2Vec 2.0 Fine-tuning

Detection of synthetic (deepfake) speech using a fine-tuned Wav2Vec 2.0 model. Trained on ASVspoof 2019 LA, evaluated cross-dataset on ASVspoof 2019 LA eval, ASVspoof 2021 LA, and WaveFake.

## Headline results

| Evaluation | EER | What it measures |
|---|---|---|
| ASVspoof 2019 LA dev (seen attacks A01-A06) | **0.69%** | In-distribution memorization check |
| ASVspoof 2019 LA eval (unseen attacks A07-A19) | **5.55%** | Generalization to new attack types |
| ASVspoof 2021 LA eval (unseen attacks + codec degradation) | **9.09%** | Real-world transmission conditions |
| WaveFake (LJSpeech vocoders, mean) | **29.4%** | Out-of-distribution vocoder synthesis |

### Comparison to published baselines

| System | 2019 LA EER | 2021 LA EER |
|---|---|---|
| Official LFCC-GMM baseline | 8.09% | 25.56% |
| Official CQCC-GMM baseline | 9.57% | 19.30% |
| Official LFCC-LCNN baseline | - | 9.26% |
| Official RawNet2 baseline | - | 9.50% |
| **This work (Wav2Vec 2.0 + fine-tuning)** | **5.55%** | **9.09%** |

Our model outperforms the LFCC-GMM 2019 baseline by 2.54 percentage points and matches the strongest neural baselines on 2021 LA, despite no codec-specific augmentation during training.

## Architecture

Pipeline:

1. Raw waveform input (16 kHz, 4 sec, 64,000 samples)
2. Wav2Vec 2.0 Base backbone (95M params, 12 transformer layers)
   - Stage 1: fully frozen
   - Stage 2: top 2 layers + final LayerNorm unfrozen (~14M trainable)
3. Mean pooling over time dimension
4. Linear classification head (768 -> 2)
5. Softmax -> P(spoof), P(bonafide)

### Two-stage training rationale

- **Stage 1**: train only the linear head on top of frozen pretrained features. Establishes that pretrained Wav2Vec representations already carry strong anti-spoofing signal. Result: 10.09% dev EER with 1,538 trainable params.
- **Stage 2**: unfreeze the top 2 transformer layers, lower learning rate from 1e-3 to 1e-5 (warmup + linear decay), enable mixed precision. Result: 0.69% dev EER, 14.18M trainable params.

## Quickstart

### Inference on a single file

    from src.inference.predict import DeepfakeDetector
    
    detector = DeepfakeDetector(checkpoint_path="path/to/stage2_best.pt")
    result = detector.predict("path/to/audio.wav")
    print(result)
    # {
    #   "spoof_probability": 0.84,
    #   "prediction": "spoof",
    #   "confidence": 0.84,
    #   "utterance_duration_sec": 3.42,
    #   "n_windows": 1,
    #   "threshold_used": 0.5
    # }

The detector handles any audio format readable by torchaudio. Audio is automatically resampled to 16 kHz and segmented into 4-second windows; per-window scores are mean-aggregated.

## Repository structure

    .
    ├── src/
    │   ├── data/
    │   │   ├── protocols.py             # ASVspoof 2019 LA protocol parser
    │   │   ├── protocols_2021.py        # ASVspoof 2021 LA protocol parser
    │   │   ├── preprocessing.py         # audio loading, resampling, windowing
    │   │   └── dataset.py               # PyTorch Dataset
    │   ├── models/
    │   │   └── wav2vec_classifier.py    # Wav2Vec backbone + head
    │   ├── training/
    │   │   └── trainer.py               # training loop with mixed precision + LR scheduler
    │   ├── evaluation/
    │   │   └── metrics.py               # EER, AUC, window-to-utterance aggregation
    │   └── inference/
    │       └── predict.py               # production inference wrapper
    ├── notebooks/
    │   ├── 01_data_acquisition.ipynb    # Phase 2: data exploration + pipeline
    │   ├── 02_train_stage2.ipynb        # Phase 3 + 4: training (Stage 1 + Stage 2)
    │   └── 03_evaluation.ipynb          # Phase 5: cross-dataset evaluation
    ├── results/
    │   ├── metrics/                     # JSON results for each phase
    │   │   ├── stage1_results.json
    │   │   ├── stage2_results.json
    │   │   ├── stage2_eval2019_results.json
    │   │   ├── stage2_eval2021_results.json
    │   │   └── stage2_eval_wavefake_results.json
    │   └── scores/                      # raw per-utterance inference scores (.npz)
    └── docs/
        └── environment.md               # verified runtime environment

## Datasets

This project uses three external datasets, none of which are committed to this repository:

- **ASVspoof 2019 LA** ([paper](https://arxiv.org/abs/1911.01601)) - training and primary eval. Available via Kaggle: `anishsarkar22/asvpoof-2019-dataset-la`.
- **ASVspoof 2021 LA** ([paper](https://arxiv.org/abs/2109.00537)) - secondary eval, real-world codec conditions. Available via Kaggle: `ajaysuryal/asvspoof2021-la` plus key file `simontrann/asvspoof2021-la-key`.
- **WaveFake** ([paper](https://arxiv.org/abs/2111.02813)) - supplementary eval, neural vocoder synthesis. Available via Kaggle: `walimuhammadahmad/fakeaudio` plus LJSpeech `mathurinache/the-lj-speech-dataset`.

## Experiment tracking

All training runs logged to Weights & Biases:
- Project: https://wandb.ai/sara-jaffrani17-dlp/deepfake-audio-detection
- Stage 2 final run: see `stage2-full` (5,320 training steps, 10 epochs)

## Key findings

1. **Pretrained Wav2Vec features carry significant anti-spoofing signal.** A frozen-backbone linear classifier achieves 10.09% dev EER on ASVspoof 2019 LA - competitive with hand-crafted feature baselines.
2. **Two-stage fine-tuning is highly effective.** Unfreezing the top 2 transformer layers (15% of model params) drops dev EER from 10.09% to 0.69% - a 93% relative error reduction.
3. **Generalization profile maps cleanly to distribution shift type:**
   - Unseen attacks (same dataset): +4.86 pp degradation
   - Real-world codec degradation: +3.54 pp additional degradation
   - Novel vocoder pipelines (different domain): +17.24 pp additional degradation
4. **Per-codec analysis identifies model weaknesses.** Aggressive lossy compression (GSM, PSTN) degrades performance ~6 pp vs uncompressed audio. Modern codecs (Opus, G.722) preserve detection signal well.
5. **WaveFake reveals an ASVspoof-specific overfitting pattern.** The model has learned ASVspoof-style attack artifacts but not standalone neural vocoder artifacts. This matches findings in the original WaveFake paper.

## Hardware

Trained on Google Colab Pro:
- Stage 1: T4 GPU, 4h 8m wall-clock
- Stage 2: T4 GPU with mixed precision, 2h 56m wall-clock
- All evaluations: T4 GPU, 35-45 minutes total

## Authors

- Sara Iqbal (23K-0669)
- Areeba Arif (23K-0618)

Course: Deep Learning Project, FAST-NUCES, Spring 2026.

## License

Code: MIT. Datasets retain their original licenses (see individual dataset pages).
