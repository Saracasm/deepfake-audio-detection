"""
Audio preprocessing for ASVspoof 2019 LA.

Given a .flac path, produces a list of fixed-length 4-second segments at 16kHz
mono, ready to feed into Wav2Vec 2.0.

Pipeline:
    load .flac -> ensure mono -> ensure 16kHz -> window with 50% overlap
    (short clips are zero-padded to one full window)
"""

from typing import List
import torch
import torchaudio
import torchaudio.functional as F


SAMPLE_RATE = 16000
WINDOW_SECONDS = 4.0
OVERLAP_RATIO = 0.5

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)              # 64000
HOP_SAMPLES = int(WINDOW_SAMPLES * (1.0 - OVERLAP_RATIO))       # 32000


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load a .flac file and return a 1-D mono waveform at target_sr."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = F.resample(waveform, orig_freq=sr, new_freq=target_sr)
    return waveform.squeeze(0)


def segment_waveform(
    waveform: torch.Tensor,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
) -> List[torch.Tensor]:
    """Split a 1-D waveform into fixed-length windows with overlap."""
    n = waveform.shape[0]
    if n <= window_samples:
        padded = torch.zeros(window_samples, dtype=waveform.dtype)
        padded[:n] = waveform
        return [padded]

    windows = []
    start = 0
    while start < n:
        end = start + window_samples
        if end <= n:
            windows.append(waveform[start:end])
        else:
            tail = waveform[start:]
            padded = torch.zeros(window_samples, dtype=waveform.dtype)
            padded[:tail.shape[0]] = tail
            windows.append(padded)
            break
        start += hop_samples
    return windows


def preprocess(path: str) -> List[torch.Tensor]:
    """Full pipeline: load + window."""
    waveform = load_audio(path)
    return segment_waveform(waveform)
