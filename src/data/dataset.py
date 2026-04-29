"""
PyTorch Dataset for ASVspoof 2019 LA.

One Dataset entry = one 4-second audio window + its binary label.
Long utterances expand into multiple entries (one per window).
"""

from typing import List, Tuple
import torch
from torch.utils.data import Dataset

from src.data.protocols import Utterance
from src.data.preprocessing import (
    load_audio,
    segment_waveform,
    WINDOW_SAMPLES,
    HOP_SAMPLES,
)


def _count_windows(duration_samples: int) -> int:
    if duration_samples <= WINDOW_SAMPLES:
        return 1
    n = (duration_samples - WINDOW_SAMPLES + HOP_SAMPLES - 1) // HOP_SAMPLES + 1
    return max(1, n)


class ASVspoofDataset(Dataset):
    """One sample = one 4-sec window + its label."""

    def __init__(
        self,
        utterances: List[Utterance],
        durations_samples: List[int] = None,
    ):
        self.utterances = utterances
        self.index: List[Tuple[int, int]] = []
        if durations_samples is None:
            for i in range(len(utterances)):
                self.index.append((i, 0))
        else:
            assert len(durations_samples) == len(utterances)
            for i, d in enumerate(durations_samples):
                n_windows = _count_windows(d)
                for w in range(n_windows):
                    self.index.append((i, w))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        utt_idx, window_idx = self.index[idx]
        utt = self.utterances[utt_idx]
        waveform = load_audio(utt.flac_path)
        windows = segment_waveform(waveform)
        window_idx = min(window_idx, len(windows) - 1)
        return windows[window_idx], utt.label_int, utt.utterance_id
