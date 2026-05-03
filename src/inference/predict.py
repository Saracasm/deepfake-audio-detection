"""
Inference module for deepfake audio detection.

Wraps the Stage 2 Wav2Vec 2.0 classifier with a clean public API.

Usage:
    from src.inference.predict import DeepfakeDetector
    detector = DeepfakeDetector(checkpoint_path="path/to/stage2_best.pt")
    result = detector.predict("path/to/audio.wav")
    print(result)
    # {"spoof_probability": 0.84, "prediction": "spoof", "confidence": 0.84,
    #  "utterance_duration_sec": 3.42, "n_windows": 1, "model_version": "stage2"}
"""

import os
from typing import Dict, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np

from src.models.wav2vec_classifier import Wav2VecClassifier
from src.data.preprocessing import load_audio, segment_waveform, WINDOW_SAMPLES


# Default classifier threshold. 0.5 is naive; we tuned it during eval.
# Values closer to 0.5 = balanced; lower = more sensitive (more false alarms);
# higher = more conservative (more misses).
DEFAULT_THRESHOLD = 0.5


class DeepfakeDetector:
    """Anti-spoofing classifier wrapper for one-shot inference."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        backbone_name: str = "facebook/wav2vec2-base",
        threshold: float = DEFAULT_THRESHOLD,
        use_mixed_precision: bool = True,
    ):
        """
        Args:
            checkpoint_path: path to a Stage 2 .pt checkpoint
            device: 'cuda', 'cpu', or None (auto-detect)
            backbone_name: HuggingFace model name for Wav2Vec backbone
            threshold: probability threshold above which we predict "spoof"
            use_mixed_precision: use fp16 inference (faster on GPU)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.threshold = threshold
        self.use_mixed_precision = use_mixed_precision and (device == "cuda")

        # Build model and load weights
        self.model = Wav2VecClassifier(
            backbone_name=backbone_name,
            num_classes=2,
            freeze_backbone=True,
        )
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(device)
        self.model.eval()

        # Store metadata for transparency
        self.checkpoint_metadata = {
            "epoch": ckpt.get("epoch"),
            "best_eer": ckpt.get("best_eer"),
            "checkpoint_path": checkpoint_path,
        }

    @torch.no_grad()
    def predict(
        self,
        audio_input: Union[str, torch.Tensor, np.ndarray],
        return_per_window: bool = False,
    ) -> Dict:
        """Predict bonafide vs spoof for a single audio input.

        Args:
            audio_input: either a file path (str), a 1-D Tensor at 16 kHz, or
                         a 1-D numpy array at 16 kHz.
            return_per_window: if True, include per-window probabilities in
                               the result for debugging.

        Returns:
            Dict with keys:
                spoof_probability: float in [0, 1]
                bonafide_probability: float in [0, 1]
                prediction: "bonafide" or "spoof"
                confidence: float in [0, 1] (probability of the predicted class)
                utterance_duration_sec: total audio length
                n_windows: number of 4-sec windows the audio was split into
                window_scores: (only if return_per_window=True) list of per-window spoof probs
        """
        # Step 1: Load and resample audio if needed
        if isinstance(audio_input, str):
            waveform = load_audio(audio_input)  # returns 1-D tensor at 16 kHz
        elif isinstance(audio_input, np.ndarray):
            waveform = torch.from_numpy(audio_input.astype(np.float32))
        elif isinstance(audio_input, torch.Tensor):
            waveform = audio_input.float()
            if waveform.dim() > 1:
                waveform = waveform.squeeze()
        else:
            raise ValueError(
                f"audio_input must be str, np.ndarray, or torch.Tensor; got {type(audio_input)}"
            )

        duration_sec = float(waveform.shape[0] / 16000)

        # Step 2: Segment into 4-sec windows
        windows = segment_waveform(waveform)  # list of 1-D tensors of length 64000
        n_windows = len(windows)

        # Step 3: Stack into a batch and run inference
        batch = torch.stack(windows, dim=0).to(self.device, non_blocking=True)

        if self.use_mixed_precision:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = self.model(batch)
        else:
            logits = self.model(batch)

        # Step 4: Compute per-window probabilities, then aggregate (mean)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()  # (n_windows, 2)
        window_spoof_probs = probs[:, 1].tolist()
        utt_spoof_prob = float(np.mean(window_spoof_probs))
        utt_bonafide_prob = 1.0 - utt_spoof_prob

        # Step 5: Apply threshold for hard prediction
        prediction = "spoof" if utt_spoof_prob > self.threshold else "bonafide"
        confidence = utt_spoof_prob if prediction == "spoof" else utt_bonafide_prob

        result = {
            "spoof_probability": utt_spoof_prob,
            "bonafide_probability": utt_bonafide_prob,
            "prediction": prediction,
            "confidence": confidence,
            "utterance_duration_sec": duration_sec,
            "n_windows": n_windows,
            "threshold_used": self.threshold,
        }
        if return_per_window:
            result["window_scores"] = window_spoof_probs
        return result

    def info(self) -> Dict:
        """Return metadata about this model checkpoint."""
        return {
            **self.checkpoint_metadata,
            "device": self.device,
            "threshold": self.threshold,
            "mixed_precision": self.use_mixed_precision,
        }
