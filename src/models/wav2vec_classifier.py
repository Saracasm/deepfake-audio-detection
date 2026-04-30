"""
Wav2Vec 2.0-based classifier for deepfake audio detection.

Architecture:
    Raw waveform (16 kHz, 4 sec, 64000 samples)
        → Wav2Vec 2.0 Base backbone (95M params, 12 transformer layers)
        → mean pooling over time dimension
        → linear classification head (768 → 2)
        → logits for [bonafide, spoof]

In Stage 1, the backbone is frozen and only the head is trained.
In Stage 2, the top N transformer layers are unfrozen and fine-tuned.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2VecClassifier(nn.Module):
    """Wav2Vec 2.0 + mean pooling + linear head."""

    def __init__(
        self,
        backbone_name: str = "facebook/wav2vec2-base",
        num_classes: int = 2,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # Load pretrained backbone from HuggingFace
        self.backbone = Wav2Vec2Model.from_pretrained(backbone_name)

        # Get hidden size from the backbone config (768 for Base, 1024 for Large)
        hidden_size = self.backbone.config.hidden_size

        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Freeze backbone if requested (Stage 1 default)
        self.freeze_backbone(freeze_backbone)

    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze the entire Wav2Vec backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def unfreeze_top_n_layers(self, n: int):
        """Unfreeze only the top N transformer layers (Stage 2)."""
        # First freeze everything
        self.freeze_backbone(True)

        # Then unfreeze top N transformer encoder layers
        total_layers = len(self.backbone.encoder.layers)
        for i in range(total_layers - n, total_layers):
            for param in self.backbone.encoder.layers[i].parameters():
                param.requires_grad = True

        # Also unfreeze the layer norm at the end (small but matters)
        for param in self.backbone.encoder.layer_norm.parameters():
            param.requires_grad = True

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: (batch_size, num_samples) tensor of raw audio at 16 kHz

        Returns:
            logits: (batch_size, num_classes) tensor of unnormalized scores
        """
        # Backbone produces (batch, time_frames, hidden_size)
        outputs = self.backbone(waveforms)
        hidden_states = outputs.last_hidden_state  # (B, T, H)

        # Mean pool over time dimension → (B, H)
        pooled = hidden_states.mean(dim=1)

        # Classification head → (B, num_classes)
        logits = self.classifier(pooled)
        return logits

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
