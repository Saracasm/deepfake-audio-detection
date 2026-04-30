"""
Training loop for Wav2Vec-based deepfake audio detection.

Responsibilities:
    - Forward + backward pass with the Stage 1 or Stage 2 model
    - Per-step loss logging (to wandb if enabled)
    - Per-epoch dev-set evaluation: EER, AUC, accuracy
    - Checkpoint saving (best dev EER)
    - Early stopping on validation EER
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import (
    compute_eer,
    compute_auc,
    aggregate_window_scores_to_utterance,
)


@dataclass
class TrainConfig:
    """Hyperparameters and run-level settings for one training stage."""
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    early_stopping_patience: int = 3
    checkpoint_dir: str = "/content/drive/MyDrive/deepfake_audio/checkpoints"
    checkpoint_name: str = "stage1_best.pt"
    log_every_n_steps: int = 20
    use_wandb: bool = True
    wandb_project: str = "deepfake-audio-detection"
    wandb_run_name: Optional[str] = None
    class_weights: Optional[List[float]] = None  # for weighted loss


def make_loss_fn(class_weights: Optional[List[float]], device: str) -> Callable:
    """Build cross-entropy loss, optionally with class weights."""
    if class_weights is not None:
        weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weights)
    return nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dev_loader: DataLoader,
    device: str,
    desc: str = "Eval",
) -> dict:
    """Run inference over the dev set and compute metrics.

    Returns a dict with keys: eer, auc, accuracy, loss, n_utterances.
    """
    model.eval()
    all_window_scores = []
    all_window_labels = []
    all_window_utts = []
    total_loss = 0.0
    n_batches = 0
    loss_fn = nn.CrossEntropyLoss()

    for waveforms, labels, utt_ids in tqdm(dev_loader, desc=desc, leave=False):
        waveforms = waveforms.to(device, non_blocking=True)
        labels_gpu = labels.to(device, non_blocking=True)

        logits = model(waveforms)
        loss = loss_fn(logits, labels_gpu)
        total_loss += loss.item()
        n_batches += 1

        # Score = P(spoof) from softmax
        probs = torch.softmax(logits, dim=-1)
        spoof_probs = probs[:, 1].detach().cpu().numpy()

        all_window_scores.extend(spoof_probs.tolist())
        all_window_labels.extend(labels.tolist())
        all_window_utts.extend(list(utt_ids))

    # Aggregate per-window scores back to per-utterance scores (mean)
    utt_scores, utt_ids_sorted = aggregate_window_scores_to_utterance(
        np.array(all_window_scores),
        all_window_utts,
        method="mean",
    )

    # Get per-utterance labels (look up first occurrence of each id)
    label_map = {}
    for s, l, u in zip(all_window_scores, all_window_labels, all_window_utts):
        if u not in label_map:
            label_map[u] = l
    utt_labels = np.array([label_map[u] for u in utt_ids_sorted])

    eer, threshold = compute_eer(utt_scores, utt_labels)
    auc = compute_auc(utt_scores, utt_labels)
    preds = (utt_scores > threshold).astype(int)
    accuracy = float((preds == utt_labels).mean())

    return {
        "eer": eer,
        "auc": auc,
        "accuracy": accuracy,
        "threshold": float(threshold),
        "loss": total_loss / max(n_batches, 1),
        "n_utterances": len(utt_ids_sorted),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    config: TrainConfig,
    device: str = "cuda",
) -> dict:
    """Train the model for `config.epochs` epochs, evaluating on dev each epoch.

    Returns a history dict and saves the best checkpoint to disk.
    """
    # Set up wandb if requested
    wandb = None
    if config.use_wandb:
        import wandb as _wandb
        run = _wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "weight_decay": config.weight_decay,
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
            settings=_wandb.Settings(init_timeout=180),
        )
        wandb = _wandb

    # Optimizer: only train parameters with requires_grad=True (i.e., the head)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    loss_fn = make_loss_fn(config.class_weights, device)

    # Make sure checkpoint dir exists
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_dir, config.checkpoint_name)

    history = {"train_loss": [], "dev_eer": [], "dev_auc": [], "dev_accuracy": []}
    best_eer = float("inf")
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_start = time.time()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for waveforms, labels, utt_ids in pbar:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(waveforms)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

            if wandb is not None and global_step % config.log_every_n_steps == 0:
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/global_step": global_step,
                })

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # End of epoch — evaluate
        train_loss = float(np.mean(epoch_losses))
        dev_metrics = evaluate(model, dev_loader, device, desc=f"Epoch {epoch+1} dev")
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["dev_eer"].append(dev_metrics["eer"])
        history["dev_auc"].append(dev_metrics["auc"])
        history["dev_accuracy"].append(dev_metrics["accuracy"])

        print(f"\nEpoch {epoch+1}/{config.epochs} ({epoch_time:.0f}s)")
        print(f"  train_loss: {train_loss:.4f}")
        print(f"  dev_eer:    {dev_metrics['eer']*100:.2f}%")
        print(f"  dev_auc:    {dev_metrics['auc']:.4f}")
        print(f"  dev_accuracy: {dev_metrics['accuracy']*100:.2f}%")

        if wandb is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                "dev/eer": dev_metrics["eer"],
                "dev/auc": dev_metrics["auc"],
                "dev/accuracy": dev_metrics["accuracy"],
                "dev/loss": dev_metrics["loss"],
            })

        # Checkpoint if best
        if dev_metrics["eer"] < best_eer:
            best_eer = dev_metrics["eer"]
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_eer": best_eer,
                "config": vars(config),
            }, checkpoint_path)
            print(f"  → Saved best checkpoint (EER={best_eer*100:.2f}%)")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {epochs_without_improvement}).")
            break

    if wandb is not None:
        wandb.summary["best_dev_eer"] = best_eer
        wandb.finish()

    return {"history": history, "best_eer": best_eer, "checkpoint_path": checkpoint_path}
