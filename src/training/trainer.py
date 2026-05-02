"""
Training loop for Wav2Vec-based deepfake audio detection.

Supports both Stage 1 (frozen backbone, simple) and Stage 2 (fine-tuning,
with mixed precision + warmup scheduler).
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
    """Hyperparameters for one training stage."""
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
    class_weights: Optional[List[float]] = None
    # Stage 2 additions
    use_mixed_precision: bool = False
    warmup_ratio: float = 0.0  # fraction of total steps used for LR warmup
    use_lr_scheduler: bool = False  # set True with warmup_ratio > 0


def make_loss_fn(class_weights: Optional[List[float]], device: str) -> Callable:
    if class_weights is not None:
        weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weights)
    return nn.CrossEntropyLoss()


def make_lr_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    """Linear warmup followed by linear decay to zero."""
    warmup_steps = int(total_steps * warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dev_loader: DataLoader,
    device: str,
    desc: str = "Eval",
    use_mixed_precision: bool = False,
) -> dict:
    """Run inference over the dev set and compute per-utterance metrics."""
    model.eval()
    all_window_scores, all_window_labels, all_window_utts = [], [], []
    total_loss, n_batches = 0.0, 0
    loss_fn = nn.CrossEntropyLoss()

    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_mixed_precision)

    for waveforms, labels, utt_ids in tqdm(dev_loader, desc=desc, leave=False):
        waveforms = waveforms.to(device, non_blocking=True)
        labels_gpu = labels.to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(waveforms)
            loss = loss_fn(logits, labels_gpu)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.softmax(logits.float(), dim=-1)
        spoof_probs = probs[:, 1].detach().cpu().numpy()

        all_window_scores.extend(spoof_probs.tolist())
        all_window_labels.extend(labels.tolist())
        all_window_utts.extend(list(utt_ids))

    utt_scores, utt_ids_sorted = aggregate_window_scores_to_utterance(
        np.array(all_window_scores), all_window_utts, method="mean",
    )
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
        "eer": eer, "auc": auc, "accuracy": accuracy,
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
    """Train the model for `config.epochs` epochs, evaluating each epoch."""
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
                "use_mixed_precision": config.use_mixed_precision,
                "warmup_ratio": config.warmup_ratio,
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
            settings=_wandb.Settings(init_timeout=180),
        )
        wandb = _wandb

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Optional LR scheduler with warmup
    scheduler = None
    if config.use_lr_scheduler and config.warmup_ratio > 0:
        total_steps = len(train_loader) * config.epochs
        scheduler = make_lr_scheduler(optimizer, total_steps, config.warmup_ratio)

    # Mixed precision setup
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_mixed_precision)
    autocast_ctx_factory = lambda: torch.amp.autocast(
        device_type="cuda", enabled=config.use_mixed_precision
    )

    loss_fn = make_loss_fn(config.class_weights, device)

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
            with autocast_ctx_factory():
                logits = model(waveforms)
                loss = loss_fn(logits, labels)

            if config.use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            epoch_losses.append(loss.item())
            global_step += 1

            if wandb is not None and global_step % config.log_every_n_steps == 0:
                log_data = {
                    "train/step_loss": loss.item(),
                    "train/global_step": global_step,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                wandb.log(log_data)

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss = float(np.mean(epoch_losses))
        dev_metrics = evaluate(
            model, dev_loader, device,
            desc=f"Epoch {epoch+1} dev",
            use_mixed_precision=config.use_mixed_precision,
        )
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["dev_eer"].append(dev_metrics["eer"])
        history["dev_auc"].append(dev_metrics["auc"])
        history["dev_accuracy"].append(dev_metrics["accuracy"])

        print(f"\nEpoch {epoch+1}/{config.epochs} ({epoch_time:.0f}s)")
        print(f"  train_loss:   {train_loss:.4f}")
        print(f"  dev_eer:      {dev_metrics['eer']*100:.2f}%")
        print(f"  dev_auc:      {dev_metrics['auc']:.4f}")
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

        if dev_metrics["eer"] < best_eer:
            best_eer = dev_metrics["eer"]
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_eer": best_eer,
                "config": vars(config),
            }, checkpoint_path)
            print(f"  → Saved best checkpoint (EER={best_eer*100:.2f}%)")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"\nEarly stopping after {epoch+1} epochs.")
            break

    if wandb is not None:
        wandb.summary["best_dev_eer"] = best_eer
        wandb.finish()

    return {"history": history, "best_eer": best_eer, "checkpoint_path": checkpoint_path}
