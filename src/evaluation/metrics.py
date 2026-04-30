"""
Evaluation metrics for anti-spoofing.

The standard metric in this field is Equal Error Rate (EER):
    The threshold at which the false-accept rate (FAR) equals
    the false-reject rate (FRR).

For a binary classifier outputting per-utterance scores:
    score > threshold  → predict spoof
    score <= threshold → predict bonafide

We sweep thresholds, compute FAR and FRR at each, and find the
threshold where they are equal. The error rate at that point is the EER.

Lower EER = better. State-of-the-art on ASVspoof 2019 LA is below 1%;
a strong student project lands in the 2-6% range.
"""

import numpy as np
from typing import Tuple
from sklearn.metrics import roc_curve, roc_auc_score


def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and the threshold at which it occurs.

    Args:
        scores: 1-D array of per-sample scores. Higher = more spoof-like.
                We use P(spoof) from softmax for this.
        labels: 1-D array of binary ground-truth labels.
                0 = bonafide, 1 = spoof.

    Returns:
        eer: Equal error rate as a fraction in [0, 1].
        threshold: Score threshold at which FAR equals FRR.
    """
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel()

    # roc_curve returns false-positive rate, true-positive rate, thresholds.
    # FPR = FAR (spoofs accepted as bonafide ... wait, careful).
    # Convention here:
    #   "positive" = spoof (label=1)
    #   FAR = false alarm = bonafide flagged as spoof = FPR
    #   FRR = miss = spoof predicted as bonafide = 1 - TPR
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr  # false negative rate = miss rate

    # Find the threshold where FAR (fpr) is closest to FRR (fnr)
    abs_diffs = np.abs(fpr - fnr)
    idx_eer = np.argmin(abs_diffs)
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2.0
    threshold = thresholds[idx_eer]

    return float(eer), float(threshold)


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve. 0.5 = random, 1.0 = perfect."""
    return float(roc_auc_score(labels, scores))


def aggregate_window_scores_to_utterance(
    window_scores: np.ndarray,
    window_utt_ids: list,
    method: str = "mean",
) -> Tuple[np.ndarray, list]:
    """Aggregate per-window scores back to per-utterance scores.

    Many utterances in our dataset produce multiple 4-sec windows.
    For evaluation we need ONE score per utterance, so we aggregate
    the per-window scores.

    Args:
        window_scores: shape (N_windows,) — scores for every window.
        window_utt_ids: list of N_windows utterance IDs (with duplicates).
        method: "mean" or "max" — how to aggregate within each utterance.

    Returns:
        utt_scores: shape (N_utterances,) — one score per unique utterance.
        utt_ids: list of N_utterances unique utterance IDs (sorted).
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for s, uid in zip(window_scores, window_utt_ids):
        grouped[uid].append(float(s))

    utt_ids_sorted = sorted(grouped.keys())
    if method == "mean":
        utt_scores = np.array([np.mean(grouped[u]) for u in utt_ids_sorted])
    elif method == "max":
        utt_scores = np.array([np.max(grouped[u]) for u in utt_ids_sorted])
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return utt_scores, utt_ids_sorted
