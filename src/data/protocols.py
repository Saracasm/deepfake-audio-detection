"""
ASVspoof 2019 LA protocol parser.

Reads the official .txt protocol files and yields structured Utterance objects
that downstream code (datasets, evaluation) can use.

Protocol file format (5 space-separated columns):
    speaker_id  utterance_id  -  attack_id  label

    speaker_id   : anonymized speaker (e.g., "LA_0079")
    utterance_id : filename without extension (e.g., "LA_T_1138215")
    column 3     : unused, always "-"
    attack_id    : "-" for bonafide, "A01"-"A19" for spoof samples
    label        : "bonafide" or "spoof"
"""

from dataclasses import dataclass
from typing import List, Dict
import os


@dataclass
class Utterance:
    """One row from an ASVspoof 2019 LA protocol file."""
    speaker_id: str
    utterance_id: str
    attack_id: str        # "-" for bonafide, "A01"-"A19" for spoof
    label: str            # "bonafide" or "spoof"
    label_int: int        # 0 = bonafide, 1 = spoof
    flac_path: str        # absolute path to the .flac file


def parse_protocol(protocol_path: str, audio_root: str) -> List[Utterance]:
    """Parse one ASVspoof 2019 LA cm protocol file.

    Args:
        protocol_path: full path to the .txt protocol file.
        audio_root: full path to the folder containing the .flac files.

    Returns:
        List of Utterance objects, one per valid line.
    """
    utterances: List[Utterance] = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            speaker_id, utt_id, _unused, attack_id, label = parts
            label_int = 0 if label == "bonafide" else 1
            flac_path = os.path.join(audio_root, f"{utt_id}.flac")
            utterances.append(Utterance(
                speaker_id=speaker_id,
                utterance_id=utt_id,
                attack_id=attack_id,
                label=label,
                label_int=label_int,
                flac_path=flac_path,
            ))
    return utterances


def parse_all_partitions(la_root: str) -> Dict[str, List[Utterance]]:
    """Parse train, dev, and eval protocols at once.

    Args:
        la_root: path to the LA folder, e.g.
                 ".../asvspoof_2019/LA"

    Returns:
        Dict with keys "train", "dev", "eval" mapping to lists of Utterances.
    """
    proto_dir = os.path.join(la_root, "ASVspoof2019_LA_cm_protocols")
    partitions = {
        "train": (
            os.path.join(proto_dir, "ASVspoof2019.LA.cm.train.trn.txt"),
            os.path.join(la_root, "ASVspoof2019_LA_train", "flac"),
        ),
        "dev": (
            os.path.join(proto_dir, "ASVspoof2019.LA.cm.dev.trl.txt"),
            os.path.join(la_root, "ASVspoof2019_LA_dev", "flac"),
        ),
        "eval": (
            os.path.join(proto_dir, "ASVspoof2019.LA.cm.eval.trl.txt"),
            os.path.join(la_root, "ASVspoof2019_LA_eval", "flac"),
        ),
    }
    return {
        name: parse_protocol(proto, audio)
        for name, (proto, audio) in partitions.items()
    }


def class_counts(utterances: List[Utterance]) -> Dict[str, int]:
    """Return {'bonafide': N, 'spoof': M} counts."""
    counts = {"bonafide": 0, "spoof": 0}
    for u in utterances:
        counts[u.label] += 1
    return counts
