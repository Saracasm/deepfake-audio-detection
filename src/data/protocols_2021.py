"""
ASVspoof 2021 LA protocol parser.

Format (8 space-separated columns):
    speaker_id  utterance_id  codec  channel  attack_id  label  trim  partition

    speaker_id   : anonymized speaker
    utterance_id : filename without extension (e.g., "LA_E_9332881")
    codec        : audio codec applied (alaw, ulaw, g722, mp3, pcm, ...)
    channel      : transmission channel (ita_tx, sin_tx, loc_tx, ...)
    attack_id    : "-" for bonafide, "A07"-"A19" for spoof
    label        : "bonafide" or "spoof"
    trim         : "trim" or "notrim"
    partition    : "eval", "progress", or "hidden"
"""

from dataclasses import dataclass
from typing import List, Optional
import os


@dataclass
class Utterance2021:
    """One row from an ASVspoof 2021 LA cm protocol file."""
    speaker_id: str
    utterance_id: str
    codec: str
    channel: str
    attack_id: str
    label: str
    label_int: int
    trim: str
    partition: str
    flac_path: str


def parse_protocol_2021(
    protocol_path: str,
    audio_root: str,
    partition_filter: Optional[str] = "eval",
) -> List[Utterance2021]:
    """Parse the 2021 LA cm protocol with keys.

    Args:
        protocol_path: full path to trial_metadata.txt
        audio_root: full path to the flac/ folder
        partition_filter: only return rows matching this partition.
                          Valid: "eval", "progress", "hidden", or None for all.

    Returns:
        List of Utterance2021 objects.
    """
    utterances: List[Utterance2021] = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            speaker_id, utt_id, codec, channel, attack_id, label, trim, partition = parts

            if partition_filter is not None and partition != partition_filter:
                continue

            label_int = 0 if label == "bonafide" else 1
            flac_path = os.path.join(audio_root, f"{utt_id}.flac")

            utterances.append(Utterance2021(
                speaker_id=speaker_id,
                utterance_id=utt_id,
                codec=codec,
                channel=channel,
                attack_id=attack_id,
                label=label,
                label_int=label_int,
                trim=trim,
                partition=partition,
                flac_path=flac_path,
            ))
    return utterances
