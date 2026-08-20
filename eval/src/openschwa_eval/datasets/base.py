"""Dataset-adapter interface: map a corpus into canonical-IPA labeled utterances.

Each adapter owns one corpus's quirks (label formats, phone inventories,
directory layout) and yields uniform Utterance objects the harness can run
through the engine and score against. M1.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhoneToken:
    """One phone of an utterance, with its expert label where annotated."""

    index: int
    phone: str  # canonical IPA
    start_s: float | None = None  # interval in the audio, when the corpus has it
    end_s: float | None = None
    label: str = "correct"  # correct | substituted | deleted
    substituted_with: str | None = None  # canonical IPA, when substituted
    #: How many of speechocean762's five experts marked this token as an
    #: error. None for corpora without per-token expert votes.
    expert_error_votes: int | None = None


@dataclass(frozen=True)
class Utterance:
    """One labeled utterance: everything the harness needs for one analysis."""

    utterance_id: str
    audio_path: Path
    transcript: str
    l1: str  # speaker's first language, for per-L1 metric breakdown
    phones: tuple[PhoneToken, ...]
    split: str = ""  # corpus-native train/test split when one exists (speechocean762)
    corpus: str = ""  # adapter name, for the per-corpus breakdown
    #: Speaker id, when the corpus exposes it. The three-way split is
    #: speaker-disjoint: a speaker's voice must never leak across train/cal/test.
    speaker: str = ""

    def tokens(self, target_phone: str) -> Iterator[PhoneToken]:
        """Every occurrence of `target_phone` in the canonical sequence."""
        for token in self.phones:
            if token.phone == target_phone:
                yield token


class DatasetAdapter(ABC):
    def __init__(self, root: Path) -> None:
        self.root = root

    @abstractmethod
    def utterances(self, target_phone: str) -> Iterator[Utterance]:
        """Yield every labeled utterance containing `target_phone`.

        Utterances without expert labels for that phone are excluded: they
        cannot score anything. Adapters fail loud on unmappable labels.
        """
