"""Dataset-adapter interface: map a corpus into canonical-IPA labeled tokens.

Each adapter owns one corpus's quirks (label formats, phone inventories,
directory layout) and yields uniform LabeledTokens the harness can score
against engine output. Implemented in M1.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabeledToken:
    """One expert-labeled phone token."""

    utterance_id: str
    audio_path: Path
    transcript: str
    l1: str  # speaker's first language, for per-L1 metric breakdown
    target_phone: str  # canonical IPA (mapped from the corpus inventory)
    label: str  # "correct" | "substituted" | "deleted"
    substituted_with: str | None = None  # canonical IPA, when label == "substituted"


class DatasetAdapter(ABC):
    def __init__(self, root: Path) -> None:
        self.root = root

    @abstractmethod
    def tokens(self, target_phone: str) -> Iterator[LabeledToken]:
        """Yield every labeled token of `target_phone` (canonical IPA)."""
