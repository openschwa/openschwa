"""L2-ARCTIC adapter (https://psi.engr.tamu.edu/l2-arctic-corpus/). M1.

Six L1 backgrounds; TextGrid annotations tag substitutions/deletions
(e.g. "ð,d,s" tiers). The adapter parses annotation TextGrids and maps
labels into the canonical IPA inventory.
"""

from collections.abc import Iterator

from .base import DatasetAdapter, LabeledToken


class L2Arctic(DatasetAdapter):
    def tokens(self, target_phone: str) -> Iterator[LabeledToken]:
        raise NotImplementedError("M1")
