"""Forced alignment: CTC phoneme posteriors + Viterbi over the target sequence.

`ctc` is pure numpy and independently testable; `acoustic` is the only module
that touches torch; `aligner` joins them and puts intervals back on the original
upload timeline.
"""

from openschwa_engine.alignment.aligner import (
    AlignedPhone,
    AlignmentOutcome,
    align_exercise,
    canonical_targets,
)
from openschwa_engine.alignment.ctc import AlignmentError, PhoneSegment, align, forced_align

__all__ = [
    "AlignedPhone",
    "AlignmentError",
    "AlignmentOutcome",
    "PhoneSegment",
    "align",
    "align_exercise",
    "canonical_targets",
    "forced_align",
]
