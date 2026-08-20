"""Prosody: F0 extraction, semitone normalisation, and (from M2) reference
comparison.

M0 ships the learner's own contour so the UI has a pitch curve to draw. DTW
against the teacher reference and rule-based nuclear tone classification arrive
in M2, once reference recordings exist — the contract already carries the
`reference`, `dtw_distance`, and `nuclear_tone` fields they populate.
"""

from openschwa_engine.prosody.compare import contour_match, dtw_distance, nuclear_tone
from openschwa_engine.prosody.f0 import F0Track, reference_track, track

__all__ = [
    "F0Track",
    "contour_match",
    "dtw_distance",
    "nuclear_tone",
    "reference_track",
    "track",
]
