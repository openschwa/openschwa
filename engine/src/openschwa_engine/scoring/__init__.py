"""Closed-set contrast scoring with committed calibration (M1).

Contract: per-phone GOP from aligned frame posteriors (alignment/); for each
focus phone, renormalize posterior mass over {target} + confusion_set, then
margin, then Platt-calibrated confidence, then verdict (on_target /
substituted / uncertain). Operating thresholds live in scoring/calibration.yaml,
produced ONLY by the eval harness (eval/) - never hand-tuned. Precision >= 0.90
on held-out L2 data is the shipping bar (eval/README.md).
"""

from openschwa_engine.scoring.calibration import (
    CALIBRATION_PATH,
    AlignmentCalibration,
    Calibration,
    ContrastCalibration,
    load_calibration,
    load_or_fail,
)
from openschwa_engine.scoring.contrast import ContrastScore, decide, score_contrast

__all__ = [
    "CALIBRATION_PATH",
    "AlignmentCalibration",
    "Calibration",
    "ContrastCalibration",
    "ContrastScore",
    "decide",
    "load_calibration",
    "load_or_fail",
    "score_contrast",
]
