"""The committed calibration: how raw contrast scores map to verdicts.

calibration.yaml lives next to this module and is produced ONLY by the eval
harness (eval/) - see eval/README.md for the procedure and the shipping bar.
The engine treats it as read-only truth: a missing or invalid file, or a file
whose model_id does not match the running alignment model, means the engine
cannot judge, which it reports loudly and then degrades to honest silence
rather than guessing.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from openschwa_engine.scoring.contrast import ContrastScore

log = logging.getLogger(__name__)

CALIBRATION_PATH = Path(__file__).with_name("calibration.yaml")


class PlattCalibration(BaseModel):
    """Sigmoid parameters: p = sigmoid(a * score + b)."""

    model_config = ConfigDict(extra="forbid")

    a: float
    b: float

    def probability(self, score: float) -> float:
        return 1.0 / (1.0 + math.exp(-(self.a * score + self.b)))


class ContrastCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    confusions: list[str]
    #: Which raw aggregation the Platt fit applies to: "mean" (the label-frame
    #: average), "spike" (the single most confusion-favouring frame), "vote"
    #: (share of frames a confusion wins), or "gop" (the whole-vocabulary
    #: goodness-of-pronunciation). Chosen by the eval harness.
    score_variant: Literal["mean", "spike", "vote", "gop"] = "mean"
    #: Maps the raw contrast score (log(p_best_confusion / p_target)) to
    #: P(substituted) - fitted by the eval harness on the train split.
    substitution_platt: PlattCalibration
    #: Operating point from the precision-first PR sweep (train split only).
    #: One threshold for every learner: the judge is blind to who is speaking.
    #: The harness's per-L1 breakdown exists to *audit* that blindness (does
    #: the single line treat every language group fairly?), not to ship
    #: per-language lines.
    threshold: float = Field(ge=0.5, le=1.0)
    #: Maps GOP to Phone.score in [0, 1]; null while GOP calibration is absent.
    gop_platt: PlattCalibration | None = None

    def score_of(self, raw: ContrastScore, gop: float | None = None) -> float | None:
        """The raw value the calibration was fitted on, by variant.

        Returns None when the variant's evidence is absent (no GOP measured),
        which the caller must treat as 'uncertain', never as a verdict.
        """
        if self.score_variant == "spike":
            return raw.spike_score
        if self.score_variant == "vote":
            return raw.vote_fraction
        if self.score_variant == "gop":
            return gop
        return raw.score


class AlignmentCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(ge=0, le=1)
    low_confidence: float = Field(ge=0, le=1)


class Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    #: Path of the committed eval report that produced this file. Every
    #: shipped threshold stays traceable to its evidence (eval/README.md).
    generated_by: str
    #: The acoustic model this was fitted for; verdicts computed against a
    #: different model's calibration are exactly the wrong-verdict failure
    #: mode the whole design exists to prevent, so that is refused loudly.
    model_id: str
    contrasts: list[ContrastCalibration]
    alignment: AlignmentCalibration

    def contrast(self, target: str) -> ContrastCalibration | None:
        return next((c for c in self.contrasts if c.target == target), None)


_missing_warned = False


def load_calibration(path: Path = CALIBRATION_PATH) -> Calibration | None:
    """Load the committed calibration, or None (never raise) when unusable.

    A corrupt or missing file must not crash the engine: it degrades to
    uncertain verdicts and no segmental feedback - honest silence. The
    missing-file warning fires once per process: the eval harness calls this
    per token, and a 7000-line warning chorus buries the real logs.
    """
    global _missing_warned
    if not path.is_file():
        if not _missing_warned:
            _missing_warned = True
            log.warning(
                "calibration file %s is missing - the engine cannot judge pronunciation", path
            )
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return Calibration.model_validate(raw)
    except (yaml.YAMLError, ValidationError, OSError) as exc:
        log.error("calibration file %s is unusable (%s) - the engine cannot judge", path, exc)
        return None


def load_or_fail(path: Path = CALIBRATION_PATH) -> Calibration:
    """Hard-failing variant for tests and for the eval harness."""
    calibration = load_calibration(path)
    if calibration is None:
        raise RuntimeError(f"calibration file {path} is missing or unusable")
    return calibration
