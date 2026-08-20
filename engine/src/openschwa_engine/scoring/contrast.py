"""Closed-set contrast scoring for one focus phone (M1).

Pure numpy, no torch: the acoustic model produced a log-posterior matrix and
this module only needs the focus phone's label frames plus the {target} ∪
confusions set from the exercise.

Renormalizing over that closed set is the designed fix for GOP's
whole-vocabulary denominator (docs/architecture.md section 9): a Mandarin
tone-tagged vowel no longer competes with an English phone it has nothing to
do with. The raw score here is *uncalibrated evidence*; turning it into a
verdict and a confidence is decide(), which needs the committed
calibration.yaml (calibration.py). The eval harness produces that file;
the engine never invents it.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from openschwa_engine.models.phone_set import PhoneMap
from openschwa_engine.scoring.calibration import ContrastCalibration

LogProbs = npt.NDArray[np.float32]

EPS = 1e-12


@dataclass(frozen=True)
class ContrastScore:
    """Raw (pre-calibration) closed-set evidence for one focus phone.

    Three aggregations of the same per-frame evidence, because CTC posteriors
    are peaky and the bake-off (docs/architecture.md) names this explicitly:
    the label-frame mean dilutes a substitution that wins only a few frames,
    while spike-frame scoring keeps exactly that evidence.

    The mirror (M1 pivot, docs/research/mirror-pivot) reads the same evidence
    as a *hearing*: `heard` is the argmax phone over the closed set and
    `hearing_score` its odds, calibrated separately into P(heard == realized)
    by the eval harness's hearing block.
    """

    target: str
    confusions: tuple[str, ...]
    posteriors: dict[str, float]  # renormalized over {target} + confusions; sums to 1
    score: float  # mean: log(p_best_confusion / p_target); > 0 = substitution-like
    spike_score: float  # the single frame most favouring the best confusion
    vote_fraction: float  # share of label frames where a confusion outvotes the target
    best_confusion: str  # the confusion with the most posterior mass
    heard: str  # argmax over {target} + confusions - the phone the model heard
    hearing_score: float  # log(p_heard / (1 - p_heard)); calibrated into P(heard == realized)


def score_contrast(
    log_probs: LogProbs,
    frame_indices: "npt.NDArray[np.int64] | tuple[int, ...]",
    target: str,
    confusions: "tuple[str, ...] | list[str]",
    phone_map: PhoneMap,
) -> ContrastScore:
    """Renormalize the focus phone's label-frame posteriors over the closed set.

    Averaging happens over the *label* frames of the aligned focus phone, the
    same choice the GOP in alignment/ctc.py makes, and in probability space,
    so the result is the expected posterior mass of each candidate over the
    segment the UI highlights.
    """
    if not confusions:
        raise ValueError("a contrast needs a non-empty confusion set")
    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.size == 0:
        raise ValueError("the focus phone received no label frames")
    if frames.min() < 0 or frames.max() >= log_probs.shape[0]:
        raise ValueError("focus frame indices fall outside the posterior matrix")

    target_index = phone_map.to_index(target)
    confusion_indices = [phone_map.to_index(c) for c in confusions]
    closed = np.array([target_index, *confusion_indices], dtype=np.int64)

    logp = log_probs[frames][:, closed].astype(np.float64)
    shifted = logp - logp.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    # The mean aggregates MASS, not frame votes: CTC outputs are blank-dominant
    # (a 250 ms segment is ~95% blank frames), and renormalizing per frame
    # before averaging gives every blank frame's dust a full vote - the v4
    # exam measured that as an AUC collapse to chance. Mass weighting is also
    # what the training-time validation measured, so the exam finally scores
    # the model on the same aggregation it was selected on.
    mean = np.exp(logp).mean(axis=0)
    mean /= mean.sum() + EPS  # numerical dust back into the normalization

    names = [target, *confusions]
    posteriors = {name: float(mean[i]) for i, name in enumerate(names)}
    best_confusion = max(confusions, key=lambda c: posteriors[c])
    score = float(np.log((posteriors[best_confusion] + EPS) / (posteriors[target] + EPS)))
    heard = max(names, key=lambda name: posteriors[name])
    p_heard = posteriors[heard]
    hearing_score = float(np.log((p_heard + EPS) / (1.0 - p_heard + EPS)))

    # Spike frame: the single frame where a confusion beat the target hardest.
    # A substitution that wins only a frame or two survives here even when the
    # interval mean washes it out (the CTC peakiness caveat).
    confusion_mass = probs[:, 1:].max(axis=1)
    target_mass = probs[:, 0]
    per_frame_ratio = np.log((confusion_mass + EPS) / (target_mass + EPS))
    spike_score = float(per_frame_ratio.max(axis=0))

    # Vote: the share of label frames where some confusion outvoted the target.
    vote_fraction = float(np.mean(confusion_mass > target_mass))

    return ContrastScore(
        target=target,
        confusions=tuple(confusions),
        posteriors=posteriors,
        score=score,
        spike_score=spike_score,
        vote_fraction=vote_fraction,
        best_confusion=best_confusion,
        heard=heard,
        hearing_score=hearing_score,
    )


def decide(
    raw: ContrastScore,
    calibration: ContrastCalibration,
    gop: float | None = None,
) -> tuple[Literal["on_target", "substituted", "uncertain"], float, str | None]:
    """Turn a raw contrast score into a verdict via the committed calibration.

    Returns (verdict, confidence, detected). The confidence is the
    Platt-calibrated P(substituted), the number the eval harness swept to
    choose the operating point, and the number the composer gates on.

    The verdict is decided against the *calibrated* operating threshold, not a
    fixed 0.5: below the band it is on_target, above it substituted, and
    inside the band uncertain; the engine refuses to guess there. One
    threshold serves every learner: the judge is blind to who is speaking. A
    variant whose evidence is absent (e.g. a GOP calibration with no GOP)
    degrades to uncertain rather than guessing. A calibration without a judge
    fit (no substitution_platt / threshold - the mirror-only case) yields
    uncertain too: it has nothing to judge with.
    """
    if calibration.substitution_platt is None or calibration.threshold is None:
        return "uncertain", 0.0, None
    value = calibration.score_of(raw, gop)
    if value is None:
        return "uncertain", 0.0, None
    p_sub = calibration.substitution_platt.probability(value)
    threshold = calibration.threshold
    if p_sub >= threshold:
        return "substituted", p_sub, raw.best_confusion
    if p_sub <= 1.0 - threshold:
        return "on_target", p_sub, None
    return "uncertain", p_sub, None
