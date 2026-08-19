"""Closed-set contrast scoring, tested on hand-built posteriors.

No model involved: the arithmetic is pure numpy, and every verdict the eval
harness reads depends on it being right. The PhoneMap here is synthetic - the
real one is covered by test_phone_set.py.
"""

import numpy as np
import pytest

from openschwa_engine.models.phone_set import PhoneMap
from openschwa_engine.scoring.calibration import ContrastCalibration, PlattCalibration
from openschwa_engine.scoring.contrast import ContrastScore, decide, score_contrast


def phone_map() -> PhoneMap:
    return PhoneMap(
        model_id="synthetic",
        table_name="test",
        token_of={"ð": "dh", "z": "z", "d": "d", "v": "v"},
        index_of={"ð": 1, "z": 2, "d": 3, "v": 4},
        blank_index=0,
    )


def log_probs_for(frame_tokens: list[int], confidence: float = 0.9) -> np.ndarray:
    """Posteriors that put 'confidence' on the intended token of each frame."""
    vocab = 6
    probs = np.full((len(frame_tokens), vocab), (1.0 - confidence) / (vocab - 1))
    probs[np.arange(len(frame_tokens)), frame_tokens] = confidence
    return np.log(probs).astype(np.float32)


def test_posteriors_renormalize_to_one():
    # Frames mostly on /z/ (token 2), some on /ð/ (token 1).
    raw = log_probs_for([1, 2, 2, 2], confidence=0.85)
    result = score_contrast(raw, [0, 1, 2, 3], "ð", ["z", "d", "v"], phone_map())
    assert sum(result.posteriors.values()) == pytest.approx(1.0, abs=1e-6)


def test_score_is_positive_when_a_confusion_dominates():
    raw = log_probs_for([2, 2, 2], confidence=0.9)
    result = score_contrast(raw, [0, 1, 2], "ð", ["z", "d", "v"], phone_map())
    assert result.score > 0
    assert result.best_confusion == "z"


def test_score_is_negative_when_the_target_dominates():
    raw = log_probs_for([1, 1, 1], confidence=0.9)
    result = score_contrast(raw, [0, 1, 2], "ð", ["z", "d", "v"], phone_map())
    assert result.score < 0
    # 0.9 on /ð/, 0.02 on each of z/d/v -> 0.9 / (0.9 + 3 * 0.02) after
    # renormalization over the closed set.
    assert result.posteriors["ð"] == pytest.approx(0.9 / 0.96, abs=1e-4)


def test_mass_moves_with_the_evidence():
    """A perfectly ambiguous recording yields a symmetric split."""
    probs = np.full((8, 6), 1e-6)
    probs[:, 1] = 0.5  # ð
    probs[:, 2] = 0.5  # z
    raw = np.log(probs).astype(np.float32)
    result = score_contrast(raw, np.arange(8), "ð", ["z", "d", "v"], phone_map())
    assert result.posteriors["ð"] == pytest.approx(0.5, abs=1e-6)
    assert result.posteriors["z"] == pytest.approx(0.5, abs=1e-6)
    assert result.score == pytest.approx(0.0, abs=1e-4)


def test_blank_frames_do_not_outvote_the_evidence():
    """CTC is blank-dominant: ten frames of 99% blank + 1% /ð/ dust must not
    outweigh one frame of real /z/ evidence. The v4 exam regressed to chance
    exactly because the old per-frame renormalization gave every blank
    frame's dust a full vote; mass weighting fixes it."""
    probs = np.zeros((11, 6))
    probs[:10, 0] = 0.99  # blank
    probs[:10, 1] = 0.01  # ð dust (map: ð=1)
    probs[10, 0] = 0.009  # blank
    probs[10, 1] = 0.001  # ð dust
    probs[10, 2] = 0.5  # z (map: z=2)
    probs[10, 3] = 0.49  # d (map: d=3)
    raw = np.log(probs).astype(np.float32)
    result = score_contrast(raw, np.arange(11), "ð", ["z", "d", "v"], phone_map())
    assert result.posteriors["z"] > result.posteriors["ð"]
    assert result.posteriors["z"] == pytest.approx(0.5 / 1.091, abs=1e-3)
    assert result.posteriors["ð"] == pytest.approx(0.101 / 1.091, abs=1e-3)
    assert result.best_confusion == "z"


def test_non_target_mass_is_ignored():
    """Posterior mass on tokens outside {target} + confusions must not leak in."""
    probs = np.full((4, 6), 1e-6)
    probs[:, 1] = 0.1  # ð
    probs[:, 2] = 0.1  # z
    probs[:, 5] = 0.8  # some unrelated phone
    raw = np.log(probs).astype(np.float32)
    result = score_contrast(raw, np.arange(4), "ð", ["z", "d", "v"], phone_map())
    assert sum(result.posteriors.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.posteriors["ð"] == pytest.approx(0.5, abs=1e-4)
    assert result.posteriors["d"] == pytest.approx(0.0, abs=1e-4)


def test_empty_confusion_set_is_rejected():
    raw = log_probs_for([1])
    with pytest.raises(ValueError, match="confusion"):
        score_contrast(raw, [0], "ð", [], phone_map())


def test_empty_frames_are_rejected():
    raw = log_probs_for([1])
    with pytest.raises(ValueError, match="no label frames"):
        score_contrast(raw, np.array([], dtype=np.int64), "ð", ["z"], phone_map())


def test_out_of_range_frames_are_rejected():
    raw = log_probs_for([1])
    with pytest.raises(ValueError, match="outside"):
        score_contrast(raw, [0, 99], "ð", ["z"], phone_map())


def calibration(threshold: float = 0.85) -> ContrastCalibration:
    # a=1, b=0: p = sigmoid(score); threshold 0.85 => |score| > ~1.73 to judge.
    return ContrastCalibration(
        target="ð",
        confusions=["z", "d", "v"],
        substitution_platt=PlattCalibration(a=1.0, b=0.0),
        threshold=threshold,
    )


def raw(score: float) -> ContrastScore:
    return ContrastScore(
        target="ð",
        confusions=("z", "d", "v"),
        posteriors={"ð": 0.5, "z": 0.5, "d": 0.0, "v": 0.0},
        score=score,
        spike_score=score,
        vote_fraction=0.5,
        best_confusion="z",
    )


def test_decide_flags_a_clear_substitution():
    verdict, confidence, detected = decide(raw(3.0), calibration())
    assert verdict == "substituted"
    assert confidence > 0.9
    assert detected == "z"


def test_decide_passes_a_clear_target():
    verdict, confidence, detected = decide(raw(-3.0), calibration())
    assert verdict == "on_target"
    assert confidence < 0.1
    assert detected is None


def test_decide_refuses_the_middle_band():
    verdict, confidence, detected = decide(raw(0.0), calibration())
    assert verdict == "uncertain"
    assert detected is None
    assert 0.4 < confidence < 0.6


def test_confidence_is_the_calibrated_probability():
    """The number the composer gates on is P(substituted), not the raw margin."""
    _, confidence, _ = decide(raw(1.0), calibration())
    assert confidence == pytest.approx(1 / (1 + np.exp(-1.0)), abs=1e-6)


def test_spike_score_survives_what_the_mean_washes_out():
    """The CTC peakiness case: a substitution that wins one frame in five is
    invisible to the mean but obvious to the spike frame."""
    probs = np.full((5, 6), 1e-6)
    probs[:, 1] = 0.98  # ð dominates four frames...
    probs[:, 2] = 0.01
    probs[0, 1] = 0.05  # ...but /z/ wins one frame
    probs[0, 2] = 0.9
    raw_log = np.log(probs).astype(np.float32)
    result = score_contrast(raw_log, np.arange(5), "ð", ["z", "d", "v"], phone_map())
    assert result.score < 0, "the mean says on-target"
    assert result.spike_score > 1.0, "the spike frame saw the substitution"
    assert result.vote_fraction == pytest.approx(0.2, abs=1e-6)


def test_vote_fraction_counts_confusion_winning_frames():
    probs = np.full((4, 6), 1e-6)
    probs[:, 1] = 0.9  # ð
    probs[:, 2] = 0.05  # z
    probs[2:, 2] = 0.6
    probs[2:, 1] = 0.3
    raw_log = np.log(probs).astype(np.float32)
    result = score_contrast(raw_log, np.arange(4), "ð", ["z", "d", "v"], phone_map())
    assert result.vote_fraction == pytest.approx(0.5, abs=1e-6)


def test_decide_uses_the_single_calibrated_threshold():
    """One threshold for every learner: the judge is blind to who is
    speaking. The same evidence yields the same verdict no matter who said
    it - there is no per-language input anywhere in the decision."""
    # p_sub = sigmoid(2.0) ~ 0.88: above the 0.85 threshold.
    verdict, confidence, detected = decide(raw(2.0), calibration())
    assert verdict == "substituted"
    assert confidence == pytest.approx(0.8808, abs=1e-3)
    assert detected == "z"
    # p_sub = sigmoid(0.8) ~ 0.69: inside the band -> refuse to judge.
    assert decide(raw(0.8), calibration())[0] == "uncertain"
    # p_sub = sigmoid(-2.2) ~ 0.10: below the band.
    assert decide(raw(-2.2), calibration())[0] == "on_target"


def test_decide_uses_the_variant_the_calibration_names():
    """A calibration fitted on spike scores must read spike scores, not the
    interval mean: mixing aggregations silently corrupts every verdict."""
    spike_calibration = calibration().model_copy(
        update={"score_variant": "spike", "threshold": 0.85}
    )
    mixed = raw(-3.0).__class__(  # mean says on-target...
        target="ð",
        confusions=("z", "d", "v"),
        posteriors={"ð": 0.9, "z": 0.1, "d": 0.0, "v": 0.0},
        score=-3.0,
        spike_score=3.0,  # ...but the spike frame says substitution
        vote_fraction=0.0,
        best_confusion="z",
    )
    verdict, _, detected = decide(mixed, spike_calibration)
    assert verdict == "substituted"
    assert detected == "z"
