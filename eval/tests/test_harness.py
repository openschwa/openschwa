"""Harness metrics and calibration maths, on synthetic records (no audio)."""

import math

import pytest

from openschwa_eval.harness import (
    TokenRecord,
    _metrics,
    build_calibration,
    fit_platt,
    pick_threshold,
    raw_score_from_posteriors,
    sigmoid,
)


def record(
    score=None, label="correct", l1="arabic", corpus="l2arctic", split="test", wall_ms=10.0
) -> TokenRecord:
    return TokenRecord(
        utterance_id=f"u-{id(score)}-{label}",
        token_index=0,
        corpus=corpus,
        l1=l1,
        split=split,
        label=label,
        substituted_with=None,
        audio_path="x.wav",
        start_s=None,
        end_s=None,
        alignment="ok",
        reason=None,
        audio_problem=False,
        score=score,
        spike_score=score,
        vote_fraction=0.5 if score is not None else None,
        best_confusion="z" if score else None,
        gop=-0.5 if score is not None else None,
        verdict=None,
        confidence=None,
        alignment_confidence=0.9,
        wall_ms=wall_ms,
    )


def test_raw_score_matches_the_log_ratio():
    score, best = raw_score_from_posteriors({"ð": 0.2, "z": 0.7, "d": 0.1}, "ð")
    assert best == "z"
    assert score == pytest.approx(math.log(0.7 / 0.2), abs=1e-6)


def test_perfect_separation_scores_perfect_metrics():
    a, b = 1.0, 0.0  # p = sigmoid(score)
    records = [
        record(score=3.0, label="substituted"),
        record(score=-3.0, label="correct"),
        record(score=3.0, label="substituted"),
        record(score=-3.0, label="correct"),
    ]
    m = _metrics(records, a, b, threshold=0.85)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_refusals_count_against_recall():
    a, b = 1.0, 0.0
    records = [
        record(score=3.0, label="substituted"),
        record(score=None, label="substituted"),  # refused -> missed
        record(score=-3.0, label="correct"),
    ]
    m = _metrics(records, a, b, threshold=0.85)
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    assert m["refused"] == 1


def test_platt_fit_recovers_a_known_sigmoid():
    scores = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
    labels = [False, False, False, True, True, True]
    a, b = fit_platt(scores, labels)
    for score, label in zip(scores, labels, strict=True):
        p = sigmoid(a * score + b)
        assert (p >= 0.5) == label


def test_threshold_sweep_is_precision_first():
    # One correct token sits at p=0.93; two errors at 0.99+. A recall-greedy
    # threshold would flag the correct token; precision-first must not.
    a, b = 1.0, 0.0
    records = [
        record(score=2.4, label="correct"),  # p ~ 0.917
        record(score=3.0, label="substituted"),  # p ~ 0.953
        record(score=4.0, label="substituted"),  # p ~ 0.982
    ]
    threshold, train_metrics, status = pick_threshold(records, a, b)
    assert status == "ok"
    assert train_metrics["precision"] >= 0.90
    # The chosen threshold must not flag the 0.917 correct token.
    assert threshold > sigmoid(2.4)
    assert train_metrics["recall"] == 1.0


def test_l1_floor_blocks_a_threshold_that_fails_one_group():
    a, b = 1.0, 0.0
    records = [
        # arabic: perfect
        record(score=3.0, label="substituted", l1="arabic"),
        record(score=3.0, label="substituted", l1="arabic"),
        record(score=3.0, label="substituted", l1="arabic"),
        record(score=3.0, label="substituted", l1="arabic"),
        record(score=3.0, label="substituted", l1="arabic"),
        # korean: five positives, but a false positive would leak in at low T
        record(score=2.2, label="substituted", l1="korean"),  # p ~ 0.90
        record(score=2.2, label="substituted", l1="korean"),
        record(score=2.2, label="substituted", l1="korean"),
        record(score=2.2, label="substituted", l1="korean"),
        record(score=2.2, label="substituted", l1="korean"),
        record(score=1.4, label="correct", l1="korean"),  # p ~ 0.80, flags below T=0.81
    ]
    threshold, train_metrics, status = pick_threshold(records, a, b)
    # At any threshold below ~0.803 the korean group precision drops to 5/6 < 0.8.
    assert threshold >= 0.803
    assert status == "ok"


def test_shipping_bar_unmet_is_reported():
    a, b = 1.0, 0.0
    records = [
        record(score=1.5, label="correct"),
        record(score=1.0, label="substituted"),
    ]
    _, _, status = pick_threshold(records, a, b)
    assert status == "SHIPPING BAR NOT MET"


def test_build_calibration_matches_the_engine_model():
    from openschwa_engine.config import Settings
    from openschwa_engine.scoring.calibration import Calibration

    content = build_calibration(
        "wav2vec2-espeak-cv-ft",
        "ð",
        ["z", "d", "v"],
        (1.5, -0.2),
        0.85,
        (2.0, 1.0),
        "eval/reports/test.json",
        Settings(),
    )
    calibration = Calibration.model_validate(content)
    contrast = calibration.contrast("ð")
    assert contrast is not None
    assert contrast.threshold == 0.85
    assert contrast.substitution_platt.a == 1.5
    assert contrast.gop_platt is not None
