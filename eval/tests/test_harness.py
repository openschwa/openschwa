"""Harness metrics and calibration maths, on synthetic records (no audio)."""

import math

import pytest

from openschwa_eval.harness import (
    L1_MIN_POSITIVES,
    TokenRecord,
    _bar_status,
    _metrics,
    _metrics_per_l1,
    build_calibration,
    fit_l1_thresholds,
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


def _group(l1: str, scores_labels: list[tuple[float | None, str]]) -> list[TokenRecord]:
    return [record(score=s, label=label, l1=l1) for s, label in scores_labels]


def test_fit_l1_thresholds_fits_big_groups_and_skips_small_ones():
    a, b = 1.0, 0.0
    train = _group("arabic", [(3.0, "substituted")] * 6)
    train += _group("korean", [(3.0, "substituted")] * (L1_MIN_POSITIVES - 1))
    thresholds, details = fit_l1_thresholds(train, a, b, "mean")
    assert "arabic" in thresholds
    assert "korean" not in thresholds  # too few positives -> global fallback
    assert details["arabic"]["fitted"] is True
    assert details["korean"]["fitted"] is False
    assert details["korean"]["positives"] == L1_MIN_POSITIVES - 1


def test_metrics_per_l1_flags_each_group_at_its_own_point():
    a, b = 1.0, 0.0
    # Mandarin has a fitted low point (0.6); everyone else uses the strict
    # global point (0.99). The same p ~ 0.75 evidence must flag the Mandarin
    # error and not the Arabic one.
    records = _group("mandarin", [(1.1, "substituted"), (-3.0, "correct")])
    records += _group("arabic", [(1.1, "substituted"), (-3.0, "correct")])
    m = _metrics_per_l1(records, a, b, "mean", {"mandarin": 0.6}, 0.99)
    assert m["tp"] == 1  # only the mandarin error flags
    assert m["fp"] == 0
    assert m["fn"] == 1
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    # A pooled line would have caught neither error:
    pooled = _metrics(records, a, b, threshold=0.99)
    assert pooled["recall"] == 0.0


def test_bar_status_ok_when_every_group_meets_the_bar():
    a, b = 1.0, 0.0
    test = _group("mandarin", [(3.0, "substituted")] * 5 + [(-3.0, "correct")] * 5)
    test += _group("korean", [(-3.0, "correct")])  # no positives -> fine
    status, per_group = _bar_status("ok", test, a, b, "mean", {"mandarin": 0.9}, 0.9)
    assert status == "ok"
    assert per_group["mandarin"]["ok"] is True
    assert per_group["korean"]["ok"] is True


def test_bar_status_blocks_a_group_short_on_recall():
    a, b = 1.0, 0.0
    # Four of five Mandarin errors sit below the fitted point: recall 0.2.
    test = _group(
        "mandarin",
        [(2.0, "substituted")] * 4 + [(3.5, "substituted")] + [(-3.0, "correct")] * 4,
    )
    status, per_group = _bar_status("ok", test, a, b, "mean", {"mandarin": 0.95}, 0.95)
    assert status == "ok-no-l1-floor"
    assert per_group["mandarin"]["ok"] is False


def test_bar_status_passes_the_global_failure_through():
    a, b = 1.0, 0.0
    test = _group("mandarin", [(3.0, "substituted")] * 5)
    status, _ = _bar_status("SHIPPING BAR NOT MET", test, a, b, "mean", {}, 0.9)
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
        l1_thresholds={"arabic": 0.9, "mandarin": 0.8},
    )
    calibration = Calibration.model_validate(content)
    contrast = calibration.contrast("ð")
    assert contrast is not None
    assert contrast.threshold == 0.85
    assert contrast.substitution_platt.a == 1.5
    assert contrast.gop_platt is not None
    assert contrast.l1_thresholds == {"arabic": 0.9, "mandarin": 0.8}
    assert contrast.threshold_for("arabic") == 0.9
    assert contrast.threshold_for("russian") == 0.85  # unknown -> global
