"""Harness metrics and calibration maths, on synthetic records (no audio)."""

import math

import pytest

from openschwa_eval.harness import (
    TokenRecord,
    _final_status,
    _metrics,
    _per_l1_audit,
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


def test_pooled_precision_survives_a_mixed_l1_train_set():
    a, b = 1.0, 0.0
    records = [
        # arabic: perfect
        record(score=3.0, label="substituted", l1="arabic"),
        record(score=3.0, label="substituted", l1="arabic"),
        record(score=3.0, label="substituted", l1="arabic"),
        record(score=3.0, label="substituted", l1="arabic"),
        record(score=3.0, label="substituted", l1="arabic"),
        # korean: five positives at p ~ 0.90, one correct at p ~ 0.80
        record(score=2.2, label="substituted", l1="korean"),
        record(score=2.2, label="substituted", l1="korean"),
        record(score=2.2, label="substituted", l1="korean"),
        record(score=2.2, label="substituted", l1="korean"),
        record(score=2.2, label="substituted", l1="korean"),
        record(score=1.4, label="correct", l1="korean"),
    ]
    threshold, train_metrics, status = pick_threshold(records, a, b)
    # One pooled line: below sigmoid(2.2) ~ 0.9002 every positive is flagged,
    # precision 10/11 >= 0.90, so the precision-first sweep may stop there.
    assert threshold == pytest.approx(0.9)
    assert train_metrics["precision"] >= 0.90
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


def test_per_l1_audit_scores_every_group_at_the_global_point():
    a, b = 1.0, 0.0
    test = _group("mandarin", [(3.0, "substituted")] * 5 + [(-3.0, "correct")] * 5)
    test += _group("arabic", [(1.1, "substituted"), (-3.0, "correct")])  # p ~ 0.75
    audit = _per_l1_audit(test, a, b, 0.9, "mean")
    assert audit["mandarin"]["precision"] == 1.0
    assert audit["mandarin"]["recall"] == 1.0
    assert audit["mandarin"]["ok"] is True
    # The SAME line misses the arabic error (recall 0.0). That is exactly
    # what the audit is for: it names the group the blind line underserves,
    # without ever gating the shipped calibration on it.
    assert audit["arabic"]["recall"] == 0.0
    assert audit["arabic"]["ok"] is False


def test_per_l1_audit_small_group_uses_the_precision_floor():
    a, b = 1.0, 0.0
    test = _group("korean", [(3.0, "substituted"), (-3.0, "correct")])
    audit = _per_l1_audit(test, a, b, 0.9, "mean")
    assert audit["korean"]["ok"] is True  # 1 positive -> precision floor, met
    test += _group("korean", [(2.3, "correct")])  # p ~ 0.91: flags, precision 0.5
    audit = _per_l1_audit(test, a, b, 0.9, "mean")
    assert audit["korean"]["ok"] is False


def test_final_status_requires_the_pooled_bar():
    ok = {"precision": 0.92, "recall": 0.45}
    assert _final_status("ok", ok) == "ok"
    assert _final_status("ok", {"precision": 0.92, "recall": 0.3}) == "pooled-bar-not-met"
    assert _final_status("ok", {"precision": 0.8, "recall": 0.6}) == "pooled-bar-not-met"
    assert _final_status("SHIPPING BAR NOT MET", ok) == "SHIPPING BAR NOT MET"


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
    # Accent-blind by construction: no per-language keys anywhere.
    assert "l1" not in content["contrasts"][0]
