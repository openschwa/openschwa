"""Committed calibration loading: strict on read, silent on failure."""

import pytest

from openschwa_engine.scoring.calibration import Calibration, load_calibration, load_or_fail

VALID = """schema_version: "1.0"
generated_by: eval/reports/m1-2026-08-18.json
model_id: wav2vec2-espeak-cv-ft
contrasts:
  - target: "ð"
    confusions: ["z", "d", "v"]
    substitution_platt: {a: 1.5, b: -0.2}
    threshold: 0.85
    gop_platt: {a: 2.0, b: 1.0}
alignment:
  min_confidence: 0.30
  low_confidence: 0.55
"""


def test_loads_a_valid_file(tmp_path):
    path = tmp_path / "calibration.yaml"
    path.write_text(VALID, encoding="utf-8")
    calibration = load_calibration(path)
    assert calibration is not None
    contrast = calibration.contrast("ð")
    assert contrast is not None
    assert contrast.threshold == 0.85
    probability = contrast.substitution_platt.probability(0.0)
    assert probability == pytest.approx(1 / (1 + 2.71828**0.2), abs=0.02)
    assert calibration.alignment.low_confidence == 0.55


def test_missing_file_degrades_to_none(tmp_path):
    assert load_calibration(tmp_path / "nope.yaml") is None


def test_corrupt_yaml_degrades_to_none(tmp_path):
    path = tmp_path / "calibration.yaml"
    path.write_text("schema_version: ['unterminated", encoding="utf-8")
    assert load_calibration(path) is None


def test_unknown_fields_are_rejected(tmp_path):
    path = tmp_path / "calibration.yaml"
    sneaky = VALID.replace("threshold: 0.85", "threshold: 0.85\n  sneaky: true")
    path.write_text(sneaky, encoding="utf-8")
    assert load_calibration(path) is None


def test_threshold_out_of_range_is_rejected(tmp_path):
    path = tmp_path / "calibration.yaml"
    path.write_text(VALID.replace("threshold: 0.85", "threshold: 0.40"), encoding="utf-8")
    assert load_calibration(path) is None


def test_contrast_lookup_misses_cleanly(tmp_path):
    path = tmp_path / "calibration.yaml"
    path.write_text(VALID, encoding="utf-8")
    calibration = load_calibration(path)
    assert calibration is not None
    assert calibration.contrast("z") is None


def test_load_or_fail_raises_on_a_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="missing or unusable"):
        load_or_fail(tmp_path / "nope.yaml")


def test_round_trips_through_pydantic(tmp_path):
    path = tmp_path / "calibration.yaml"
    path.write_text(VALID, encoding="utf-8")
    calibration = load_calibration(path)
    assert calibration is not None
    reparsed = Calibration.model_validate(calibration.model_dump())
    assert reparsed == calibration


def test_per_l1_thresholds_fall_back_to_the_global(tmp_path):
    path = tmp_path / "calibration.yaml"
    path.write_text(
        VALID.replace(
            "threshold: 0.85",
            "threshold: 0.85\n    l1_thresholds:\n      mandarin: 0.92\n      arabic: 0.80",
        ),
        encoding="utf-8",
    )
    calibration = load_calibration(path)
    contrast = calibration.contrast("ð")
    assert contrast is not None
    assert contrast.threshold_for("mandarin") == 0.92
    assert contrast.threshold_for("arabic") == 0.80
    assert contrast.threshold_for("korean") == 0.85  # unknown L1 -> global
    assert contrast.threshold_for(None) == 0.85
