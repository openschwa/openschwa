"""Committed calibration loading: strict on read, silent on failure."""

import json
import subprocess
from pathlib import Path

import pytest

from openschwa_engine.scoring.calibration import (
    CALIBRATION_PATH,
    Calibration,
    load_calibration,
    load_or_fail,
)

#: Repo root (engine/tests/test_calibration.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]
#: Mirrors eval/src/openschwa_eval/harness.py's MIN_COMMIT_TRAIN; importing
#: the eval package here would reverse the dependency direction.
MIN_COMMIT_TRAIN = 200

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


def test_per_l1_thresholds_are_rejected(tmp_path):
    """The shipped calibration must be blind to the learner's language: a
    file carrying per-L1 operating points is invalid and the engine refuses
    it loudly (no feedback beats wrong feedback)."""
    path = tmp_path / "calibration.yaml"
    path.write_text(
        VALID.replace(
            "threshold: 0.85",
            "threshold: 0.85\n    l1_thresholds:\n      mandarin: 0.92\n      arabic: 0.80",
        ),
        encoding="utf-8",
    )
    assert load_calibration(path) is None


def test_committed_calibration_traceable_to_passing_report():
    """A committed calibration.yaml must trace, value-for-value, to a
    committed exam report whose own status is 'ok'.

    This is the tripwire for hand-forged calibrations (2026-08-20 incident:
    a file with invented Platt constants and threshold 0.5 - which removes
    the uncertain band entirely - cited a SHIPPING BAR NOT MET report as its
    provenance and validated cleanly). A forger must now also forge a full
    passing report, which is committed and reviewable. The file is expected
    to be absent in CI: the only sanctioned writer (run_eval.py --commit)
    runs on the exam laptop, and the file is committed together with its
    report in the same commit.
    """
    if not CALIBRATION_PATH.is_file():
        pytest.skip("no committed calibration - nothing to trace")
    calibration = load_or_fail(CALIBRATION_PATH)
    generated_by = calibration.generated_by
    report_path = (REPO_ROOT / generated_by).resolve()
    assert report_path.is_file(), f"generated_by {generated_by} does not exist"
    # Committed content: an untracked report cannot be provenance.
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", generated_by],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert tracked.returncode == 0, f"generated_by {generated_by} is not committed content"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "ok", f"report status {report['status']} is not 'ok'"
    assert report["model_id"] == calibration.model_id, "model_id mismatch"
    assert report["tokens"]["train"] >= MIN_COMMIT_TRAIN, "report below the train-token floor"
    # Every contrast must exist in the report and match value-for-value.
    report_targets = {report["target"]}
    yaml_targets = {contrast.target for contrast in calibration.contrasts}
    assert yaml_targets == report_targets, (
        f"contrast set {yaml_targets} does not match the report's {report_targets}"
    )
    for contrast in calibration.contrasts:
        assert contrast.substitution_platt.a == pytest.approx(report["platt"]["a"])
        assert contrast.substitution_platt.b == pytest.approx(report["platt"]["b"])
        assert contrast.threshold == pytest.approx(report["threshold"])
