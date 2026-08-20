"""The confidence gate: what may and may not reach a learner."""

from openschwa_engine.alignment import AlignedPhone, AlignmentOutcome
from openschwa_engine.feedback.composer import compose, retry_item
from openschwa_engine.schemas.analysis import ContrastResult, F0Track, NuclearTone, Prosody
from openschwa_engine.scoring.calibration import (
    AlignmentCalibration,
    Calibration,
    ContrastCalibration,
    PlattCalibration,
)


def calibration() -> Calibration:
    return Calibration(
        schema_version="1.0",
        generated_by="eval/reports/test.json",
        model_id="synthetic",
        contrasts=[
            ContrastCalibration(
                target="ð",
                confusions=["z", "d", "v"],
                substitution_platt=PlattCalibration(a=1.0, b=0.0),
                threshold=0.85,
            )
        ],
        alignment=AlignmentCalibration(min_confidence=0.30, low_confidence=0.55),
    )


def outcome(status: str = "ok") -> AlignmentOutcome:
    if status != "ok":
        return AlignmentOutcome(status, 0.0, reason="test refusal")
    phones = (
        AlignedPhone(index=0, label="ð", start_s=0.1, end_s=0.2, gop=-0.5, confidence=0.9),
        AlignedPhone(index=1, label="ɪ", start_s=0.2, end_s=0.3, gop=-0.1, confidence=0.9),
    )
    return AlignmentOutcome("ok", 0.9, phones=phones)


def contrast(verdict: str, confidence: float = 0.95) -> ContrastResult:
    return ContrastResult(
        phone_index=0,
        target="ð",
        confusion_set=["z", "d", "v"],
        posteriors={"ð": 0.05, "z": 0.8, "d": 0.1, "v": 0.05},
        verdict=verdict,  # type: ignore[arg-type]
        detected="z" if verdict == "substituted" else None,
        confidence=confidence,
    )


def test_a_failed_alignment_is_always_a_single_retry():
    items = compose(outcome("failed"), calibration=calibration())
    assert [item.kind for item in items] == ["retry"]
    assert items[0].message_key == "feedback.retry"


def test_retry_never_names_a_specific_problem_it_does_not_know():
    item = retry_item(AlignmentOutcome("failed", 0.0, reason=None))
    assert "give it another go" in item.message


def test_a_clear_substitution_passes_the_gate():
    items = compose(outcome(), [contrast("substituted")], calibration=calibration())
    assert len(items) == 1
    item = items[0]
    assert item.kind == "segmental_substitution"
    assert "/z/" in item.message and "/ð/" in item.message
    assert item.anchor is not None
    assert item.anchor.phone_index == 0
    assert item.anchor.interval_s == (0.1, 0.2)
    assert item.evidence["contrast_index"] == 0


def test_on_target_yields_no_feedback():
    """Silence beats wrong praise: an on-target verdict has not cleared the bar."""
    assert (
        compose(outcome(), [contrast("on_target", confidence=0.1)], calibration=calibration()) == []
    )


def test_uncertain_yields_no_feedback_by_default():
    assert (
        compose(outcome(), [contrast("uncertain", confidence=0.5)], calibration=calibration()) == []
    )


def test_include_ungated_reveals_uncertain_verdicts():
    items = compose(
        outcome(),
        [contrast("uncertain", confidence=0.5)],
        include_ungated=True,
        calibration=calibration(),
    )
    assert [item.kind for item in items] == ["segmental_substitution"]
    assert "hard to pin down" in items[0].message


def test_no_calibration_means_no_verdicts():
    """Even a substituted verdict is withheld when the calibration is missing:
    the engine refuses to judge without committed thresholds."""
    assert compose(outcome(), [contrast("substituted")], calibration=None) == []


def test_contrast_without_calibration_entry_is_withheld():
    empty = Calibration(
        schema_version="1.0",
        generated_by="eval/reports/test.json",
        model_id="synthetic",
        contrasts=[],
        alignment=AlignmentCalibration(min_confidence=0.30, low_confidence=0.55),
    )
    assert compose(outcome(), [contrast("substituted")], calibration=empty) == []


def test_evidence_points_at_the_contrast_position_not_the_item_position():
    """The evidence pointer names the position in contrasts[], which is what a
    rich client indexes with - not this item's position in the feedback list."""
    contrasts = [
        contrast("on_target", confidence=0.1),
        contrast("substituted"),
    ]
    items = compose(outcome(), contrasts, calibration=calibration())
    assert len(items) == 1
    assert items[0].evidence["contrast_index"] == 1


# -- the mirror (phone_hearing) --------------------------------------------------


def hearing_calibration() -> Calibration:
    """A calibration carrying the mirror's hearing block: sigmoid identity,
    threshold 0.85 - reporting needs P(heard == realized) >= 0.85."""
    return Calibration(
        schema_version="1.0",
        generated_by="eval/reports/test.json",
        model_id="synthetic",
        contrasts=[
            ContrastCalibration(
                target="ð",
                confusions=["z", "d", "v"],
                substitution_platt=PlattCalibration(a=1.0, b=0.0),
                threshold=0.85,
                hearing_platt=PlattCalibration(a=1.0, b=0.0),
                hearing_threshold=0.85,
            )
        ],
        alignment=AlignmentCalibration(min_confidence=0.30, low_confidence=0.55),
    )


def hearing_contrast(heard: str, hearing_score: float) -> ContrastResult:
    result = contrast("uncertain", confidence=0.5)
    return result.model_copy(update={"heard": heard, "hearing_score": hearing_score})


def test_mirror_reports_heard_other_above_threshold():
    items = compose(
        outcome(),
        [hearing_contrast("z", 2.0)],  # P(heard==realized) ~ 0.88 >= 0.85
        calibration=hearing_calibration(),
    )
    mirror = [item for item in items if item.kind == "phone_hearing"]
    assert len(mirror) == 1
    assert mirror[0].severity == "warning"
    assert "/z/" in mirror[0].message and "/ð/" in mirror[0].message
    assert mirror[0].message_key == "feedback.phone_hearing_other"
    assert mirror[0].anchor is not None
    assert mirror[0].anchor.phone_index == 0
    assert mirror[0].evidence["contrast_index"] == 0


def test_mirror_praises_heard_as_intended_above_threshold():
    items = compose(
        outcome(),
        [hearing_contrast("ð", 2.0)],
        calibration=hearing_calibration(),
    )
    mirror = [item for item in items if item.kind == "phone_hearing"]
    assert len(mirror) == 1
    assert mirror[0].severity == "praise"
    assert mirror[0].message_key == "feedback.phone_hearing_on_target"
    assert "right on target" in mirror[0].message


def test_mirror_refuses_with_couldnt_tell_below_threshold():
    """The honest refusal: a hearing claim below the calibrated gate becomes
    'couldn't tell' - it ships even without include_ungated because it makes
    no claim about the learner's pronunciation."""
    items = compose(
        outcome(),
        [hearing_contrast("z", 0.0)],  # P ~ 0.5 < 0.85
        calibration=hearing_calibration(),
    )
    mirror = [item for item in items if item.kind == "phone_hearing"]
    assert len(mirror) == 1
    assert mirror[0].message_key == "feedback.phone_hearing_unsure"
    assert "couldn't tell" in mirror[0].message


def test_mirror_stays_silent_without_a_hearing_block():
    """A judge-only calibration (no hearing fit) produces no mirror items:
    the mirror ships only after its own exam passes."""
    items = compose(
        outcome(),
        [hearing_contrast("z", 2.0)],
        calibration=calibration(),
    )
    assert [item.kind for item in items if item.kind == "phone_hearing"] == []


def test_mirror_stays_silent_without_heard():
    items = compose(
        outcome(),
        [contrast("uncertain", confidence=0.5)],  # heard=None
        calibration=hearing_calibration(),
    )
    assert [item.kind for item in items if item.kind == "phone_hearing"] == []



def prosody_with_tone(expected, detected, confidence):
    return Prosody(
        f0=F0Track(hop_s=0.01, start_s=0.0, semitones=[0.0, 1.0], median_hz=120.0),
        nuclear_tone=NuclearTone(
            detected=detected,
            expected=expected,
            match=detected == expected,
            confidence=confidence,
        ),
    )


def test_tone_match_is_praise():
    items = compose(outcome(), prosody=prosody_with_tone("fall", "fall", 0.9))
    tone = next(i for i in items if i.kind == "intonation_tone")
    assert tone.severity == "praise"
    assert tone.message_key == "feedback.tone_match"
    assert "falling" in tone.message


def test_tone_mismatch_is_a_warning_naming_both_tones():
    items = compose(outcome(), prosody=prosody_with_tone("fall", "rise", 0.9))
    tone = next(i for i in items if i.kind == "intonation_tone")
    assert tone.severity == "warning"
    assert tone.message_key == "feedback.tone_mismatch"
    assert "falling" in tone.message and "rising" in tone.message


def test_tone_below_gate_is_an_honest_refusal():
    items = compose(outcome(), prosody=prosody_with_tone("fall", "rise", 0.1))
    tone = next(i for i in items if i.kind == "intonation_tone")
    assert tone.message_key == "feedback.tone_uncertain"
    assert "falling" not in tone.message


def test_no_expected_tone_means_no_tone_item():
    items = compose(outcome(), prosody=prosody_with_tone(None, "fall", 0.9))
    assert [i.kind for i in items if i.kind == "intonation_tone"] == []


def test_failed_alignment_still_reports_the_tone():
    items = compose(
        outcome("failed"),
        prosody=prosody_with_tone("fall", "fall", 0.9),
    )
    kinds = [i.kind for i in items]
    assert "retry" in kinds and "intonation_tone" in kinds


def test_without_prosody_there_is_no_tone_item():
    items = compose(outcome())
    assert [i.kind for i in items if i.kind == "intonation_tone"] == []
