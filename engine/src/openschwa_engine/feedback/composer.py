"""Turns evidence into the confidence-gated feedback[] list.

This module is the enforcement point for the project's central invariant: *no
feedback beats wrong feedback*. Nothing reaches a learner unless a feedback type
has cleared the shipping bar in eval/README.md - precision >= 0.90 at useful
recall on held-out data, no L1 group below ~0.8 - and the operating threshold
lives in the committed calibration file, produced only by the eval harness.

M1 ships three kinds:
- retry: an explicit refusal to judge (M0 behaviour, unchanged);
- segmental_substitution: "that /dh/ sounded more like /z/", anchored to the
  focus phone's interval, emitted only when the calibrated confidence passes
  the gate (the judge line - parked as research by the mirror pivot);
- phone_hearing: the mirror. "I heard /dh/", "I heard /z/ (you were going for
  /dh/)", or the honest "couldn't tell" - the ear reports what it heard at
  the focus slot instead of judging the learner, gated by the committed
  hearing block (P(heard == realized)). include_ungated=true (dev/eval) also
  reveals the items the judge gate would withhold.
"""

from openschwa_engine.alignment import AlignmentOutcome
from openschwa_engine.schemas.analysis import Anchor, ContrastResult, FeedbackItem, Prosody
from openschwa_engine.scoring import Calibration

#: The tone verdict's operating confidence, calibrated on the controlled
#: recordings exam (eval/reports-recordings/): >= 0.3 shows 93.2% correct
#: verdicts at 97.8% coverage; below it the honest couldn't-tell refusal.
TONE_CONFIDENCE_GATE = 0.3

TONE_NAMES = {
    "fall": "a falling tone",
    "rise": "a rising tone",
    "fall_rise": "a falling-then-rising tone",
    "level": "a flat tone",
}

RETRY_MESSAGES = {
    "no speech detected in the recording": "I couldn't hear any speech — try recording again.",
    "clipping": (
        "That recording was too loud and distorted. Move back from the mic and try again."
    ),
    # Reached only when the input is dead, so it points at the device rather
    # than telling someone to speak up — advice that cannot help when the
    # capture chain is delivering nothing at all.
    "no signal from the microphone": (
        "I'm getting no signal from the microphone. Check that the right input "
        "device is selected and not muted."
    ),
}
DEFAULT_RETRY_MESSAGE = "I couldn't line that up with the target — give it another go."


def retry_item(outcome: AlignmentOutcome) -> FeedbackItem:
    """The refusal to judge: reports the engine's own state, never a claim
    about the learner's pronunciation."""
    message = RETRY_MESSAGES.get(outcome.reason or "", DEFAULT_RETRY_MESSAGE)
    return FeedbackItem(
        id="retry",
        kind="retry",
        severity="warning",
        # A refusal is always shown: it reports the engine's own state rather
        # than making a claim about the learner's pronunciation.
        confidence=1.0,
        message_key="feedback.retry",
        message=message,
        anchor=Anchor(),
        evidence={},
    )


def _substitution_item(
    contrast: ContrastResult,
    contrast_index: int,
    interval_s: tuple[float, float] | None,
) -> FeedbackItem:
    if contrast.verdict == "uncertain":
        message = (
            f"/{contrast.target}/ here was hard to pin down — try it once more, a little slower."
        )
    else:
        message = f"That /{contrast.target}/ sounded more like /{contrast.detected}/."
    return FeedbackItem(
        id=f"substitution-{contrast_index}",
        kind="segmental_substitution",
        severity="warning",
        confidence=contrast.confidence,
        message_key="feedback.segmental_substitution",
        message=message,
        anchor=Anchor(phone_index=contrast.phone_index, interval_s=interval_s),
        evidence={"contrast_index": contrast_index},
    )


def _hearing_item(
    contrast: ContrastResult,
    contrast_index: int,
    interval_s: tuple[float, float] | None,
    hearing_probability: float | None,
    hearing_threshold: float | None,
) -> FeedbackItem | None:
    """The mirror item for one focus slot: what the ear heard, or honest doubt.

    Requires a committed hearing calibration (P(heard == realized) + operating
    threshold). Three states: heard-as-intended (praise), heard-other (warning),
    couldn't-tell (refusal - a claim the model declines to make, never a
    withheld judgment).
    """
    if hearing_probability is None or hearing_threshold is None or contrast.heard is None:
        return None
    anchor = Anchor(phone_index=contrast.phone_index, interval_s=interval_s)
    evidence = {"contrast_index": contrast_index}
    if hearing_probability < hearing_threshold:
        return FeedbackItem(
            id=f"hearing-{contrast_index}",
            kind="phone_hearing",
            severity="warning",
            confidence=round(hearing_probability, 4),
            message_key="feedback.phone_hearing_unsure",
            message=(
                f"I couldn't tell what that /{contrast.target}/ sounded like — "
                "try once more, a little slower."
            ),
            anchor=anchor,
            evidence=evidence,
        )
    if contrast.heard == contrast.target:
        return FeedbackItem(
            id=f"hearing-{contrast_index}",
            kind="phone_hearing",
            severity="praise",
            confidence=round(hearing_probability, 4),
            message_key="feedback.phone_hearing_on_target",
            message=f"I heard /{contrast.heard}/ — right on target.",
            anchor=anchor,
            evidence=evidence,
        )
    # Open-set ears fold every non-drilled realization into "other", which
    # must never surface as a literal "/other/" to a learner.
    heard_label = "something else" if contrast.heard == "other" else f"/{contrast.heard}/"
    return FeedbackItem(
        id=f"hearing-{contrast_index}",
        kind="phone_hearing",
        severity="warning",
        confidence=round(hearing_probability, 4),
        message_key="feedback.phone_hearing_other",
        message=f"I heard {heard_label} where you were going for /{contrast.target}/.",
        anchor=anchor,
        evidence=evidence,
    )


def intonation_item(prosody: "Prosody | None") -> FeedbackItem | None:
    """The M2 tone verdict, or None when there is nothing to say.

    Pure DSP - no alignment involved - so it ships even when the phone
    alignment failed, and only for exercises that declare an expected tone.
    Below the confidence gate the honest couldn't-tell refusal ships: a
    wrong tone verdict is worse than none.
    """
    if prosody is None or prosody.nuclear_tone is None:
        return None
    verdict = prosody.nuclear_tone
    expected = verdict.expected
    if expected is None:
        return None
    confidence = round(verdict.confidence, 4)
    if confidence < TONE_CONFIDENCE_GATE:
        return FeedbackItem(
            id="tone",
            kind="intonation_tone",
            severity="warning",
            confidence=confidence,
            message_key="feedback.tone_uncertain",
            message="I couldn't hear the melody clearly enough - try that one again.",
        )
    if verdict.match:
        return FeedbackItem(
            id="tone",
            kind="intonation_tone",
            severity="praise",
            confidence=confidence,
            message_key="feedback.tone_match",
            message=(
                f"That ended with {TONE_NAMES.get(expected, expected)} "
                "- right on target."
            ),
        )
    return FeedbackItem(
        id="tone",
        kind="intonation_tone",
        severity="warning",
        confidence=confidence,
        message_key="feedback.tone_mismatch",
        message=(
            f"The target is {TONE_NAMES.get(expected, expected)}, but that ended "
            f"with {TONE_NAMES.get(verdict.detected, verdict.detected)}."
        ),
    )


def compose(
    outcome: AlignmentOutcome,
    contrasts: "list[ContrastResult] | tuple[ContrastResult, ...]" = (),
    *,
    include_ungated: bool = False,
    calibration: "Calibration | None" = None,
    prosody: "Prosody | None" = None,
) -> list[FeedbackItem]:
    """Gate-passing feedback for one analysis.

    alignment.status != "ok" short-circuits to a single retry item. On a
    successful alignment each contrast contributes an item only when the gate
    passed it: verdict substituted at or above the calibrated operating
    threshold (which is exactly how a substituted verdict is decided).
    Uncertain verdicts ship only under include_ungated, for the eval harness.

    The mirror's phone_hearing items ride the same contrasts: the heard phone
    is reported above the hearing threshold, and below it the honest
    couldn't-tell refusal ships (it is not a judgment, so it needs no gate).
    """
    # The tone verdict is pure DSP: it ships even when the phone alignment
    # failed (no model installed, noisy take, ...) - the melody of the take
    # and the phone lineup are independent facts.
    tone = intonation_item(prosody)
    if not outcome.ok:
        items = [retry_item(outcome)]
        if tone is not None:
            items.append(tone)
        return items

    items: list[FeedbackItem] = []
    for index, contrast in enumerate(contrasts):
        if contrast.verdict not in ("substituted", "uncertain"):
            continue
        if contrast.verdict == "uncertain" and not include_ungated:
            continue
        # A verdict can only exist with a committed calibration for this
        # contrast; re-checking keeps the gate in one place.
        contrast_calibration = (
            calibration.contrast(contrast.target) if calibration is not None else None
        )
        if contrast_calibration is None:
            continue

        phone = next((p for p in outcome.phones if p.index == contrast.phone_index), None)
        interval = (phone.start_s, phone.end_s) if phone is not None else None
        items.append(_substitution_item(contrast, index, interval))

    # The mirror: every contrast with a hearing calibration contributes its
    # three-state item. Emitted after the judge items so a rich client sees
    # the mirror line last (the shipped M1 feedback).
    for index, contrast in enumerate(contrasts):
        contrast_calibration = (
            calibration.contrast(contrast.target) if calibration is not None else None
        )
        if contrast_calibration is None or contrast.heard is None:
            continue
        phone = next((p for p in outcome.phones if p.index == contrast.phone_index), None)
        interval = (phone.start_s, phone.end_s) if phone is not None else None
        item = _hearing_item(
            contrast,
            index,
            interval,
            (
                contrast_calibration.hearing_probability(contrast.hearing_score)
                if contrast.hearing_score is not None
                else None
            ),
            contrast_calibration.hearing_threshold,
        )
        if item is not None:
            items.append(item)

    # M2: the tone verdict comes last - for an intonation exercise it is the
    # headline, and for a segmental exercise it never fires (no expected tone).
    if tone is not None:
        items.append(tone)

    return items
