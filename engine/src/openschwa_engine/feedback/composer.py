"""Turns evidence into the confidence-gated feedback[] list.

This module is the enforcement point for the project's central invariant: *no
feedback beats wrong feedback*. Nothing reaches a learner unless a feedback type
has cleared the shipping bar in eval/README.md - precision >= 0.90 at useful
recall on held-out data, no L1 group below ~0.8 - and the operating threshold
lives in the committed calibration file, produced only by the eval harness.

M1 ships two kinds:
- retry: an explicit refusal to judge (M0 behaviour, unchanged);
- segmental_substitution: "that /dh/ sounded more like /z/", anchored to the
  focus phone's interval, emitted only when the calibrated confidence passes
  the gate. include_ungated=true (dev/eval) also reveals the items the gate
  would withhold.
"""

from openschwa_engine.alignment import AlignmentOutcome
from openschwa_engine.schemas.analysis import Anchor, ContrastResult, FeedbackItem
from openschwa_engine.scoring import Calibration

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


def compose(
    outcome: AlignmentOutcome,
    contrasts: "list[ContrastResult] | tuple[ContrastResult, ...]" = (),
    *,
    include_ungated: bool = False,
    calibration: "Calibration | None" = None,
) -> list[FeedbackItem]:
    """Gate-passing feedback for one analysis.

    alignment.status != "ok" short-circuits to a single retry item. On a
    successful alignment each contrast contributes an item only when the gate
    passed it: verdict substituted at or above the calibrated operating
    threshold (which is exactly how a substituted verdict is decided).
    Uncertain verdicts ship only under include_ungated, for the eval harness.
    """
    if not outcome.ok:
        return [retry_item(outcome)]

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

    return items
