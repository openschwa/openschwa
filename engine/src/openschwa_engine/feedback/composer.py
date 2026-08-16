"""Turns evidence into the confidence-gated `feedback[]` list.

This module is the enforcement point for the project's central invariant: *no
feedback beats wrong feedback*. Nothing reaches a learner unless a feedback type
has cleared the shipping bar in `eval/README.md` — precision ≥ 0.90 at useful recall on
held-out data, no L1 group below ~0.8, checked by `eval/` and recorded in a
committed calibration file.

**M0 emits exactly one kind: `retry`.** Segmental substitution feedback is
computed and gated in M1, after the bake-off and the eval harness exist. Until
then a successful analysis returns an empty `feedback[]` and the UI shows the
evidence it does have — the phone timeline and the pitch contour. That is the
correct behaviour, not a gap: the engine has no calibrated judgement to offer
yet, and inventing one would be exactly the failure mode the invariant exists to
prevent.
"""

from openschwa_engine.alignment import AlignmentOutcome
from openschwa_engine.schemas.analysis import Anchor, FeedbackItem

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
    """The one feedback type M0 ships: an explicit refusal to judge."""
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


def compose(outcome: AlignmentOutcome) -> list[FeedbackItem]:
    """Gate-passing feedback for one analysis.

    `alignment.status != "ok"` short-circuits to a single retry item; a
    successful alignment currently yields nothing, because no judgement type has
    passed the shipping bar yet (see module docstring).
    """
    if not outcome.ok:
        return [retry_item(outcome)]
    return []
