"""Exercise phone sequence + audio -> aligned phones in the upload timeline.

This is where the pieces meet: the phone map turns canonical labels into model
token indices, the acoustic model turns audio into posteriors, `ctc` turns those
into intervals, and the VAD offset puts them back on the timeline the client
recorded.

Status is decided here, and only ever downgraded: bad audio, an unalignable
sequence, or low confidence all produce a non-`ok` status, which the composer
turns into a single "retry". Refusing to judge is a first-class outcome
(docs/architecture.md §1).
"""

import logging
from dataclasses import dataclass
from typing import Literal

from openschwa_engine.alignment import ctc
from openschwa_engine.alignment.acoustic import AcousticModel
from openschwa_engine.audio import PreparedAudio
from openschwa_engine.models.phone_set import PhoneMap, PhoneSetError, normalize

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignedPhone:
    index: int
    label: str
    start_s: float
    end_s: float
    gop: float
    confidence: float


#: `failed` means the recording could not be analysed at all; `low_confidence`
#: means it was aligned but the result is not trustworthy enough to judge.
AlignmentStatus = Literal["ok", "low_confidence", "failed"]


@dataclass(frozen=True)
class AlignmentOutcome:
    status: AlignmentStatus
    confidence: float
    phones: tuple[AlignedPhone, ...] = ()
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def canonical_targets(phone_labels: "tuple[str, ...] | list[str]") -> list[str]:
    """Normalise authored labels, raising PhoneSetError on anything unmappable."""
    return [normalize(p) for p in phone_labels]


def audio_problem(audio: PreparedAudio) -> str | None:
    """A recording-level reason not to analyse, or None.

    Separate from `align_exercise` because it needs no model: "I couldn't hear
    any speech" is both certain and actionable, and the learner should get that
    answer even on an engine whose weights are missing — where the alternative
    is a vague "try again" that blames them for the engine's state.
    """
    if audio.speech_interval_s is None:
        return "no speech detected in the recording"
    if audio.quality.clipping:
        return "clipping"
    if audio.quality.too_quiet:
        return "no signal from the microphone"
    return None


def align_exercise(
    audio: PreparedAudio,
    phone_labels: "tuple[str, ...] | list[str]",
    phone_map: PhoneMap,
    model: AcousticModel,
    *,
    min_confidence: float,
    low_confidence: float,
) -> AlignmentOutcome:
    problem = audio_problem(audio)
    if problem is not None:
        return AlignmentOutcome("failed", 0.0, reason=problem)

    try:
        targets = canonical_targets(phone_labels)
        token_indices = phone_map.to_indices(targets)
    except PhoneSetError as exc:
        # An unmappable phone is a content bug, not a learner problem; the pack
        # loader normally catches it at startup.
        log.error("phone mapping failed: %s", exc)
        return AlignmentOutcome("failed", 0.0, reason=str(exc))

    posteriors = model.posteriors(audio.speech_16k)
    try:
        segments, confidence = ctc.align(
            posteriors.log_probs, token_indices, blank=phone_map.blank_index
        )
    except ctc.AlignmentError as exc:
        return AlignmentOutcome("failed", 0.0, reason=str(exc))

    offset = audio.speech_offset_s
    duration = audio.duration_s
    phones = tuple(
        AlignedPhone(
            index=segment.index,
            label=targets[segment.index],
            start_s=round(min(offset + segment.start_frame * posteriors.hop_s, duration), 4),
            end_s=round(min(offset + segment.end_frame * posteriors.hop_s, duration), 4),
            gop=round(segment.gop, 4),
            confidence=round(segment.confidence, 4),
        )
        for segment in segments
    )

    if confidence < min_confidence:
        return AlignmentOutcome(
            "failed", confidence, phones, reason="alignment confidence below the usable floor"
        )
    status: AlignmentStatus = "ok" if confidence >= low_confidence else "low_confidence"
    return AlignmentOutcome(status, round(confidence, 4), phones)
