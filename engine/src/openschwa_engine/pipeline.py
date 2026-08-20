"""The analysis pipeline, callable as a library.

api/analyze.py handles HTTP concerns - multipart decode, size limits, 404s -
and calls analyze_recording here. eval/ imports this module directly, so the
eval harness runs with no HTTP in the loop (eval/README.md).

The function never raises for bad audio, a missing model, or a missing
calibration: every stage that can fail degrades to a non-ok alignment status
or to honest silence, never to an HTTP 500 or an invented verdict.
"""

import logging
from typing import Literal

import numpy as np

from openschwa_engine import __version__
from openschwa_engine.alignment import AlignedPhone, AlignmentOutcome, acoustic, align_exercise
from openschwa_engine.alignment.acoustic import AcousticModel
from openschwa_engine.alignment.aligner import audio_problem
from openschwa_engine.audio import PreparedAudio
from openschwa_engine.config import Settings
from openschwa_engine.content import Exercise
from openschwa_engine.feedback import compose
from openschwa_engine.models.phone_set import PhoneMap, PhoneSetError
from openschwa_engine.models.registry import ModelError, ModelRegistry
from openschwa_engine.prosody import track
from openschwa_engine.schemas.analysis import (
    SCHEMA_VERSION,
    Alignment,
    AnalysisResult,
    AudioInfo,
    AudioQuality,
    ContrastResult,
    F0Track,
    Phone,
    Prosody,
    Word,
)
from openschwa_engine.scoring import (
    Calibration,
    ContrastScore,
    decide,
    load_calibration,
    score_contrast,
)

log = logging.getLogger(__name__)


_mismatch_warned: set[tuple[str, str]] = set()


def _matching_calibration(settings: Settings) -> Calibration | None:
    """The committed calibration, or None when it must not be used.

    A calibration fitted for a different acoustic model is exactly the
    wrong-verdict failure mode the design exists to prevent, so a model_id
    mismatch is refused loudly rather than applied quietly.
    """
    global _mismatch_warned
    calibration = load_calibration()
    if calibration is None:
        return None
    # Verdicts are produced by the contrast model when one is configured; a
    # calibration fitted for anything else must never be applied.
    scoring_model = settings.contrast_model_id or settings.alignment_model
    if calibration.model_id != scoring_model:
        pair = (calibration.model_id, scoring_model)
        if pair not in _mismatch_warned:
            _mismatch_warned.add(pair)
            log.warning(
                "calibration.yaml was fitted for model '%s' but the engine scores with '%s' - "
                "refusing to judge with another model's thresholds",
                calibration.model_id,
                scoring_model,
            )
        return None
    return calibration


def _alignment_thresholds(
    settings: Settings, calibration: Calibration | None
) -> tuple[float, float]:
    """Alignment gates: calibrated values when the eval produced them, the
    config placeholders otherwise (M0 behaviour, docs/architecture.md section 9)."""
    if calibration is not None:
        return calibration.alignment.min_confidence, calibration.alignment.low_confidence
    return settings.min_alignment_confidence, settings.low_alignment_confidence


def _run_alignment(
    audio: PreparedAudio,
    exercise: Exercise,
    registry: ModelRegistry,
    settings: Settings,
    min_confidence: float,
    low_confidence: float,
) -> tuple[AlignmentOutcome, PhoneMap | None]:
    """Align, converting any model-availability problem into a retry outcome.

    Recording-level problems are diagnosed first: they need no model, and
    "I couldn't hear any speech" is a better answer than a generic retry even
    when the engine also happens to have no weights installed.
    """
    problem = audio_problem(audio)
    if problem is not None:
        return AlignmentOutcome("failed", 0.0, reason=problem), None

    try:
        spec = registry.spec(settings.alignment_model)
        model_dir = registry.require_ready(spec)
        phone_map = registry.phone_map(spec)
        model = acoustic.load(model_dir)
    except ModelError as exc:
        log.warning("alignment unavailable: %s", exc)
        return AlignmentOutcome("failed", 0.0, reason=str(exc)), None

    return (
        align_exercise(
            audio,
            exercise.phone_labels,
            phone_map,
            model,
            min_confidence=min_confidence,
            low_confidence=low_confidence,
        ),
        phone_map,
    )


def _load_contrast(
    registry: ModelRegistry, settings: Settings
) -> tuple[AcousticModel, PhoneMap] | None:
    """The dedicated closed-set contrast judge, or None to use the aligner.

    The Option 3 model's vocabulary is exactly {blank, ð, z, d, v}: it cannot
    align, so it only ever scores the focus segment. When it is not configured
    or not downloaded, the engine degrades to aligner-based contrast scoring
    (the M0/M1 path) rather than failing.
    """
    if settings.contrast_model_id is None:
        return None
    try:
        spec = registry.spec(settings.contrast_model_id)
        model_dir = registry.require_ready(spec)
        phone_map = registry.phone_map(spec)
        return acoustic.load(model_dir), phone_map
    except ModelError as exc:
        log.warning(
            "contrast model '%s' unavailable (%s) - using aligner-based contrast",
            settings.contrast_model_id,
            exc,
        )
        return None


def _focus_segment(audio: PreparedAudio, phone: AlignedPhone, pad_s: float) -> np.ndarray:
    """The focus phone's 16 kHz samples, with context padding and onset anchoring.

    When phone.index == 0, we anchor the start to include speech onset from VAD,
    ensuring early substituted fricatives (e.g. /z/ in 'zis') are never clipped
    if forced alignment drifted forward onto the vowel.
    """
    if phone.index == 0 and audio.speech_interval_s is not None:
        speech_start = audio.speech_interval_s[0]
        start = max(0, min(int(speech_start * 16_000), int((phone.start_s - pad_s) * 16_000)))
    else:
        start = max(0, int((phone.start_s - pad_s) * 16_000))
    end = min(audio.samples_16k.size, int((phone.end_s + pad_s) * 16_000))
    segment = audio.samples_16k[start:end]
    if segment.size == 0:
        raise ValueError("focus interval lies outside the audio")
    return segment


def _words(exercise: Exercise, outcome: AlignmentOutcome) -> list[Word]:
    """M0 emits one word spanning the utterance; real word segmentation needs
    per-word phone grouping, which the exercise schema does not carry yet."""
    if not outcome.phones:
        return []
    return [
        Word(
            text=exercise.text,
            start_s=outcome.phones[0].start_s,
            end_s=outcome.phones[-1].end_s,
            phone_indices=[p.index for p in outcome.phones],
        )
    ]


def _contrasts(
    audio: PreparedAudio,
    exercise: Exercise,
    outcome: AlignmentOutcome,
    phone_map: PhoneMap | None,
    calibration: Calibration | None,
    contrast: tuple[AcousticModel, PhoneMap] | None = None,
    focus_pad_s: float = 0.10,
) -> list[ContrastResult]:
    """Closed-set contrast evidence for the focus phone (M1).

    With a dedicated contrast model (Option 3) the focus SEGMENT is scored by
    it - its vocabulary is exactly the closed set, so every frame is in-set
    and the renormalization is over its whole alphabet. Without one, the
    aligner's posteriors over the forced label frames are scored instead.

    The raw posteriors always come back - they are evidence, like the phone
    timeline. The verdict needs the committed calibration; without it the
    result stays 'uncertain' and the composer stays silent.
    """
    # 'low_confidence' keeps its phones but withholds verdicts (contract);
    # 'failed' has no phones at all. Only 'ok' may judge.
    if outcome.status != "ok":
        return []
    focus = exercise.focus_phone
    if focus is None or (phone_map is None and contrast is None):
        return []
    phone = next((p for p in outcome.phones if p.index == focus.index), None)
    if phone is None:
        log.warning("focus phone /%s/ is missing from the alignment - no contrast", focus.ph)
        return []

    raw: ContrastScore | None = None
    try:
        if contrast is not None:
            contrast_model, contrast_map = contrast
            if focus.ph in contrast_map.index_of:
                valid_confusions = tuple(c for c in focus.confusions if c in contrast_map.index_of)
                # Open-set judges (Stage 3) carry an "other" class that packs
                # never author: when the model's vocabulary has it, it always
                # joins the closed set - every non-drilled realization votes.
                if "other" in contrast_map.index_of and "other" not in valid_confusions:
                    valid_confusions = (*valid_confusions, "other")
                if valid_confusions:
                    segment = _focus_segment(audio, phone, focus_pad_s)
                    posteriors = contrast_model.posteriors(segment)
                    raw = score_contrast(
                        posteriors.log_probs,
                        np.arange(posteriors.frames, dtype=np.int64),
                        focus.ph,
                        valid_confusions,
                        contrast_map,
                    )
        if raw is None:
            if phone_map is None or outcome.posteriors is None or not phone.frame_indices:
                log.warning("focus phone /%s/ has no aligner frames - no contrast", focus.ph)
                return []
            raw = score_contrast(
                outcome.posteriors.log_probs,
                phone.frame_indices,
                focus.ph,
                focus.confusions,
                phone_map,
            )
    except (ValueError, PhoneSetError) as exc:
        log.error("contrast scoring failed for /%s/: %s", focus.ph, exc)
        return []

    contrast_calibration = calibration.contrast(focus.ph) if calibration is not None else None
    verdict: Literal["on_target", "substituted", "uncertain"] = "uncertain"
    confidence: float = 0.0
    detected: str | None = None
    if contrast_calibration is not None:
        verdict, confidence, detected = decide(raw, contrast_calibration, gop=phone.gop)

    return [
        ContrastResult(
            phone_index=focus.index,
            target=focus.ph,
            confusion_set=list(raw.confusions),
            posteriors={name: round(value, 4) for name, value in raw.posteriors.items()},
            verdict=verdict,
            detected=detected,
            confidence=round(confidence, 4),
            spike_score=round(raw.spike_score, 4),
            vote_fraction=round(raw.vote_fraction, 4),
        )
    ]


def _phone_score(
    phone: AlignedPhone, exercise: Exercise, calibration: Calibration | None
) -> float | None:
    """GOP mapped to [0, 1] through the committed calibration.

    Only the drilled phone has calibration evidence; every other phone keeps
    score=null, because an uncalibrated number there would read as a verdict.
    """
    focus = exercise.focus_phone
    if calibration is None or focus is None:
        return None
    if phone.index != focus.index:
        return None
    contrast_calibration = calibration.contrast(focus.ph)
    if contrast_calibration is None or contrast_calibration.gop_platt is None:
        return None
    return round(contrast_calibration.gop_platt.probability(phone.gop), 4)


def analyze_recording(
    audio: PreparedAudio,
    exercise: Exercise,
    registry: ModelRegistry,
    settings: Settings,
    *,
    include_ungated: bool = False,
) -> AnalysisResult:
    """Run the full pipeline on prepared audio. See the module docstring.

    include_ungated=True is the eval harness's switch: the composer then
    reports items the confidence gate would withhold, so the harness can score
    flags against corpus labels across the whole PR curve.
    """
    calibration = _matching_calibration(settings)
    min_confidence, low_confidence = _alignment_thresholds(settings, calibration)
    outcome, phone_map = _run_alignment(
        audio, exercise, registry, settings, min_confidence, low_confidence
    )

    f0 = track(audio.samples_16k, 16_000)
    prosody = (
        Prosody(
            f0=F0Track(
                hop_s=f0.hop_s,
                start_s=f0.start_s,
                semitones=list(f0.semitones),
                median_hz=f0.median_hz,
            )
        )
        if f0
        else None
    )

    contrast = _load_contrast(registry, settings)
    contrasts = _contrasts(
        audio,
        exercise,
        outcome,
        phone_map,
        calibration,
        contrast,
        focus_pad_s=settings.focus_pad_s,
    )

    return AnalysisResult(
        schema_version=SCHEMA_VERSION,  # type: ignore[arg-type]
        engine_version=__version__,
        exercise_id=exercise.id,
        audio=AudioInfo(
            duration_s=round(audio.duration_s, 4),
            sample_rate=audio.sample_rate,
            speech_interval_s=audio.speech_interval_s,
            quality=AudioQuality(
                clipping=audio.quality.clipping,
                snr_db_est=audio.quality.snr_db_est,
                too_quiet=audio.quality.too_quiet,
                speech_level_dbfs=audio.quality.speech_level_dbfs,
                peak_dbfs=audio.quality.peak_dbfs,
            ),
        ),
        alignment=Alignment(
            status=outcome.status,
            confidence=outcome.confidence,
            reason=outcome.reason if outcome.status == "failed" else None,
            # 'low_confidence' keeps its phones: the contract withholds
            # *verdicts* below the gate, not evidence. The timeline still helps
            # a learner see what was heard, and feedback already says retry.
            words=_words(exercise, outcome) if outcome.status != "failed" else [],
            phones=[
                Phone(
                    index=phone.index,
                    label=phone.label,
                    start_s=phone.start_s,
                    end_s=phone.end_s,
                    gop=phone.gop,
                    score=_phone_score(phone, exercise, calibration),
                    confidence=phone.confidence,
                )
                for phone in outcome.phones
            ]
            if outcome.status != "failed"
            else [],
        ),
        contrasts=contrasts,
        prosody=prosody,
        annotations=[],
        feedback=compose(
            outcome,
            contrasts,
            include_ungated=include_ungated,
            calibration=calibration,
        ),
    )
