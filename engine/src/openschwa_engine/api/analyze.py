"""POST /v1/analyze — the pipeline entry point.

decode -> resample -> VAD trim -> quality checks -> forced alignment -> F0 ->
feedback composition. Every stage that can fail degrades to a non-`ok`
alignment status rather than an HTTP error: a learner recording something
unusable, or an engine with no model downloaded, must still get a
schema-valid result carrying a "retry".

HTTP errors are reserved for the client getting the *request* wrong — unknown
exercise, non-WAV upload, absurd file size.

Defined with `def` rather than `async def` on purpose: the pipeline is
CPU-bound, so FastAPI runs it in a worker thread and the event loop stays free
to serve health polls and the model-download stream.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from openschwa_engine import __version__
from openschwa_engine.alignment import AlignmentOutcome, acoustic, align_exercise
from openschwa_engine.alignment.aligner import audio_problem
from openschwa_engine.api.deps import get_library, get_registry, get_settings
from openschwa_engine.audio import AudioDecodeError, PreparedAudio, decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.content import ContentLibrary, Exercise
from openschwa_engine.feedback import compose
from openschwa_engine.models.registry import ModelError, ModelRegistry
from openschwa_engine.prosody import track
from openschwa_engine.schemas.analysis import (
    SCHEMA_VERSION,
    Alignment,
    AnalysisResult,
    AudioInfo,
    AudioQuality,
    F0Track,
    Phone,
    Prosody,
    Word,
)

log = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DURATION_S = 30.0


def _words(exercise: Exercise, outcome: AlignmentOutcome) -> list[Word]:
    """M0 emits one word spanning the utterance.

    Real word segmentation needs per-word phone grouping, which the exercise
    schema does not carry yet — correct for the word and minimal-pair drills
    that exist today, and a content-schema change before sentence drills ship.
    """
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


def _run_alignment(
    audio_prepared: PreparedAudio,
    exercise: Exercise,
    registry: ModelRegistry,
    settings: Settings,
) -> AlignmentOutcome:
    """Align, converting any model-availability problem into a retry outcome."""
    # Recording-level problems are diagnosed first: they need no model, and
    # "I couldn't hear any speech" is a better answer than a generic retry even
    # when the engine also happens to have no weights installed.
    problem = audio_problem(audio_prepared)
    if problem is not None:
        return AlignmentOutcome("failed", 0.0, reason=problem)

    try:
        spec = registry.spec(settings.alignment_model)
        model_dir = registry.require_ready(spec)
        phone_map = registry.phone_map(spec)
        model = acoustic.load(model_dir)
    except ModelError as exc:
        log.warning("alignment unavailable: %s", exc)
        return AlignmentOutcome("failed", 0.0, reason=str(exc))

    return align_exercise(
        audio_prepared,
        exercise.phone_labels,
        phone_map,
        model,
        min_confidence=settings.min_alignment_confidence,
        low_confidence=settings.low_alignment_confidence,
    )


@router.post("/analyze")
def analyze(
    library: Annotated[ContentLibrary, Depends(get_library)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    settings: Annotated[Settings, Depends(get_settings)],
    exercise_id: Annotated[str, Form()],
    audio: Annotated[UploadFile, File()],
) -> AnalysisResult:
    exercise = library.get(exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail=f"unknown exercise '{exercise_id}'")

    raw = audio.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
        )

    try:
        decoded = decode_wav(raw)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if decoded.duration_s > MAX_DURATION_S:
        raise HTTPException(status_code=400, detail=f"recording longer than {MAX_DURATION_S:.0f}s")

    prepared = prepare(decoded.samples, decoded.sample_rate)
    outcome = _run_alignment(prepared, exercise, registry, settings)

    f0 = track(prepared.samples_16k, 16_000)
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

    return AnalysisResult(
        schema_version=SCHEMA_VERSION,  # type: ignore[arg-type]
        engine_version=__version__,
        exercise_id=exercise.id,
        audio=AudioInfo(
            duration_s=round(prepared.duration_s, 4),
            sample_rate=prepared.sample_rate,
            speech_interval_s=prepared.speech_interval_s,
            quality=AudioQuality(
                clipping=prepared.quality.clipping,
                snr_db_est=prepared.quality.snr_db_est,
                too_quiet=prepared.quality.too_quiet,
                speech_level_dbfs=prepared.quality.speech_level_dbfs,
                peak_dbfs=prepared.quality.peak_dbfs,
            ),
        ),
        alignment=Alignment(
            status=outcome.status,
            confidence=outcome.confidence,
            # `low_confidence` keeps its phones: the contract withholds
            # *verdicts* below the gate, not evidence. The timeline still helps a
            # learner see what was heard, and `feedback` already says "retry".
            # `failed` means there is no trustworthy alignment to show at all.
            words=_words(exercise, outcome) if outcome.status != "failed" else [],
            phones=[
                Phone(
                    index=p.index,
                    label=p.label,
                    start_s=p.start_s,
                    end_s=p.end_s,
                    gop=p.gop,
                    # `score` stays null until calibration exists: mapping GOP to
                    # a 0-1 "how good was it" needs the M1 eval harness, and an
                    # uncalibrated number here would read as a verdict.
                    score=None,
                    confidence=p.confidence,
                )
                for p in outcome.phones
            ]
            if outcome.status != "failed"
            else [],
        ),
        # Closed-set contrast scoring ships in M1 with its calibration; M0 has no
        # judgement to report.
        contrasts=[],
        prosody=prosody,
        annotations=[],
        feedback=compose(outcome),
    )
