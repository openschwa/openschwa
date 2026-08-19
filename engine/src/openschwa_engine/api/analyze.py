"""POST /v1/analyze - the HTTP surface of the pipeline.

This module handles only HTTP concerns: the multipart upload, size limits, and
the 404 for an unknown exercise. The analysis itself lives in
pipeline.analyze_recording, which the eval harness imports directly (no HTTP
in the loop, eval/README.md).

Defined with 'def' rather than 'async def' on purpose: the pipeline is
CPU-bound, so FastAPI runs it in a worker thread and the event loop stays free
to serve health polls and the model-download stream.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from openschwa_engine.api.deps import get_library, get_registry, get_settings
from openschwa_engine.audio import AudioDecodeError, decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.content import ContentLibrary, Exercise
from openschwa_engine.models.registry import ModelRegistry
from openschwa_engine.pipeline import analyze_recording
from openschwa_engine.schemas.analysis import AnalysisResult

router = APIRouter()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DURATION_S = 30.0


@router.post("/analyze")
def analyze(
    library: Annotated[ContentLibrary, Depends(get_library)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    settings: Annotated[Settings, Depends(get_settings)],
    exercise_id: Annotated[str, Form()],
    audio: Annotated[UploadFile, File()],
    include_ungated: Annotated[bool, Query()] = False,
    #: The learner's first language (e.g. "mandarin"): the engine applies the
    #: per-L1 operating threshold when calibration carries one for it.
    l1: Annotated[str | None, Query()] = None,
) -> AnalysisResult:
    exercise: Exercise | None = library.get(exercise_id)
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

    prepared = prepare(decoded.samples, decoded.sample_rate, vad_backend=settings.vad_backend)
    return analyze_recording(
        prepared, exercise, registry, settings, include_ungated=include_ungated, l1=l1
    )
