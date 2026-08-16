"""Exercise catalog. The UI renders target text, IPA, and focus phone from
these before the learner records anything."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from openschwa_engine.api.deps import get_library
from openschwa_engine.content import ContentLibrary, Exercise
from openschwa_engine.schemas.analysis import SCHEMA_VERSION
from openschwa_engine.schemas.content import (
    ExerciseCatalog,
    ExerciseDetail,
    ExercisePhone,
    ExerciseProsody,
    ExerciseSummary,
)

router = APIRouter()

Library = Annotated[ContentLibrary, Depends(get_library)]


def _summary(exercise: Exercise) -> ExerciseSummary:
    focus = exercise.focus_phone
    return ExerciseSummary(
        id=exercise.id,
        pack_id=exercise.pack_id,
        type=exercise.type,  # type: ignore[arg-type]  # loader validated against the enum
        title=exercise.title,
        text=exercise.text,
        ipa=exercise.ipa,
        focus_phone=focus.ph if focus else None,
    )


def _require(library: ContentLibrary, exercise_id: str) -> Exercise:
    exercise = library.get(exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail=f"unknown exercise '{exercise_id}'")
    return exercise


@router.get("/exercises")
def list_exercises(library: Library) -> ExerciseCatalog:
    return ExerciseCatalog(
        schema_version=SCHEMA_VERSION,  # type: ignore[arg-type]
        exercises=[_summary(e) for e in library.ordered()],
    )


@router.get("/exercises/{exercise_id}")
def get_exercise(exercise_id: str, library: Library) -> ExerciseDetail:
    exercise = _require(library, exercise_id)
    return ExerciseDetail(
        schema_version=SCHEMA_VERSION,  # type: ignore[arg-type]
        id=exercise.id,
        pack_id=exercise.pack_id,
        type=exercise.type,  # type: ignore[arg-type]
        title=exercise.title,
        lang=exercise.lang,
        text=exercise.text,
        ipa=exercise.ipa,
        phones=[
            ExercisePhone(index=p.index, ph=p.ph, focus=p.focus, confusions=list(p.confusions))
            for p in exercise.phones
        ],
        pair_with=exercise.pair_with,
        prosody=(
            ExerciseProsody(
                nuclear_syllable_index=exercise.prosody.nuclear_syllable_index,
                expected_tone=exercise.prosody.expected_tone,  # type: ignore[arg-type]
            )
            if exercise.prosody
            else None
        ),
        learner_notes=exercise.learner_notes,
        has_reference_audio=exercise.has_reference_audio,
    )


@router.get("/exercises/{exercise_id}/reference-audio")
def get_reference_audio(exercise_id: str, library: Library) -> FileResponse:
    """Teacher reference WAV. 404 while a pack's recordings are still pending —
    `has_reference_audio` on the detail response tells the UI in advance."""
    exercise = _require(library, exercise_id)
    if not exercise.has_reference_audio:
        raise HTTPException(
            status_code=404, detail=f"no reference audio recorded for '{exercise_id}'"
        )
    assert exercise.reference_audio_path is not None  # implied by has_reference_audio
    return FileResponse(exercise.reference_audio_path, media_type="audio/wav")
