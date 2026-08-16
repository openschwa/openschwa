"""Exercise API contract (v1) — the wire shape of /v1/exercises.

Distinct from `content/schema/exercise.schema.json`, which validates *authored*
YAML. This module describes what the UI consumes: authoring details it does not
need are dropped, and server-derived fields are added (`pack_id`,
`has_reference_audio`).

Like `analysis.py`, these models are the source of truth — `just schema`
exports them to `schemas/*.v1.schema.json` and regenerates the UI's types, and
CI fails on drift.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ExerciseType = Literal["minimal_pair", "word", "sentence", "intonation"]
Tone = Literal["fall", "rise", "fall_rise", "level"]


class _Model(BaseModel):
    """Strict base: unknown fields are contract drift, so they are rejected."""

    model_config = ConfigDict(extra="forbid")


class ExercisePhone(_Model):
    index: int = Field(description="Position in the target phone sequence; matches Phone.index.")
    ph: str = Field(description="Canonical IPA label from the engine's internal inventory.")
    focus: bool = False
    confusions: list[str] = []


class ExerciseProsody(_Model):
    nuclear_syllable_index: int
    expected_tone: Tone


class ExerciseSummary(_Model):
    """Enough to render a picker without fetching every exercise."""

    id: str
    pack_id: str
    type: ExerciseType
    title: str
    text: str
    ipa: str
    focus_phone: str | None = Field(
        default=None, description="The drilled phone; null for intonation exercises."
    )


class ExerciseCatalog(_Model):
    schema_version: Literal["1.0"]
    exercises: list[ExerciseSummary] = []


class ExerciseDetail(_Model):
    """Everything the UI needs to present a drill before recording."""

    schema_version: Literal["1.0"]
    id: str
    pack_id: str
    type: ExerciseType
    title: str
    lang: str
    text: str
    ipa: str
    phones: list[ExercisePhone]
    pair_with: str | None = None
    prosody: ExerciseProsody | None = None
    learner_notes: str | None = None
    has_reference_audio: bool = Field(
        description=(
            "False when the pack declares reference audio that has not been recorded yet — "
            "the UI hides playback rather than offering a broken control."
        )
    )
