"""AnalysisResult v1 — the wire contract between engine and UI.

These pydantic models are the single source of truth for the OpenSchwa API.
`schemas/analysis_result.v1.schema.json` at the repo root and the UI's
generated TypeScript types (`ui/src/lib/api/types.gen.ts`) are both derived
from them; CI fails if either committed artifact drifts (`just schema`
regenerates both).

Contract invariants:
- All times are seconds in the *original uploaded-audio* timeline, so the
  client renders without offset math.
- `feedback` contains ONLY items that passed the confidence gate. Everything
  else in the result is evidence a rich UI may visualize; a naive UI can
  render `feedback` alone and be correct.
- `alignment.status != "ok"` short-circuits: no contrasts, no verdicts, and
  `feedback` holds a single "retry" item. Refusing to judge is a first-class
  outcome.
- Breaking changes require a v2 module and schema file; v1 is never mutated
  once released.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


class _Model(BaseModel):
    """Strict base: unknown fields are contract drift, so they are rejected."""

    model_config = ConfigDict(extra="forbid")


class AudioQuality(_Model):
    clipping: bool
    snr_db_est: float | None = None
    too_quiet: bool = Field(
        description=(
            "The input is effectively dead — no usable signal at all. NOT merely a low "
            "recording level: absolute level says nothing about whether speech can be "
            "analysed, so a quiet-but-clean recording is analysed normally and the UI "
            "advises from `speech_level_dbfs` instead of refusing."
        )
    )
    speech_level_dbfs: float | None = Field(
        default=None, description="RMS of the detected speech region, dB relative to full scale."
    )
    peak_dbfs: float | None = Field(
        default=None, description="Peak sample level of the whole recording, dBFS."
    )


class AudioInfo(_Model):
    duration_s: float
    sample_rate: int
    speech_interval_s: tuple[float, float] | None = Field(
        default=None, description="Post-VAD speech region; null when no speech was detected."
    )
    quality: AudioQuality


class Word(_Model):
    text: str
    start_s: float
    end_s: float
    phone_indices: list[int]


class Phone(_Model):
    index: int
    label: str = Field(description="Canonical IPA label from the engine's internal inventory.")
    start_s: float
    end_s: float
    gop: float | None = Field(
        default=None, description="Raw goodness-of-pronunciation (log-posterior based)."
    )
    score: float | None = Field(
        default=None, ge=0, le=1, description="GOP mapped to [0,1] via committed calibration."
    )
    confidence: float = Field(ge=0, le=1)


class Alignment(_Model):
    status: Literal["ok", "low_confidence", "failed"]
    confidence: float = Field(ge=0, le=1)
    words: list[Word] = []
    phones: list[Phone] = []
    reason: str | None = Field(
        default=None,
        description="Why the analysis refused (no speech, clipping, missing model, ...). "
        "Set only when status=failed; the composer turns it into a retry message.",
    )


class ContrastResult(_Model):
    """Closed-set discrimination for one focus phone of the exercise."""

    phone_index: int
    target: str
    confusion_set: list[str]
    posteriors: dict[str, float] = Field(
        description="Posterior mass renormalized over {target} ∪ confusion_set."
    )
    verdict: Literal["on_target", "substituted", "uncertain"]
    detected: str | None = Field(
        default=None, description="The confusion phone heard; set only when verdict=substituted."
    )
    confidence: float = Field(ge=0, le=1, description="Calibrated (Platt-scaled), not raw margin.")
    #: The mirror (M1 pivot): the argmax phone over the closed set - what the
    #: model actually heard - and its raw odds. Calibration-free evidence:
    #: turning it into the shipped phone_hearing item needs the committed
    #: hearing block (calibration.yaml), exactly like the verdict path.
    heard: str | None = Field(
        default=None, description="Argmax phone over {target} ∪ confusion_set."
    )
    hearing_score: float | None = Field(
        default=None, description="log(p_heard / (1 - p_heard)), raw (uncalibrated)."
    )
    #: Alternative score aggregations of the same frames (bake-off evidence;
    #: clients may ignore them). See scoring/contrast.py for the definitions.
    spike_score: float | None = Field(
        default=None,
        description="Log-ratio on the single frame most favouring a confusion.",
    )
    vote_fraction: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Share of label frames where a confusion outvoted the target.",
    )


class F0Track(_Model):
    hop_s: float
    start_s: float
    semitones: list[float | None] = Field(
        description="F0 in semitones relative to the speaker's median; null = unvoiced frame."
    )
    median_hz: float | None = None
    #: Share of frames the pitch tracker had to rescue from (or drop to) an
    #: octave jump - the M2 bar's octave-error tripwire (< 2% of voiced frames).
    octave_error_rate: float | None = None


class NuclearTone(_Model):
    detected: Literal["fall", "rise", "fall_rise", "level"]
    expected: Literal["fall", "rise", "fall_rise", "level"] | None = None
    match: bool | None = None
    confidence: float = Field(ge=0, le=1)


class Prosody(_Model):
    f0: F0Track
    reference: F0Track | None = Field(
        default=None, description="Teacher-reference contour, precomputed at pack load."
    )
    dtw_distance: float | None = Field(
        default=None, description="Per-frame-normalized DTW over voiced regions."
    )
    nuclear_tone: NuclearTone | None = None


class Annotation(_Model):
    """Acoustic event for spectrogram overlay. New `type`s may be added in
    minor schema versions; clients must ignore types they don't know."""

    type: Literal["vot", "voicing", "duration", "formants"]
    phone_index: int | None = None
    interval_s: tuple[float, float]
    value: float | None = None
    unit: str
    expected_range: tuple[float, float] | None = None
    verdict: Literal["in_range", "outside_range"] | None = None
    confidence: float = Field(ge=0, le=1)
    f1: float | None = Field(default=None, description="formants only")
    f2: float | None = Field(default=None, description="formants only")


class Anchor(_Model):
    phone_index: int | None = None
    interval_s: tuple[float, float] | None = None


class FeedbackItem(_Model):
    id: str
    kind: str = Field(
        description=(
            "Open enum, grows per milestone. Current: segmental_substitution, retry, "
            "phone_hearing (the M1 mirror: what the ear heard at the focus slot). "
            "Planned: nuclear_tone_mismatch (M2), vot_out_of_range (M3)."
        )
    )
    severity: Literal["error", "warning", "praise"]
    confidence: float = Field(ge=0, le=1)
    message_key: str = Field(description="Stable key for future i18n.")
    message: str
    anchor: Anchor | None = Field(default=None, description="What the UI highlights.")
    evidence: dict[str, int] = Field(
        default_factory=dict,
        description="Pointers into contrasts/annotations/prosody, e.g. {'contrast_index': 0}.",
    )


class AnalysisResult(_Model):
    schema_version: Literal["1.0"]
    engine_version: str
    exercise_id: str
    audio: AudioInfo
    alignment: Alignment
    contrasts: list[ContrastResult] = []
    prosody: Prosody | None = None
    annotations: list[Annotation] = []
    feedback: list[FeedbackItem] = []
