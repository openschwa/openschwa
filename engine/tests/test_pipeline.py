"""The library pipeline: degraded paths, calibration matching.

The pipeline is the eval harness's entry point (no HTTP), so these tests pin
the two properties the harness relies on: bad input degrades instead of
raising, and a calibration fitted for another model is refused.
"""

import io
import wave

import numpy as np

from openschwa_engine import pipeline
from openschwa_engine.audio import decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.content import load_library
from openschwa_engine.models.registry import ModelRegistry
from openschwa_engine.scoring.calibration import (
    AlignmentCalibration,
    Calibration,
    ContrastCalibration,
    PlattCalibration,
)


def settings(tmp_path) -> Settings:
    # Energy VAD pinned: the pipeline tests must not depend on whether the
    # ml extra (silero) happens to be installed in this environment.
    return Settings(model_dir=tmp_path / "models", vad_backend="energy")


def library(settings: Settings):
    return load_library(settings.content_dir, settings.content_schema_path)


def prepared_audio(duration_s: float = 1.2) -> "object":
    rate = 48_000
    t = np.arange(int(rate * duration_s)) / rate
    signal = np.random.RandomState(0).normal(0, 0.0015, t.shape)
    speaking = (t > 0.2) & (t < duration_s - 0.2)
    for harmonic in range(6):
        signal[speaking] += (
            0.4 / (harmonic + 1) * np.sin(2 * np.pi * 140.0 * (harmonic + 1) * t[speaking])
        )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
    decoded = decode_wav(buffer.getvalue())
    return prepare(decoded.samples, decoded.sample_rate)


def test_empty_model_dir_degrades_to_a_valid_retry(tmp_path):
    """The harness must be able to run the pipeline without weights and still
    get a schema-valid result - the same promise the HTTP surface makes."""
    s = settings(tmp_path)
    exercise = library(s).get("en.seg.dh-z.this")
    assert exercise is not None
    result = pipeline.analyze_recording(prepared_audio(), exercise, ModelRegistry(s.model_dir), s)
    assert result.alignment.status == "failed"
    assert result.contrasts == []
    assert [item.kind for item in result.feedback] == ["retry"]


def test_silence_is_diagnosed_before_the_model_is_consulted(tmp_path):
    s = settings(tmp_path)
    exercise = library(s).get("en.seg.dh-z.this")
    rate = 48_000
    silence = np.random.RandomState(2).normal(0, 0.0005, rate).astype(np.float32)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((silence * 32767).astype("<i2").tobytes())
    result = pipeline.analyze_recording(
        prepare(decode_wav(buffer.getvalue()).samples, rate),
        exercise,
        ModelRegistry(s.model_dir),
        s,
    )
    assert "couldn't hear any speech" in result.feedback[0].message


def foreign_calibration() -> Calibration:
    return Calibration(
        schema_version="1.0",
        generated_by="eval/reports/test.json",
        model_id="some-other-model",
        contrasts=[
            ContrastCalibration(
                target="ð",
                confusions=["z", "d", "v"],
                substitution_platt=PlattCalibration(a=1.0, b=0.0),
                threshold=0.85,
            )
        ],
        alignment=AlignmentCalibration(min_confidence=0.30, low_confidence=0.55),
    )


def test_calibration_fitted_for_another_model_is_refused(tmp_path, monkeypatch):
    """Judging with another model's thresholds is the silent-wrong-verdict
    failure mode; it must be refused loudly, not applied."""
    monkeypatch.setattr(pipeline, "load_calibration", lambda: foreign_calibration())
    assert pipeline._matching_calibration(settings(tmp_path)) is None


def test_matching_calibration_is_used(tmp_path, monkeypatch):
    cal = foreign_calibration().model_copy(update={"model_id": "charsiu-en-w2v2-ctc"})
    monkeypatch.setattr(pipeline, "load_calibration", lambda: cal)
    assert pipeline._matching_calibration(settings(tmp_path)) == cal
