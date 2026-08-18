"""The library pipeline: degraded paths, calibration matching.

The pipeline is the eval harness's entry point (no HTTP), so these tests pin
the two properties the harness relies on: bad input degrades instead of
raising, and a calibration fitted for another model is refused.
"""

import io
import wave

import numpy as np
import pytest

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


def test_calibration_matches_the_contrast_model_when_configured(tmp_path, monkeypatch):
    cal = foreign_calibration().model_copy(update={"model_id": "dh-contrast-v1"})
    monkeypatch.setattr(pipeline, "load_calibration", lambda: cal)
    s = settings(tmp_path).model_copy(update={"contrast_model_id": "dh-contrast-v1"})
    assert pipeline._matching_calibration(s) == cal


def test_contrast_model_scores_the_focus_segment(monkeypatch):
    """Option 3: the closed-set judge scores the focus SEGMENT, not the
    aligner's frame posteriors."""
    import json

    import numpy as np

    from openschwa_engine.alignment import AlignedPhone, AlignmentOutcome
    from openschwa_engine.alignment.acoustic import Posteriors
    from openschwa_engine.audio import PreparedAudio, QualityReport
    from openschwa_engine.content.loader import Exercise, PhoneSpec
    from openschwa_engine.models.phone_set import PhoneMap
    from openschwa_engine.models.registry import VOCAB_DIR

    class StubContrast:
        def posteriors(self, samples):
            logp = np.log(np.full((4, 6), 0.01, dtype=np.float32))
            logp[:, 3] = np.log(0.9)  # z dominates
            return Posteriors(log_probs=logp, hop_s=0.02)

    vocab = json.loads((VOCAB_DIR / "dh-contrast-v1.json").read_text(encoding="utf-8"))
    contrast_map = PhoneMap.build("dh-contrast-v1", "dhz_en", vocab)

    exercise = Exercise(
        id="eval-x",
        pack_id="eval",
        type="word",
        title="",
        lang="en",
        text="this",
        ipa="",
        phones=(
            PhoneSpec(index=0, ph="ð", focus=True, confusions=("z", "d", "v")),
            PhoneSpec(index=1, ph="ɪ", focus=False, confusions=()),
        ),
        source_path=__import__("pathlib").Path("eval"),
    )
    audio = PreparedAudio(
        samples_16k=np.zeros(16_000, dtype=np.float32),
        duration_s=1.0,
        sample_rate=16_000,
        speech_interval_s=(0.0, 1.0),
        quality=QualityReport(clipping=False, too_quiet=False, snr_db_est=10.0),
    )
    outcome = AlignmentOutcome(
        "ok",
        0.9,
        phones=(
            AlignedPhone(
                index=0,
                label="ð",
                start_s=0.1,
                end_s=0.3,
                gop=-0.5,
                confidence=0.9,
                frame_indices=(1,),
            ),
        ),
    )
    results = pipeline._contrasts(
        audio, exercise, outcome, None, None, (StubContrast(), contrast_map)
    )
    assert len(results) == 1
    # Renormalized over {ð, z, d, v}: 0.9 / (0.9 + 3 * 0.01).
    assert results[0].posteriors["z"] == pytest.approx(0.9 / 0.93, abs=1e-3)
    assert results[0].spike_score > 1.0
