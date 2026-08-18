"""Audio front end: decoding, resampling, speech detection, quality gates.

Quality checks are the first line of the confidence gate — a clipped or silent
recording must become a "retry", never a verdict — so the thresholds are pinned
by test rather than left to drift.
"""

import io
import wave

import numpy as np
import pytest

from openschwa_engine.audio import (
    MODEL_SAMPLE_RATE,
    AudioDecodeError,
    decode_wav,
    detect_speech,
    prepare,
)
from openschwa_engine.audio.decode import WAVE_FORMAT_IEEE_FLOAT

RATE = 48_000


def wav_bytes(samples: np.ndarray, rate: int = RATE, channels: int = 1, width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        scale = {1: 128, 2: 32767, 4: 2147483647}[width]
        dtype = {1: "u1", 2: "<i2", 4: "<i4"}[width]
        data = np.clip(samples, -1, 1) * scale
        handle.writeframes((data + 128 if width == 1 else data).astype(dtype).tobytes())
    return buffer.getvalue()


def tone(duration_s: float, freq: float = 140.0, rate: int = RATE, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * duration_s)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def utterance(rate: int = RATE) -> np.ndarray:
    """Quiet room, then 0.6 s of a voiced-sounding burst, then quiet again."""
    t = np.arange(int(rate * 1.2)) / rate
    signal = np.random.RandomState(0).normal(0, 0.0015, t.shape)
    speaking = (t > 0.3) & (t < 0.9)
    for harmonic in range(6):
        signal[speaking] += (
            0.4 / (harmonic + 1) * np.sin(2 * np.pi * 140.0 * (harmonic + 1) * t[speaking])
        )
    return signal.astype(np.float32)


def test_decodes_16_bit_pcm():
    decoded = decode_wav(wav_bytes(tone(0.5)))
    assert decoded.sample_rate == RATE
    assert decoded.duration_s == pytest.approx(0.5, abs=0.01)
    assert decoded.samples.dtype == np.float32
    assert np.abs(decoded.samples).max() == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("width", [1, 2, 4])
def test_decodes_every_supported_pcm_width(width):
    decoded = decode_wav(wav_bytes(tone(0.2), width=width))
    assert decoded.samples.size == int(RATE * 0.2)
    assert np.abs(decoded.samples).max() == pytest.approx(0.5, abs=0.02)


def test_decodes_32_bit_float_wav():
    samples = tone(0.2)
    body = samples.astype("<f4").tobytes()
    header = (
        b"RIFF"
        + (36 + len(body)).to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + WAVE_FORMAT_IEEE_FLOAT.to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + RATE.to_bytes(4, "little")
        + (RATE * 4).to_bytes(4, "little")
        + (4).to_bytes(2, "little")
        + (32).to_bytes(2, "little")
        + b"data"
        + len(body).to_bytes(4, "little")
    )
    decoded = decode_wav(header + body)
    assert decoded.samples == pytest.approx(samples, abs=1e-6)


def test_downmixes_stereo_to_mono():
    left = tone(0.2)
    interleaved = np.stack([left, -left], axis=1).ravel()
    decoded = decode_wav(wav_bytes(interleaved, channels=2))
    assert decoded.samples.size == left.size
    assert np.abs(decoded.samples).max() < 1e-3  # opposed channels cancel


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"definitely not a wav", "RIFF/WAVE"),
        (b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 40, "fmt"),
    ],
)
def test_rejects_non_wav_uploads(payload, message):
    with pytest.raises(AudioDecodeError, match=message):
        decode_wav(payload)


def test_rejects_compressed_wav():
    """A codec would have erased the cues the engine measures, so guessing is
    worse than refusing."""
    good = bytearray(wav_bytes(tone(0.1)))
    fmt_offset = good.index(b"fmt ") + 8
    good[fmt_offset : fmt_offset + 2] = (0x0011).to_bytes(2, "little")  # IMA ADPCM
    with pytest.raises(AudioDecodeError, match="compressed"):
        decode_wav(bytes(good))


def test_resamples_to_the_model_rate():
    prepared = prepare(tone(1.0), RATE)
    assert prepared.samples_16k.size == pytest.approx(MODEL_SAMPLE_RATE, rel=0.01)
    assert prepared.sample_rate == RATE  # the original rate is what the API reports
    assert prepared.duration_s == pytest.approx(1.0, abs=0.01)


def test_detects_the_speech_region():
    interval, snr = detect_speech(utterance(MODEL_SAMPLE_RATE), MODEL_SAMPLE_RATE)
    assert interval is not None
    start, end = interval
    assert start == pytest.approx(0.3, abs=0.1)
    assert end == pytest.approx(0.9, abs=0.1)
    assert snr is not None and snr > 20


def test_reports_no_speech_for_a_silent_recording():
    silence = np.random.RandomState(1).normal(0, 0.001, MODEL_SAMPLE_RATE).astype(np.float32)
    interval, _ = detect_speech(silence, MODEL_SAMPLE_RATE)
    assert interval is None


def test_speech_interval_is_in_the_original_timeline():
    """Times are reported in seconds against the upload, whatever rate it used."""
    prepared = prepare(utterance(RATE), RATE)
    assert prepared.speech_interval_s is not None
    assert prepared.speech_interval_s[0] == pytest.approx(0.3, abs=0.1)
    assert prepared.speech_offset_s == prepared.speech_interval_s[0]
    assert prepared.speech_16k.size == pytest.approx(0.72 * MODEL_SAMPLE_RATE, rel=0.25)


def test_flags_clipping():
    prepared = prepare(np.clip(tone(1.0, amp=3.0), -1, 1), RATE)
    assert prepared.quality.clipping is True
    assert prepared.quality.usable is False


def test_flags_a_dead_input():
    """Only a genuinely dead device is refused — roughly -80 dBFS here."""
    prepared = prepare(tone(1.0, amp=0.00005), RATE)
    assert prepared.quality.too_quiet is True
    assert prepared.quality.usable is False


@pytest.mark.parametrize("attenuation_db", [-15, -25, -35])
def test_quiet_but_clean_speech_is_analysed_not_refused(attenuation_db):
    """The regression this replaces: a browser honouring `autoGainControl:
    false` hands back a raw capture 15-25 dB below what a consumer recorder
    produces. Its SNR is unchanged and the acoustic model normalises its input
    anyway, so refusing it told users to speak up when speaking up could not
    help."""
    quiet = (utterance(RATE) * 10 ** (attenuation_db / 20)).astype(np.float32)
    prepared = prepare(quiet, RATE)

    assert prepared.quality.too_quiet is False
    assert prepared.quality.usable is True
    assert prepared.speech_interval_s is not None
    # Scaling changes the level, not the separation from the noise floor.
    assert prepared.quality.snr_db_est is not None
    assert prepared.quality.snr_db_est > 20


def test_reports_the_measured_level_so_the_ui_can_advise():
    prepared = prepare(utterance(RATE), RATE)
    assert prepared.quality.speech_level_dbfs is not None
    assert -45 < prepared.quality.speech_level_dbfs < 0
    assert prepared.quality.peak_dbfs is not None
    assert prepared.quality.low_level is False


def test_a_normal_recording_passes_quality():
    prepared = prepare(utterance(RATE), RATE)
    assert prepared.quality.usable is True
    assert prepared.quality.clipping is False


def test_energy_backend_is_forced_by_name():
    """Whatever the environment, backend='energy' picks the M0 detector."""
    samples = utterance(MODEL_SAMPLE_RATE)
    energy_interval, snr = detect_speech(samples, MODEL_SAMPLE_RATE, vad_backend="energy")
    assert energy_interval is not None
    assert energy_interval[0] == pytest.approx(0.3, abs=0.1)
    assert snr is not None and snr > 20


def test_auto_backend_falls_back_to_energy_without_the_ml_extra():
    """CI installs no ml extra, so 'auto' must behave exactly like 'energy'."""
    samples = utterance(MODEL_SAMPLE_RATE)
    interval, _ = detect_speech(samples, MODEL_SAMPLE_RATE, vad_backend="auto")
    assert interval is not None
    assert interval[0] == pytest.approx(0.3, abs=0.1)


def test_silero_backend_refuses_when_silero_is_unavailable():
    """Forcing silero without the runtime must refuse (None), not silently
    hand back another detector's answer."""
    samples = utterance(MODEL_SAMPLE_RATE)
    interval, _ = detect_speech(samples, MODEL_SAMPLE_RATE, vad_backend="silero")
    assert interval is None


def test_silero_interval_is_used_when_it_loads(monkeypatch):
    from openschwa_engine.audio import preprocess

    monkeypatch.setattr(preprocess, "_silero_interval", lambda samples: (0.31, 0.88))
    samples = utterance(MODEL_SAMPLE_RATE)
    interval, _ = detect_speech(samples, MODEL_SAMPLE_RATE, vad_backend="auto")
    assert interval is not None
    assert interval[0] == pytest.approx(0.31 - preprocess._PAD_S, abs=0.01)
    assert interval[1] == pytest.approx(0.88 + preprocess._PAD_S, abs=0.01)
