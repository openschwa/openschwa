"""Reference-contour extraction: the teacher track the UI overlays (M2)."""

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openschwa_engine.prosody import reference_track


def _chirp_wav(path: Path, f0_start: float = 120.0, f0_end: float = 220.0) -> Path:
    """A 0.6 s rising chirp: Praat must track it as a rising contour."""
    sample_rate = 16_000
    duration = 0.6
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate
    phase = 2 * np.pi * (f0_start * t + (f0_end - f0_start) * t**2 / (2 * duration))
    samples = (0.5 * np.sin(phase) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return path


def test_reference_track_reads_a_rising_reference(tmp_path):
    wav = _chirp_wav(tmp_path / "ref.wav")
    exercise = SimpleNamespace(has_reference_audio=True, reference_audio_path=wav)
    track = reference_track(exercise)
    assert track is not None
    voiced = [s for s in track.semitones if s is not None]
    assert len(voiced) > 20  # a real contour, not a blip
    # A 120->220 Hz chirp rises about one octave: the semitone contour must
    # end clearly higher than it starts.
    assert voiced[-1] - voiced[0] > 6.0


def test_reference_track_is_none_without_audio(tmp_path):
    exercise = SimpleNamespace(has_reference_audio=False, reference_audio_path=None)
    assert reference_track(exercise) is None