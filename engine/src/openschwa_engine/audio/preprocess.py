"""Resample, trim to speech, and judge recording quality.

Quality checks exist to protect the confidence gate: a clipped or near-silent
recording produces confident-looking garbage downstream, so it is caught here
and turned into a "retry" rather than a verdict.

VAD sits behind detect_speech. M1 runs silero-vad (the neural detector) when
the ml extra is installed and its model is cached, and falls back to the
energy-hysteresis detector everywhere else - a degraded engine must still trim
speech. The energy path matters on its own: it is what keeps CI, packaged
builds without torch, and first-run-before-download working.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import soxr

log = logging.getLogger(__name__)

MODEL_SAMPLE_RATE = 16_000

_FRAME_S = 0.025
_HOP_S = 0.010
_PAD_S = 0.08  # keep a little context around the detected region
_MIN_SPEECH_S = 0.08

# A recording with less than this spread between its quiet and loud frames has
# no usable speech/background separation.
_MIN_DYNAMIC_RANGE_DB = 6.0

# Hysteresis. Voiceless fricatives sit 20-30 dB below the vowel beside them, so
# a single threshold high enough to find the word reliably clips the /s/ off the
# end of it. The high threshold locates the utterance; the low one extends the
# boundaries out through the quiet edges.
#
# The asymmetry is deliberate: trimming real speech makes alignment fail, while
# a little extra silence costs nothing — CTC absorbs it as leading and trailing
# blanks.
_CORE_THRESHOLD_FRACTION = 0.45
_CORE_THRESHOLD_MIN_DB = 8.0
_EDGE_THRESHOLD_FRACTION = 0.15
_EDGE_THRESHOLD_MIN_DB = 3.0
_CLIP_LEVEL = 0.995
_CLIP_FRACTION = 0.001  # 0.1% of samples at full scale is audible distortion

# Absolute level is close to meaningless as a usability test, and gating on it
# was a real bug: browsers that honour `autoGainControl: false` — which this app
# requires — hand back a raw capture commonly 15-25 dB below what a consumer
# recorder produces. That audio is perfectly analysable; its SNR is unchanged
# and the acoustic model normalises its input to zero mean and unit variance
# before it sees anything. Rejecting it told users to speak up when speaking up
# could not possibly help.
#
# So this floor now marks a *dead* input — an unplugged or muted device — and
# nothing else. Speech that is merely quiet is analysed, and whether the result
# can be trusted is decided downstream by alignment confidence, which measures
# it directly.
_DEAD_INPUT_DBFS = -70.0

#: Not a gate. The UI surfaces `speech_level_dbfs` below this as advice.
LOW_LEVEL_ADVISORY_DBFS = -40.0


def _dbfs(amplitude: float) -> float:
    return 20.0 * float(np.log10(max(amplitude, 1e-10)))


@dataclass(frozen=True)
class QualityReport:
    clipping: bool
    too_quiet: bool
    snr_db_est: float | None
    speech_level_dbfs: float | None = None
    peak_dbfs: float | None = None

    @property
    def usable(self) -> bool:
        return not (self.clipping or self.too_quiet)

    @property
    def low_level(self) -> bool:
        """Quiet enough to be worth mentioning, not enough to refuse."""
        return self.speech_level_dbfs is not None and (
            self.speech_level_dbfs < LOW_LEVEL_ADVISORY_DBFS
        )


@dataclass(frozen=True)
class PreparedAudio:
    """Everything downstream stages need, with times in the original timeline."""

    samples_16k: npt.NDArray[np.float32]
    duration_s: float
    sample_rate: int
    speech_interval_s: tuple[float, float] | None
    quality: QualityReport

    @property
    def speech_16k(self) -> npt.NDArray[np.float32]:
        """The trimmed region the model actually sees."""
        if self.speech_interval_s is None:
            return self.samples_16k
        start, end = self.speech_interval_s
        return self.samples_16k[int(start * MODEL_SAMPLE_RATE) : int(end * MODEL_SAMPLE_RATE)]

    @property
    def speech_offset_s(self) -> float:
        """Added to model-relative times to get back to the upload timeline."""
        return self.speech_interval_s[0] if self.speech_interval_s else 0.0


def resample_to_model_rate(
    samples: npt.NDArray[np.float32], sample_rate: int
) -> npt.NDArray[np.float32]:
    if sample_rate == MODEL_SAMPLE_RATE:
        return samples
    resampled = soxr.resample(samples, sample_rate, MODEL_SAMPLE_RATE, quality="HQ")
    return np.ascontiguousarray(resampled, dtype=np.float32)


def _frame_rms_db(samples: npt.NDArray[np.float32], sample_rate: int) -> npt.NDArray[np.float64]:
    frame = int(_FRAME_S * sample_rate)
    hop = int(_HOP_S * sample_rate)
    if len(samples) < frame:
        return np.array([], dtype=np.float64)
    count = 1 + (len(samples) - frame) // hop
    strided = np.lib.stride_tricks.as_strided(
        samples,
        shape=(count, frame),
        strides=(samples.strides[0] * hop, samples.strides[0]),
    )
    rms = np.sqrt(np.mean(np.square(strided, dtype=np.float64), axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-10))


def _energy_interval(
    samples: npt.NDArray[np.float32], sample_rate: int
) -> tuple[tuple[float, float] | None, float | None]:
    """The energy-hysteresis detector: (interval, snr) or (None, snr)."""
    db = _frame_rms_db(samples, sample_rate)
    if db.size == 0:
        return None, None

    # 5th rather than 10th percentile: a short drill recording is mostly speech,
    # so a higher quantile starts measuring the utterance instead of the room.
    floor_db = float(np.percentile(db, 5))
    peak_db = float(np.percentile(db, 95))
    dynamic_range = peak_db - floor_db
    snr = round(dynamic_range, 1)

    if dynamic_range < _MIN_DYNAMIC_RANGE_DB:
        return None, snr

    core_threshold = floor_db + max(
        _CORE_THRESHOLD_FRACTION * dynamic_range, _CORE_THRESHOLD_MIN_DB
    )
    core = np.flatnonzero(db > core_threshold)
    if core.size == 0:
        return None, snr

    edge_threshold = floor_db + max(
        _EDGE_THRESHOLD_FRACTION * dynamic_range, _EDGE_THRESHOLD_MIN_DB
    )
    above_edge = db > edge_threshold
    first, last = int(core[0]), int(core[-1])
    while first > 0 and above_edge[first - 1]:
        first -= 1
    while last < len(db) - 1 and above_edge[last + 1]:
        last += 1

    start_s = max(0.0, first * _HOP_S - _PAD_S)
    end_s = min(len(samples) / sample_rate, last * _HOP_S + _FRAME_S + _PAD_S)
    if end_s - start_s < _MIN_SPEECH_S:
        return None, snr
    return (round(start_s, 4), round(end_s, 4)), snr


_silero_runtime = None
_silero_failed = False


def _silero_model_or_none() -> tuple[Any, Any, Any] | None:
    """The cached silero runtime (model, get_speech_timestamps, torch), or None.

    Loaded once per process; every failure disables the path for the process's
    lifetime, because retrying a broken download per analysis would add a
    network timeout to every recording.
    """
    global _silero_runtime, _silero_failed
    if _silero_failed:
        return None
    if _silero_runtime is not None:
        return _silero_runtime
    try:
        import torch  # noqa: PLC0415 - ml extra
        from silero_vad import (  # noqa: PLC0415 - ml extra
            get_speech_timestamps,
            load_silero_vad,
        )
    except ImportError:
        _silero_failed = True
        return None
    try:
        # silero_vad ships no type information (mypy override in pyproject).
        model = load_silero_vad()
        _silero_runtime = (model, get_speech_timestamps, torch)
        return _silero_runtime
    except Exception as exc:  # noqa: BLE001 - degrade, never crash an analysis
        _silero_failed = True
        log.warning("silero VAD unavailable (%s) - falling back to the energy detector", exc)
        return None


def _silero_interval(samples_16k: npt.NDArray[np.float32]) -> tuple[float, float] | None:
    """Silero's first-to-last speech span, in seconds."""
    runtime = _silero_model_or_none()
    if runtime is None:
        return None
    model, get_speech_timestamps, torch = runtime
    try:
        audio = torch.from_numpy(np.ascontiguousarray(samples_16k)).float()
        chunks = get_speech_timestamps(audio, model, return_seconds=True)
        if not chunks:
            return None
        return float(chunks[0]["start"]), float(chunks[-1]["end"])
    except Exception as exc:  # noqa: BLE001 - the energy path still works
        _silero_failed = True
        log.warning("silero VAD failed on this input (%s) - falling back to energy", exc)
        return None


def detect_speech(
    samples: npt.NDArray[np.float32],
    sample_rate: int,
    vad_backend: str = "auto",
) -> tuple[tuple[float, float] | None, float | None]:
    """Return ((start_s, end_s) or None, estimated SNR in dB or None).

    The interval spans the first to the last speech frame - interior pauses are
    kept, because a drill utterance is scored as one unit and dropping internal
    silence would corrupt the phone timeline.

    vad_backend: "auto" (silero when it loads, else energy), "silero", or
    "energy". The SNR estimate always comes from the energy analysis; silero
    contributes only the interval.
    """
    energy_interval, snr = _energy_interval(samples, sample_rate)
    if vad_backend == "energy":
        return energy_interval, snr
    if vad_backend in ("auto", "silero") and sample_rate == MODEL_SAMPLE_RATE:
        silero = _silero_interval(samples)
        if silero is not None:
            start, end = silero
            start = max(0.0, start - _PAD_S)
            end = min(len(samples) / MODEL_SAMPLE_RATE, end + _PAD_S)
            if end - start >= _MIN_SPEECH_S:
                return (round(start, 4), round(end, 4)), snr
    if vad_backend == "silero":
        return None, snr  # asked for silero, got nothing: refuse rather than lie
    return energy_interval, snr


def assess_quality(
    samples: npt.NDArray[np.float32],
    speech: npt.NDArray[np.float32],
    snr_db: float | None,
) -> QualityReport:
    clipped = int(np.count_nonzero(np.abs(samples) >= _CLIP_LEVEL))
    clipping = clipped > max(1, int(_CLIP_FRACTION * samples.size))

    region = speech if speech.size else samples
    rms = float(np.sqrt(np.mean(np.square(region, dtype=np.float64)))) if region.size else 0.0
    level_dbfs = _dbfs(rms)
    peak_dbfs = _dbfs(float(np.abs(samples).max())) if samples.size else _dbfs(0.0)

    # bool() rather than the numpy scalars these comparisons produce: these end
    # up in the JSON contract, and np.bool_ is not the `bool` pydantic declares.
    return QualityReport(
        clipping=bool(clipping),
        too_quiet=bool(level_dbfs < _DEAD_INPUT_DBFS),
        snr_db_est=snr_db,
        speech_level_dbfs=round(level_dbfs, 1),
        peak_dbfs=round(peak_dbfs, 1),
    )


def prepare(
    samples: npt.NDArray[np.float32], sample_rate: int, vad_backend: str = "auto"
) -> PreparedAudio:
    """Decoded audio -> model-rate signal, speech interval, and quality flags."""
    duration_s = len(samples) / sample_rate
    samples_16k = resample_to_model_rate(samples, sample_rate)
    interval, snr = detect_speech(samples_16k, MODEL_SAMPLE_RATE, vad_backend=vad_backend)

    speech = samples_16k
    if interval is not None:
        speech = samples_16k[
            int(interval[0] * MODEL_SAMPLE_RATE) : int(interval[1] * MODEL_SAMPLE_RATE)
        ]

    return PreparedAudio(
        samples_16k=samples_16k,
        duration_s=duration_s,
        sample_rate=sample_rate,
        speech_interval_s=interval,
        quality=assess_quality(samples_16k, speech, snr),
    )
