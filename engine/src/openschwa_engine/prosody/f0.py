"""F0 tracking, octave-jump cleanup, and conversion to semitones.

Praat (via parselmouth) does the tracking; this module adds the two things that
make a contour comparable across speakers and safe to draw:

1. **Two-pass floor/ceiling.** A single fixed 75–600 Hz range is the main source
   of octave errors — too wide for a low voice, too narrow for a child's. The
   first pass estimates the speaker's range, the second re-tracks inside it.
   This is Praat's own recommended recipe.
2. **Semitones relative to the speaker's median.** Absolute hertz cannot be
   compared between a learner and a teacher of a different sex; semitones re
   median can. `docs/architecture.md` makes the semitone domain the contract.

M0 renders the learner's own contour. Reference comparison, DTW, and nuclear
tone classification are M2 — the shapes they need are already in the contract.
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)

_TIME_STEP_S = 0.01
_SEARCH_FLOOR_HZ = 60.0
_SEARCH_CEILING_HZ = 800.0
_ABSOLUTE_FLOOR_HZ = 50.0
_ABSOLUTE_CEILING_HZ = 900.0

#: A jump larger than this from the local median is treated as a tracking error
#: rather than real intonation — a seventh inside one 10 ms hop is not speech.
_OCTAVE_JUMP_SEMITONES = 9.0
_MEDIAN_WINDOW = 7


@dataclass(frozen=True)
class F0Track:
    hop_s: float
    start_s: float
    semitones: tuple[float | None, ...]
    median_hz: float | None

    @property
    def voiced_fraction(self) -> float:
        if not self.semitones:
            return 0.0
        return sum(1 for s in self.semitones if s is not None) / len(self.semitones)


def _to_semitones(hz: npt.NDArray[np.float64], median: float) -> npt.NDArray[np.float64]:
    return 12.0 * np.log2(np.maximum(hz, 1e-6) / median)


def _rolling_median(values: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """Median over a centred window, ignoring NaN (unvoiced) neighbours.

    A window falling entirely inside an unvoiced stretch has no median; that is
    expected, not exceptional, so the all-NaN warning is silenced and the NaN
    result propagates (those frames stay unvoiced).
    """
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(windows, axis=1)


def _fix_octave_jumps(hz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Halve or double isolated frames that sit an octave off their neighbours.

    Frames that cannot be rescued into the local range are marked unvoiced: a
    gap in the contour is honest, a wrong pitch point is not.
    """
    voiced = hz > 0
    if voiced.sum() < 3:
        return hz

    working = np.where(voiced, hz, np.nan)
    local = _rolling_median(working, _MEDIAN_WINDOW)
    result = working.copy()

    for factor in (2.0, 0.5):
        deviation = np.abs(12.0 * np.log2(result / local))
        candidates = np.isfinite(deviation) & (deviation > _OCTAVE_JUMP_SEMITONES)
        if not candidates.any():
            break
        corrected = result * factor
        improved = candidates & (np.abs(12.0 * np.log2(corrected / local)) < _OCTAVE_JUMP_SEMITONES)
        result = np.where(improved, corrected, result)

    deviation = np.abs(12.0 * np.log2(result / local))
    unrescuable = np.isfinite(deviation) & (deviation > _OCTAVE_JUMP_SEMITONES)
    if unrescuable.any():
        log.debug("dropped %d unrecoverable F0 frames", int(unrescuable.sum()))
    result[unrescuable] = np.nan
    return np.where(np.isfinite(result), result, 0.0)


def track(samples: npt.NDArray[np.float32], sample_rate: int) -> F0Track | None:
    """Extract a cleaned F0 contour, or None when the clip has no usable pitch."""
    try:
        import parselmouth  # noqa: PLC0415 - keeps import cost off engine startup
    except ImportError:  # pragma: no cover - parselmouth is a base dependency
        log.warning("parselmouth unavailable; skipping prosody")
        return None

    if samples.size < int(0.05 * sample_rate):
        return None

    sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=sample_rate)

    coarse = sound.to_pitch(
        time_step=_TIME_STEP_S, pitch_floor=_SEARCH_FLOOR_HZ, pitch_ceiling=_SEARCH_CEILING_HZ
    )
    coarse_hz = np.asarray(coarse.selected_array["frequency"], dtype=np.float64)
    voiced_hz = coarse_hz[coarse_hz > 0]
    if voiced_hz.size < 3:
        return None

    # Hirst's two-pass recipe: re-track inside the speaker's own range.
    low, high = np.percentile(voiced_hz, [5, 95])
    floor = float(np.clip(0.75 * low, _ABSOLUTE_FLOOR_HZ, _ABSOLUTE_CEILING_HZ - 1))
    ceiling = float(np.clip(1.5 * high, floor + 50.0, _ABSOLUTE_CEILING_HZ))

    pitch = sound.to_pitch(time_step=_TIME_STEP_S, pitch_floor=floor, pitch_ceiling=ceiling)
    hz = _fix_octave_jumps(np.asarray(pitch.selected_array["frequency"], dtype=np.float64))

    voiced = hz > 0
    if voiced.sum() < 3:
        return None

    median = float(np.median(hz[voiced]))
    semitone_values = _to_semitones(hz, median)
    times = np.asarray(pitch.xs(), dtype=np.float64)

    return F0Track(
        hop_s=_TIME_STEP_S,
        start_s=round(float(times[0]) if times.size else 0.0, 4),
        semitones=tuple(
            round(float(value), 3) if is_voiced else None
            for value, is_voiced in zip(semitone_values, voiced, strict=True)
        ),
        median_hz=round(median, 2),
    )
