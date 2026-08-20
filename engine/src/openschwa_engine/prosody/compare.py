"""M2 intonation measurements: nuclear-tone classification, DTW contour
matching, and octave-error accounting. All deterministic DSP over the
F0Track - no models, nothing to train. Thresholds are module constants the
eval harness fits on the calibration pool before any verdict ships.
"""

from __future__ import annotations

import math

import numpy as np

from openschwa_engine.prosody.f0 import F0Track

#: Terminal movement is the nuclear tone for short drills: the final voiced
#: stretch carries the fall or rise the drill is about.
TERMINAL_S = 0.35
#: Minimum absolute slope (semitones/second) to call a fall or a rise.
SLOPE_THRESHOLD_ST_S = 8.0

Tone = str  # fall | rise | fall_rise | level


def _voiced_series(
    track: F0Track, start_s: float = 0.0, end_s: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(times, semitones) of voiced frames in [start_s, end_s]."""
    times: list[float] = []
    values: list[float] = []
    for index, value in enumerate(track.semitones):
        if value is None:
            continue
        t = track.start_s + index * track.hop_s
        if t >= start_s - 1e-6 and (end_s is None or t <= end_s + 1e-6):
            times.append(t)
            values.append(value)
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _slope(times: np.ndarray, values: np.ndarray) -> float | None:
    """Linear-regression slope in semitones/second, or None when flat."""
    if times.size < 3 or np.ptp(times) < 1e-3:
        return None
    centered = times - times.mean()
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        return None
    return float(np.sum(centered * values) / denominator)


def nuclear_tone(
    track: F0Track, terminal_s: float = TERMINAL_S, end_s: float | None = None
) -> tuple[Tone, float]:
    """(tone, confidence) from the terminal voiced movement.

    The slope of the final voiced stretch decides: a steep terminal glide is
    a fall or a rise; a gentle one is level; a fall followed by a rise
    (first half falling, second half rising) is a fall-rise. Confidence is
    the slope's margin over the threshold, clamped to [0, 1]; unmeasurable
    contours report level with zero confidence.

    end_s (when given) ends the terminal window at the end of *speech*
    instead of the end of the track - recordings with trailing silence
    must not have the tone read from that silence.
    """
    duration = track.start_s + (len(track.semitones) - 1) * track.hop_s
    end = min(duration, end_s) if end_s is not None else duration
    # The window ends at the last *voiced* frame at or before end: weak
    # final consonants (the /z/ of "please") leave an unvoiced tail after
    # the glide that would otherwise swallow the terminal window.
    last_voiced = None
    for index, value in enumerate(track.semitones):
        if value is None:
            continue
        t = track.start_s + index * track.hop_s
        if t <= end + 1e-6:
            last_voiced = t
        else:
            break
    if last_voiced is not None:
        end = min(end, last_voiced)
    window_start = max(0.0, end - terminal_s)
    times, values = _voiced_series(track, window_start, end)
    slope = _slope(times, values)
    if slope is None or values.size < 3:
        return "level", 0.0
    # Fall-rise check: split the terminal window in half.
    mid = float(times[0]) + (float(times[-1]) - float(times[0])) / 2.0
    first = values[times <= mid]
    second = values[times > mid]
    first_slope = _slope(times[times <= mid], first) if first.size >= 3 else None
    second_slope = _slope(times[times > mid], second) if second.size >= 3 else None
    if (
        first_slope is not None
        and second_slope is not None
        and first_slope <= -SLOPE_THRESHOLD_ST_S
        and second_slope >= SLOPE_THRESHOLD_ST_S
    ):
        return "fall_rise", 1.0
    confidence = min(1.0, abs(slope) / (2 * SLOPE_THRESHOLD_ST_S))
    if slope >= SLOPE_THRESHOLD_ST_S:
        return "rise", confidence
    if slope <= -SLOPE_THRESHOLD_ST_S:
        return "fall", confidence
    return "level", confidence


def dtw_distance(learner: F0Track, reference: F0Track) -> float | None:
    """Per-frame-normalized DTW distance over the voiced contours.

    Voiced semitone sequences only (unvoiced frames are gaps, not pitch),
    standard DTW with squared-error local cost, normalized by the warp path
    length so long and short recordings compare. None when either side has
    fewer than 3 voiced frames.
    """
    _, a = _voiced_series(learner)
    _, b = _voiced_series(reference)
    if a.size < 3 or b.size < 3:
        return None
    cost = np.zeros((a.size + 1, b.size + 1))
    cost[0, 1:] = np.inf
    cost[1:, 0] = np.inf
    for i in range(1, a.size + 1):
        for j in range(1, b.size + 1):
            local = (a[i - 1] - b[j - 1]) ** 2
            cost[i, j] = local + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    path_length = a.size + b.size
    return float(cost[a.size, b.size] / path_length)


def contour_match(learner: F0Track, reference: F0Track, scale_st: float = 4.0) -> float | None:
    """DTW distance mapped to [0, 1]: 1 = identical contour, 0 = far apart.

    The scale (in semitones) is the one fitted constant; the harness tunes
    it on the calibration pool. None when the distance is unmeasurable.
    """
    distance = dtw_distance(learner, reference)
    if distance is None:
        return None
    return float(math.exp(-(distance**0.5) / scale_st))