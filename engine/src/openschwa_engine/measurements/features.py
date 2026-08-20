"""DSP acoustic-phonetic feature extraction for /ð/ vs {z, d, v} discrimination.

Extracts 10 deterministic physical cues designed to eliminate false positives on
reduced dental approximants [ð̞] and vowel coarticulation:

1. rms_db: Normalized segment loudness.
2. norm_zcr: Zero-crossing rate (high for /z/, low for /ð/).
3. closure_dip: Minimum 10ms frame RMS relative to segment mean (deep for /d/ stop closure).
4. burst_sharpness: Max frame-to-frame positive RMS gradient (abrupt for /d/ release burst).
5. burst_contrast: Ratio of max onset RMS to min closure RMS (high for stop /d/).
6. sibilance_prominence: Power in 5–8 kHz band vs 1–3.5 kHz formant band (isolates /z/ friction).
7. norm_centroid: Spectral center of mass frequency.
8. voicing_continuity: Max normalized autocorrelation in 70–400 Hz range (voicing periodicity).
9. norm_rolloff: 85% spectral energy boundary.
10. spectral_flatness: Wiener entropy (diffuse for /ð/, peaked for /z/).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def extract_acoustic_features(samples_16k: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    """Extract a 10-dimensional normalized acoustic feature vector from 16 kHz mono audio.

    Returns float32 array of shape [10].
    """
    if samples_16k.size == 0:
        return np.zeros(10, dtype=np.float32)

    samples = samples_16k.astype(np.float64)
    n_samples = len(samples)

    # 1. Total RMS energy
    rms = np.sqrt(np.mean(samples**2) + EPS)
    rms_db = float(np.clip(20.0 * np.log10(rms + EPS), -80.0, 0.0) / 80.0 + 1.0)

    # 2. Zero crossing rate
    zcr = float(np.mean(np.abs(np.diff(np.signbit(samples)))) * (sample_rate / 1000.0))
    norm_zcr = float(np.clip(zcr / 16.0, 0.0, 1.0))

    # 3. Short-time framing for closure dip, burst sharpness, and burst contrast
    frame_len = int(0.010 * sample_rate)  # 160 samples (10 ms)
    hop_len = int(0.005 * sample_rate)  # 80 samples (5 ms)
    if n_samples >= frame_len:
        num_frames = (n_samples - frame_len) // hop_len + 1
        frame_rms = np.array(
            [
                np.sqrt(np.mean(samples[i * hop_len : i * hop_len + frame_len] ** 2) + EPS)
                for i in range(num_frames)
            ]
        )
        min_rms = float(np.min(frame_rms))
        mean_rms = float(np.mean(frame_rms) + EPS)
        max_rms = float(np.max(frame_rms))

        closure_dip = float(np.clip(np.log10((min_rms + EPS) / (mean_rms + EPS)), -3.0, 0.0) / -3.0)
        burst_contrast = float(np.clip(np.log10((max_rms + EPS) / (min_rms + EPS)), 0.0, 3.0) / 3.0)

        if num_frames > 1:
            diffs = np.diff(frame_rms)
            burst = float(np.max(diffs) / (mean_rms + EPS))
            burst_sharpness = float(np.clip(burst / 3.0, 0.0, 1.0))
        else:
            burst_sharpness = 0.0
    else:
        closure_dip = 0.0
        burst_contrast = 0.0
        burst_sharpness = 0.0

    # 4. Voicing continuity: Autocorrelation periodicity (pitch range 70 - 400 Hz)
    min_lag = int(sample_rate / 400)  # 40 samples
    max_lag = int(sample_rate / 70)  # 228 samples
    if n_samples > max_lag:
        center = samples - np.mean(samples)
        autocorr = np.correlate(center, center, mode="full")
        mid = len(autocorr) // 2
        r0 = autocorr[mid] + EPS
        norm_ac = autocorr[mid + min_lag : mid + max_lag] / r0
        voicing_continuity = float(np.clip(np.max(norm_ac), 0.0, 1.0))
    else:
        voicing_continuity = 0.5

    # 5. Spectral analysis via FFT
    windowed = samples * np.hanning(n_samples)
    fft_vals = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    power = fft_vals**2 + EPS
    total_power = np.sum(power)

    # 6. Spectral Centroid
    centroid = float(np.sum(freqs * power) / total_power)
    norm_centroid = float(np.clip(centroid / 8000.0, 0.0, 1.0))

    # 7. Sibilance Prominence: Power in 5000–8000 Hz vs 1000–3500 Hz (formant band)
    sib_mask = (freqs >= 5000.0) & (freqs <= 8000.0)
    formant_mask = (freqs >= 1000.0) & (freqs <= 3500.0)
    sib_power = np.sum(power[sib_mask]) + EPS
    formant_power = np.sum(power[formant_mask]) + EPS
    sibilance_prominence = float(
        np.clip(np.log10(sib_power / formant_power), -2.5, 1.5) / 4.0 + 0.625
    )

    # 8. Spectral Rolloff (85% power threshold)
    cumsum_power = np.cumsum(power)
    rolloff_idx = int(np.searchsorted(cumsum_power, 0.85 * total_power))
    rolloff_freq = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    norm_rolloff = float(np.clip(rolloff_freq / 8000.0, 0.0, 1.0))

    # 9. Spectral Flatness: geometric mean / arithmetic mean
    log_power = np.log(power)
    geom_mean = np.exp(np.mean(log_power))
    arith_mean = np.mean(power)
    flatness = float(np.clip(geom_mean / arith_mean, 0.0, 1.0))

    features = np.array(
        [
            rms_db,
            norm_zcr,
            closure_dip,
            burst_sharpness,
            burst_contrast,
            sibilance_prominence,
            norm_centroid,
            voicing_continuity,
            norm_rolloff,
            flatness,
        ],
        dtype=np.float32,
    )
    return features
