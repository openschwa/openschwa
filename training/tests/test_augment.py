"""Augmentation and hard-negative weighting: the two wall-fighters."""

import random

import numpy as np

from openschwa_training.train import augment_segment


def test_augment_preserves_length_and_signal_energy():
    rng = random.Random(42)
    tone = (0.3 * np.sin(2 * np.pi * 400 * np.arange(8000) / 16000)).astype(np.float32)
    out = augment_segment(tone, rng)
    assert len(out) == len(tone)
    assert np.sqrt(np.mean(out**2)) > 0.05  # still real signal, not silence


def test_augment_is_seeded_deterministic():
    base = np.random.RandomState(0).randn(8000).astype(np.float32)
    a = augment_segment(base, random.Random(7))
    b = augment_segment(base, random.Random(7))
    assert np.array_equal(a, b)


def test_augment_never_explodes():
    rng = random.Random(1)
    for _ in range(50):
        out = augment_segment(np.random.RandomState(3).randn(4000).astype(np.float32), rng)
        assert float(np.abs(out).max()) < 6.0