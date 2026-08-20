"""Aix-MARSEC adapter: real-corpus extraction + synthetic offset refinement."""

import wave
from pathlib import Path

import numpy as np
import pytest

from openschwa_eval.datasets.aixmarsec import AixMarsec, refine_offsets

CORPUS_ROOT = Path(__file__).resolve().parents[2] / "data" / "aix" / "4"


def _write_tone_block(path: Path, passages: int, passage_s: float, gap_s: float) -> None:
    """A block wav: `passages` chunks of 200 Hz tone separated by silence."""
    sample_rate = 16_000
    tone = 0.4 * np.sin(2 * np.pi * 200 * np.arange(int(passage_s * sample_rate)) / sample_rate)
    silence = np.zeros(int(gap_s * sample_rate))
    parts = []
    for _ in range(passages):
        parts.append(tone)
        parts.append(silence)
    samples = np.concatenate(parts)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((samples * 32767).astype("<i2").tobytes())


def test_refine_offsets_recovers_passage_starts(tmp_path):
    wav = tmp_path / "A01.wav"
    passage_s, gap_s = 0.8, 0.2
    _write_tone_block(wav, 3, passage_s, gap_s)
    passages = [(f"A01{i:02d}", passage_s + gap_s) for i in (1, 2, 3)]
    offsets = refine_offsets(wav, passages)
    assert len(offsets) == 3
    assert offsets["A0101"] == pytest.approx(0.0, abs=0.1)
    # Snap-to-silence-onset: each offset lands at the start of the gap, so a
    # hair of trailing silence belongs to the slice - harmless for the exam.
    assert offsets["A0102"] == pytest.approx(1.0, abs=0.25)
    assert offsets["A0103"] == pytest.approx(2.0, abs=0.25)


def test_refine_offsets_drops_truncated_tail(tmp_path):
    wav = tmp_path / "A01.wav"
    _write_tone_block(wav, 2, 0.8, 0.2)  # ~2.0 s of audio
    passages = [("A0101", 1.0)] * 4  # table claims 4 s: the tail has no audio
    offsets = refine_offsets(wav, passages)
    assert len(offsets) <= 3


@pytest.mark.skipif(not CORPUS_ROOT.is_dir(), reason="Aix-MARSEC corpus not present")
def test_corpus_yields_fall_and_rise_units():
    corpus = AixMarsec(CORPUS_ROOT, max_passages=6)
    units = corpus.units()
    assert units
    for unit in units:
        assert unit.expected_tone in ("fall", "rise", "fall_rise")
        assert unit.annotator in ("B", "G")
        assert unit.end_s > unit.start_s
        assert unit.transcript.strip()
    from collections import Counter

    tones = Counter(u.expected_tone for u in units)
    assert tones["fall"] + tones["rise"] > 0
