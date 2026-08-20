"""Intonation exam aggregation on synthetic records (no audio)."""

import pytest

from openschwa_eval.datasets.aixmarsec import IntonationUnit
from openschwa_eval.intonation import IntonationRecord, evaluate_intonation, split_blocks


def _unit(block):
    return IntonationUnit(
        passage_id=f"{block}01G",
        block_id=block,
        annotator="G",
        audio_path=None,
        start_s=0.0,
        end_s=1.0,
        expected_tone="fall",
        transcript="x",
        phones=("ð", "e"),
    )


def test_split_blocks_is_disjoint_and_seeded():
    units = [_unit(block) for block in (f"A{i:02d}" for i in range(1, 12))]
    split = split_blocks(units, 42)
    again = split_blocks(units, 42)
    assert split == again
    roles = set(split.values())
    assert roles == {"train", "cal", "test"}
    # every block in exactly one split (dict, by construction) and test is non-empty
    assert sum(1 for v in split.values() if v == "test") >= 2


def test_bar_math_is_plain_accuracy_and_octave_rate():
    split = {"A01": "test", "A02": "test", "A03": "train"}
    records = [
        IntonationRecord(
            "A0101G",
            "A01",
            "G",
            "fall",
            detected="fall",
            match=True,
            octave_error_rate=0.01,
            alignment="ok",
        ),
        IntonationRecord(
            "A0102G",
            "A01",
            "G",
            "rise",
            detected="rise",
            match=True,
            octave_error_rate=0.01,
            alignment="ok",
        ),
        IntonationRecord(
            "A0201B",
            "A02",
            "B",
            "fall",
            detected="rise",
            match=False,
            octave_error_rate=0.03,
            alignment="ok",
        ),
        IntonationRecord(
            "A0301G",
            "A03",
            "G",
            "fall",
            detected="fall",
            match=True,
            octave_error_rate=0.0,
            alignment="ok",
        ),
    ]
    summary = evaluate_intonation(records, split)
    assert summary["fall_rise_units"] == 3
    assert summary["fall_rise_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["mean_octave_error_rate"] == pytest.approx(0.05 / 3, abs=1e-4)
    assert summary["bar"]["met"] is False  # accuracy 0.67 < 0.90


def test_perfect_records_pass_the_bar():
    split = {"A01": "test"}
    records = [
        IntonationRecord(
            "A0101G",
            "A01",
            "G",
            "fall",
            detected="fall",
            match=True,
            octave_error_rate=0.005,
            alignment="ok",
        ),
        IntonationRecord(
            "A0102G",
            "A01",
            "G",
            "rise",
            detected="rise",
            match=True,
            octave_error_rate=0.005,
            alignment="ok",
        ),
    ]
    summary = evaluate_intonation(records, split)
    assert summary["bar"]["met"] is True
