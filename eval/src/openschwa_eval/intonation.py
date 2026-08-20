"""M2 intonation exam: engine nuclear-tone verdicts vs SEC tonetic labels.

Each Aix-MARSEC unit (audio span + expected tone from the human TSM)
runs through the full engine pipeline; the engine's terminal-tone
classifier produces (detected, confidence) and the octave-error rate
comes from the F0 track. The bar is the roadmap's own: fall-vs-rise
accuracy >= 0.90 AND octave-error rate < 2% of frames, on a
block-disjoint TEST split only (a block is one speaker's recording -
the speaker-leakage lesson from M1 applies here from day one).
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from openschwa_engine.audio import decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.content.loader import Exercise, PhoneSpec, ProsodySpec
from openschwa_engine.models.registry import ModelRegistry
from openschwa_engine.pipeline import analyze_recording

from openschwa_eval.datasets.aixmarsec import IntonationUnit

FALL_RISE_ACCURACY_BAR = 0.90
OCTAVE_ERROR_RATE_BAR = 0.02


@dataclass
class IntonationRecord:
    passage_id: str
    block_id: str
    annotator: str
    expected_tone: str
    detected: str | None = None
    confidence: float | None = None
    match: bool | None = None
    octave_error_rate: float | None = None
    alignment: str = "skipped"
    wall_ms: float = 0.0
    duration_s: float = 0.0


def _exercise_for(unit: IntonationUnit) -> Exercise | None:
    if len(unit.phones) < 2:
        return None  # not enough to align; the tone cannot be trusted either
    return Exercise(
        id=f"aix-{unit.passage_id}-{unit.start_s:.1f}",
        pack_id="aix",
        type="sentence",
        title="",
        lang="en",
        text=unit.transcript,
        ipa="",
        phones=tuple(PhoneSpec(index=i, ph=phone) for i, phone in enumerate(unit.phones)),
        source_path=Path("aix"),
        prosody=ProsodySpec(nuclear_syllable_index=0, expected_tone=unit.expected_tone),
    )


def run_units(
    units: list[IntonationUnit],
    registry: ModelRegistry,
    settings: Settings,
    checkpoint: Path | None = None,
) -> list[IntonationRecord]:
    """Run every unit through the engine, checkpointing as it goes."""
    records: list[IntonationRecord] = []
    done: set[tuple[str, float]] = set()
    if checkpoint is not None and checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = IntonationRecord(**json.loads(line))
            records.append(record)
            done.add((record.passage_id, record.duration_s))
    block_cache: dict[Path, object] = {}
    for position, unit in enumerate(units):
        key = (unit.passage_id, round(unit.start_s, 2))
        if key in done:
            continue
        exercise = _exercise_for(unit)
        if exercise is None:
            records.append(
                IntonationRecord(unit.passage_id, unit.block_id, unit.annotator, unit.expected_tone)
            )
            _append_checkpoint(checkpoint, records[-1])
            continue
        decoded = block_cache.get(unit.audio_path)
        if decoded is None:
            decoded = decode_wav(unit.audio_path.read_bytes())
            block_cache[unit.audio_path] = decoded
        rate = decoded.sample_rate
        # The tone lives in the nuclear stretch: from the TSM-marked nucleus
        # to the unit end. The pre-nuclear head carries no tone and only
        # dilutes the terminal window (the smoke run measured exactly that:
        # the nucleus is often mid-unit, and the tail after it is level).
        start = max(0, int(unit.nucleus_s * rate) - int(0.1 * rate))
        end = min(len(decoded.samples), int(unit.end_s * rate) + int(0.05 * rate))
        if end - start < int(0.3 * rate):
            records.append(
                IntonationRecord(unit.passage_id, unit.block_id, unit.annotator, unit.expected_tone)
            )
            _append_checkpoint(checkpoint, records[-1])
            continue
        prepared = prepare(decoded.samples[start:end], rate, vad_backend=settings.vad_backend)
        started = time.perf_counter()
        analysis = analyze_recording(prepared, exercise, registry, settings, include_ungated=True)
        wall_ms = (time.perf_counter() - started) * 1000.0
        prosody = analysis.prosody
        nuclear = prosody.nuclear_tone if prosody is not None else None
        record = IntonationRecord(
            passage_id=unit.passage_id,
            block_id=unit.block_id,
            annotator=unit.annotator,
            expected_tone=unit.expected_tone,
            detected=nuclear.detected if nuclear is not None else None,
            confidence=nuclear.confidence if nuclear is not None else None,
            match=nuclear.match if nuclear is not None else None,
            octave_error_rate=(prosody.f0.octave_error_rate if prosody is not None else None),
            alignment=analysis.alignment.status,
            wall_ms=round(wall_ms, 2),
            duration_s=round((end - start) / rate, 3),
        )
        records.append(record)
        _append_checkpoint(checkpoint, record)
        if (position + 1) % 100 == 0:
            print(f"processed {position + 1} units", flush=True)
    return records


def _append_checkpoint(checkpoint: Path | None, record: IntonationRecord) -> None:
    if checkpoint is None:
        return
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def split_blocks(units: list[IntonationUnit], seed: int) -> dict[str, str]:
    """Block -> train | cal | test, seeded. A block is one speaker's recording."""
    blocks = sorted({unit.block_id for unit in units})
    rng = random.Random(f"{seed}:intonation-blocks")
    rng.shuffle(blocks)
    n_test = max(1, len(blocks) // 5)
    n_cal = max(1, len(blocks) // 5)
    split: dict[str, str] = {}
    for index, block in enumerate(blocks):
        if index < n_test:
            split[block] = "test"
        elif index < n_test + n_cal:
            split[block] = "cal"
        else:
            split[block] = "train"
    return split


def evaluate_intonation(
    records: list[IntonationRecord], split: dict[str, str]
) -> dict[str, object]:
    """Fall-vs-rise accuracy and octave-error rate on the TEST split, plus the bar."""
    test = [r for r in records if split.get(r.block_id) == "test" and r.detected is not None]
    fall_rise = [r for r in test if r.expected_tone in ("fall", "rise")]
    correct = sum(1 for r in fall_rise if r.detected == r.expected_tone)
    accuracy = correct / len(fall_rise) if fall_rise else 0.0
    octave_rates = [r.octave_error_rate for r in test if r.octave_error_rate is not None]
    mean_octave = sum(octave_rates) / len(octave_rates) if octave_rates else None
    per_tone = {
        tone: {
            "n": len(group),
            "accuracy": round(
                sum(1 for r in group if r.detected == r.expected_tone) / len(group), 4
            ),
        }
        for tone, group in [
            (tone, [r for r in test if r.expected_tone == tone])
            for tone in ("fall", "rise", "fall_rise")
        ]
        if group
    }
    per_annotator = {
        annotator: {
            "n": len(group),
            "accuracy": round(
                sum(1 for r in group if r.detected == r.expected_tone) / len(group), 4
            ),
        }
        for annotator, group in [
            (a, [r for r in fall_rise if r.annotator == a]) for a in ("B", "G")
        ]
        if group
    }
    bar_accuracy = accuracy >= FALL_RISE_ACCURACY_BAR
    bar_octave = mean_octave is not None and mean_octave < OCTAVE_ERROR_RATE_BAR
    return {
        "units_run": len(records),
        "test_units": len(test),
        "fall_rise_units": len(fall_rise),
        "fall_rise_accuracy": round(accuracy, 4),
        "mean_octave_error_rate": round(mean_octave, 5) if mean_octave is not None else None,
        "per_tone": per_tone,
        "per_annotator": per_annotator,
        "alignment": dict(Counter(r.alignment for r in records)),
        "bar": {
            "fall_rise_accuracy_met": bar_accuracy,
            "octave_error_rate_met": bar_octave,
            "met": bar_accuracy and bar_octave,
        },
    }
