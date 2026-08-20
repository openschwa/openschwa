"""M2 intonation exam over controlled recordings (the app's real use case).

Each recording is a deliberate tone group (e.g. "Please." vs "Please?")
whose expected tone is defined by the prompt, not by a human label. The
engine's shipped prosody chain runs on every recording: prepare -> track ->
nuclear_tone. Detected-vs-expected mismatches are NOT charged to the engine
until a human has verified the recording actually carries the intended tone
(the flag list is printed for that pass). The bar is the roadmap's own:
fall-vs-rise accuracy >= 0.90 and octave-error rate < 2% of voiced frames.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from openschwa_engine.audio import decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.prosody import nuclear_tone, track

from openschwa_eval.intonation import FALL_RISE_ACCURACY_BAR, OCTAVE_ERROR_RATE_BAR

#: Formats afconvert turns into 16 kHz mono WAV on the fly.
CONVERTIBLE = {".m4a", ".mov", ".caf", ".mp3", ".aiff", ".aif", ".m4b"}


@dataclass
class RecordingRecord:
    item_id: str
    rep: int
    text: str
    expected_tone: str
    status: str = "missing"  # missing | ok | unvoiced
    detected: str | None = None
    confidence: float | None = None
    octave_error_rate: float | None = None
    duration_s: float = 0.0


def _wav_for(path: Path) -> Path:
    """Return a 16 kHz mono WAV for any recording afconvert understands."""
    if path.suffix.lower() == ".wav":
        return path
    out = path.with_suffix(".wav")
    subprocess.run(
        [
            "afconvert",
            "-f",
            "WAVE",
            "-d",
            "LEI16@16000",
            "-c",
            "1",
            str(path),
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def run_recordings(
    manifest: dict,
    audio_dir: Path,
    exclude: set[str] | None = None,
) -> list[RecordingRecord]:
    """Run the shipped prosody chain on every item x rep recording.

    exclude holds "<id>_<rep>" keys the human verification pass dropped
    (mis-spoken takes): they are recorded as excluded and never scored.
    """
    settings = Settings(warm_model_on_start=False)
    records: list[RecordingRecord] = []
    reps = manifest.get("reps", 1)
    for item in manifest["items"]:
        for rep in range(1, reps + 1):
            if exclude and f"{item['id']}_{rep}" in exclude:
                records.append(
                    RecordingRecord(
                        item["id"], rep, item["text"], item["tone"], status="excluded"
                    )
                )
                continue
            audio_path = next(audio_dir.glob(f"{item['id']}_{rep}.*"), None)
            if audio_path is None:
                records.append(
                    RecordingRecord(item["id"], rep, item["text"], item["tone"])
                )
                continue
            wav = _wav_for(audio_path)
            decoded = decode_wav(wav.read_bytes())
            prepared = prepare(
                decoded.samples, decoded.sample_rate, vad_backend=settings.vad_backend
            )
            f0 = track(prepared.samples_16k, 16_000)
            if f0 is None or f0.semitones is None or all(v is None for v in f0.semitones):
                records.append(
                    RecordingRecord(
                        item["id"], rep, item["text"], item["tone"], status="unvoiced"
                    )
                )
                continue
            speech_end = (
                prepared.speech_interval_s[1] if prepared.speech_interval_s else None
            )
            tone, confidence = nuclear_tone(f0, end_s=speech_end)
            voiced = sum(1 for value in f0.semitones if value is not None)
            records.append(
                RecordingRecord(
                    item_id=item["id"],
                    rep=rep,
                    text=item["text"],
                    expected_tone=item["tone"],
                    status="ok",
                    detected=tone,
                    confidence=round(confidence, 3),
                    octave_error_rate=(
                        round(f0.octave_error_frames / voiced, 4) if voiced else None
                    ),
                    duration_s=round(prepared.duration_s, 3),
                )
            )
    return records


def evaluate_recordings(records: list[RecordingRecord]) -> dict:
    """Per-tone accuracy, the bar, and the human-verification flag list."""
    ok = [r for r in records if r.status == "ok"]
    fall_rise = [r for r in ok if r.expected_tone in ("fall", "rise")]
    correct = sum(1 for r in fall_rise if r.detected == r.expected_tone)
    accuracy = correct / len(fall_rise) if fall_rise else 0.0
    per_tone = {
        tone: {
            "n": len(group),
            "accuracy": round(
                sum(1 for r in group if r.detected == r.expected_tone) / len(group), 4
            ),
        }
        for tone, group in [
            (tone, [r for r in ok if r.expected_tone == tone])
            for tone in ("fall", "rise", "fall_rise", "level")
        ]
        if group
    }
    octave_rates = [r.octave_error_rate for r in ok if r.octave_error_rate is not None]
    mean_octave = sum(octave_rates) / len(octave_rates) if octave_rates else None
    flags = [
        {
            "item_id": r.item_id,
            "rep": r.rep,
            "text": r.text,
            "expected": r.expected_tone,
            "detected": r.detected,
            "confidence": r.confidence,
        }
        for r in records
        if r.status != "ok" or r.detected != r.expected_tone
    ]
    return {
        "recordings": len(records),
        "usable": len(ok),
        "fall_rise_units": len(fall_rise),
        "fall_rise_accuracy": round(accuracy, 4),
        "mean_octave_error_rate": (
            round(mean_octave, 5) if mean_octave is not None else None
        ),
        "per_tone": per_tone,
        "bar": {
            "fall_rise_accuracy_met": accuracy >= FALL_RISE_ACCURACY_BAR,
            "octave_error_rate_met": (
                mean_octave is not None and mean_octave < OCTAVE_ERROR_RATE_BAR
            ),
            "met": accuracy >= FALL_RISE_ACCURACY_BAR
            and (mean_octave is not None and mean_octave < OCTAVE_ERROR_RATE_BAR),
        },
        "flags_for_verification": flags,
    }


def records_jsonl(records: list[RecordingRecord]) -> str:
    """Per-recording JSONL for the report artifacts."""
    return "\n".join(json.dumps(asdict(r)) for r in records) + "\n"
