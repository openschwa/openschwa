"""The recordings exam: manifest, runner plumbing, and the report shape."""

import json
import wave
from pathlib import Path

import numpy as np

from openschwa_eval.recordings import (
    RecordingRecord,
    evaluate_recordings,
    run_recordings,
)

MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "intonation-recordings.json"


def _write_wav(path: Path, seconds: float = 1.0, hz: float = 200.0) -> None:
    sample_rate = 16_000
    samples = 0.3 * np.sin(2 * np.pi * hz * np.arange(int(seconds * sample_rate)) / sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((samples * 32767).astype("<i2").tobytes())


def test_manifest_is_complete_and_balanced():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = manifest["items"]
    assert manifest["reps"] == 2
    assert len(items) == 68
    fall_rise = [i for i in items if i["tone"] in ("fall", "rise")]
    # the bar's ~100 utterances: 48 pairs x 2 reps = 96
    assert len(fall_rise) == 48
    assert len(fall_rise) * manifest["reps"] >= 90
    by_tone = {
        tone: sum(1 for i in items if i["tone"] == tone)
        for tone in ("fall", "rise", "fall_rise", "level")
    }
    assert by_tone["fall"] == 24 and by_tone["rise"] == 24
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    for i in items:
        assert i["text"].strip() and i["cue"].strip()


def test_runner_scores_a_tone_recording_and_flags_missing(tmp_path):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["reps"] = 1
    manifest["items"] = manifest["items"][:2]  # fr-01a, fr-01b
    audio = tmp_path / "audio"
    audio.mkdir()
    _write_wav(audio / "fr-01a_1.wav")  # present
    # fr-01b_1 deliberately missing
    records = run_recordings(manifest, audio)
    assert len(records) == 2
    present, missing = records
    assert present.status == "ok" and present.detected is not None
    assert missing.status == "missing" and missing.detected is None
    summary = evaluate_recordings(records)
    assert summary["recordings"] == 2
    assert summary["usable"] == 1
    assert set(summary["bar"]) == {"fall_rise_accuracy_met", "octave_error_rate_met", "met"}
    assert any(f["item_id"] == "fr-01b" for f in summary["flags_for_verification"])


def test_evaluate_flags_mismatches_for_human_verification():
    records = [
        RecordingRecord("a", 1, "Please.", "fall", status="ok", detected="fall", confidence=1.0),
        RecordingRecord("b", 1, "Please?", "rise", status="ok", detected="level", confidence=0.2),
        RecordingRecord("c", 1, "No.", "fall", status="unvoiced"),
    ]
    summary = evaluate_recordings(records)
    assert summary["fall_rise_accuracy"] == 0.5  # a matches, b does not
    flags = {f["item_id"]: f for f in summary["flags_for_verification"]}
    assert set(flags) == {"b", "c"}


def test_evaluate_handles_empty_ok_set():
    summary = evaluate_recordings([RecordingRecord("a", 1, "x", "fall")])
    assert summary["fall_rise_accuracy"] == 0.0
    assert summary["mean_octave_error_rate"] is None
    assert summary["bar"]["met"] is False


def test_recorder_plan_and_save(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "record_intonation_set",
        Path(__file__).resolve().parents[1] / "record_intonation_set.py",
    )
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)

    # regression: the CLI defaults must resolve inside eval/, not the repo root
    assert recorder.DEFAULT_MANIFEST.is_file()
    expected_audio = Path(__file__).resolve().parents[1] / "data" / "recordings"
    assert recorder.DEFAULT_AUDIO_DIR == expected_audio

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    steps = recorder.plan(manifest, tmp_path)
    assert len(steps) == 136  # 68 items x 2 reps
    assert all(exists is False for _i, _r, _p, exists in steps)
    assert steps[0][2].name == "fr-01a_1.wav"
    assert steps[-1][2].name == "lv-08_2.wav"

    samples = np.zeros(1600, dtype=np.int16)
    recorder.save_wav(tmp_path / "fr-01a_1.wav", samples, 16_000)
    assert (tmp_path / "fr-01a_1.wav").is_file()
    steps = recorder.plan(manifest, tmp_path)
    assert steps[0][3] is True  # resume: recorded takes are detected
    with wave.open(str(tmp_path / "fr-01a_1.wav")) as handle:
        assert handle.getframerate() == 16_000
        assert handle.getnchannels() == 1
        assert handle.getnframes() == 1600

