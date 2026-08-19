"""Hard-negative mining: selection logic and export shape."""

import csv
import json
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

from openschwa_engine.alignment import AlignedPhone

import openschwa_training.export_hard_negatives as hardneg


def _record(utt, idx, score, label, split):
    return {
        "utterance_id": utt,
        "token_index": idx,
        "score": score,
        "label": label,
        "split": split,
    }


def test_select_keeps_top_train_correct_only():
    records = [
        _record("u1", 0, 0.9, "correct", "train"),
        _record("u2", 1, 1.8, "correct", "train"),
        _record("u3", 2, 2.5, "correct", "test"),  # test pool: never mined
        _record("u4", 3, 3.0, "substituted", "train"),  # not correct
        _record("u5", 4, 0.1, "correct", "train"),
        _record("u6", 5, None, "correct", "train"),  # unscored
    ]
    picks = hardneg.select_hard_negatives(records, top_k=3)
    assert [p["score"] for p in picks] == [1.8, 0.9, 0.1]
    assert all(p["label"] == "correct" and p["split"] == "train" for p in picks)


def _tiny_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"".join(struct.pack("<h", 4000) for _ in range(32_000)))


def test_export_writes_correct_labeled_segments(tmp_path, monkeypatch):
    wav = tmp_path / "u.wav"
    _tiny_wav(wav)
    token = SimpleNamespace(index=3)
    utterance = SimpleNamespace(
        utterance_id="u1", audio_path=wav, l1="mandarin", transcript="THE",
    )
    utterance.tokens = lambda target: [token]
    utterance.phones = [token]
    adapter = SimpleNamespace(utterances=lambda target: [utterance])
    monkeypatch.setattr(hardneg, "L2Arctic", lambda root: adapter)
    monkeypatch.setattr(hardneg, "SpeechOcean762", lambda root: adapter)
    monkeypatch.setattr(
        hardneg,
        "_align_phones",
        lambda u, p, r, s: {
            3: AlignedPhone(
                index=3, label="ð", start_s=0.1, end_s=0.25, gop=-0.1, confidence=0.9
            )
        },
    )
    checkpoint = tmp_path / "ck.jsonl"
    checkpoint.write_text(
        json.dumps(_record("u1", 3, 2.0, "correct", "train")) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    manifest = hardneg.export_hard_negatives(
        hardneg.HardNegOptions(
            checkpoint=checkpoint,
            l2arctic_root=tmp_path,
            so762_root=tmp_path,
            out_dir=out,
            top_k=5,
        )
    )
    rows = list(csv.DictReader((out / "labels.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["label"] == "ð"
    assert rows[0]["split"] == "train"
    assert manifest["rows"] == 1