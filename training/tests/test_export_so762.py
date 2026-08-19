"""so762 export: unanimous-correct only, aligner intervals, test split untouched."""

import csv
import json
import wave
from pathlib import Path

from openschwa_engine.alignment import AlignedPhone

import openschwa_training.export_so762 as exporter

SCORES = {
    "000010011": {"words": [{"text": "THE", "phones": ["DH", "AH0"]}]},
    "000010035": {"words": [{"text": "THE", "phones": ["DH", "AH0"]}]},
}
# 3/5 experts bracket phone 0 of word 0 for the second utterance -> not unanimous
DETAIL = {
    "000010011": {"words": [{"phones": ["DH AH0"] * 5}]},
    "000010035": {"words": [{"phones": ["{DH} AH0"] * 1 + ["DH AH0"] * 4}]},
}


def _tiny_wav(path) -> None:
    import struct

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        frames = b"".join(struct.pack("<h", int(12000 * (i % 50) / 50)) for i in range(32_000))
        handle.writeframes(frames)


def corpus(tmp_path) -> Path:
    root = tmp_path / "so762"
    (root / "resource").mkdir(parents=True)
    wav_dir = root / "WAVE" / "SPEAKER0001"
    wav_dir.mkdir(parents=True)
    (root / "train").mkdir()
    (root / "test").mkdir()
    (root / "resource" / "scores.json").write_text(json.dumps(SCORES), encoding="utf-8")
    (root / "resource" / "scores-detail.json").write_text(json.dumps(DETAIL), encoding="utf-8")
    for utt in ("000010011", "000010035"):
        _tiny_wav(wav_dir / f"{utt}.WAV")
    (root / "train" / "wav.scp").write_text(
        "000010011 WAVE/SPEAKER0001/000010011.WAV\n000010035 WAVE/SPEAKER0001/000010035.WAV\n",
        encoding="utf-8",
    )
    (root / "train" / "text").write_text("000010011 THE\n000010035 THE\n", encoding="utf-8")
    # The adapter requires both split files; the test split is empty.
    (root / "test" / "wav.scp").write_text("", encoding="utf-8")
    (root / "test" / "text").write_text("", encoding="utf-8")
    return root


def test_exports_only_unanimous_correct_tokens(tmp_path, monkeypatch):
    """One expert bracket (1/5) must disqualify a token: only 5/5-agreement
    tokens are clean training negatives."""
    monkeypatch.setattr(
        exporter,
        "_align_for_export",
        lambda u, p, r, s: {
            0: AlignedPhone(index=0, label="ð", start_s=0.1, end_s=0.25, gop=-0.1, confidence=0.9)
        },
    )
    out = tmp_path / "out"
    manifest = exporter.export_so762(
        exporter.So762Options(so762_root=corpus(tmp_path), out_dir=out, max_tokens=None)
    )
    rows = list(csv.DictReader((out / "labels.csv").open(encoding="utf-8")))
    # Both utterances have a token 0, but the second one carries an expert
    # bracket -> only the first is kept.
    assert len(rows) == 1
    assert rows[0]["label"] == "ð"
    assert rows[0]["l1"] == "mandarin"
    assert manifest["skipped"]["expert-bracketed"] == 1
    with wave.open(str(out / rows[0]["filename"]), "rb") as handle:
        assert handle.getframerate() == 16_000


def test_test_split_is_never_exported(tmp_path, monkeypatch):
    root = corpus(tmp_path)
    # Move the only train utterances to the test split: nothing may be exported.
    (root / "test" / "wav.scp").write_text(
        (root / "train" / "wav.scp").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "train" / "wav.scp").write_text("", encoding="utf-8")
    (root / "test" / "text").write_text(
        (root / "train" / "text").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "train" / "text").write_text("", encoding="utf-8")
    out = tmp_path / "out"
    manifest = exporter.export_so762(
        exporter.So762Options(so762_root=root, out_dir=out, max_tokens=None)
    )
    assert manifest["rows"] == 0
