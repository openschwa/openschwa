"""Export correctness: alphabet filtering, held-out blindness, val splits."""

import csv
import json
import wave
from pathlib import Path

from openschwa_eval.datasets import Utterance
from openschwa_eval.datasets.l2arctic import SPEAKER_L1
from openschwa_eval.harness import assign_split

from openschwa_training.export import ExportOptions, export

ANNOTATION = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.6
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.6
        intervals: size = 1
            intervals [1]:
                xmin = 0.1
                xmax = 1.4
                text = "then this"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.8
        intervals: size = 11
            intervals [1]:
                xmin = 0
                xmax = 0.1
                text = "sil"
            intervals [2]:
                xmin = 0.1
                xmax = 0.25
                text = "DH"
            intervals [3]:
                xmin = 0.25
                xmax = 0.45
                text = "DH,Z,s"
            intervals [4]:
                xmin = 0.45
                xmax = 0.7
                text = "EH1"
            intervals [5]:
                xmin = 0.7
                xmax = 0.9
                text = "DH,T,s"
            intervals [6]:
                xmin = 0.9
                xmax = 1.1
                text = "N"
            intervals [7]:
                xmin = 1.1
                xmax = 1.3
                text = "DH,D,d"
            intervals [8]:
                xmin = 1.3
                xmax = 1.4
                text = "Z"
            intervals [9]:
                xmin = 1.4
                xmax = 1.5
                text = "V"
            intervals [10]:
                xmin = 1.5
                xmax = 1.6
                text = "D"
            intervals [11]:
                xmin = 1.6
                xmax = 1.8
                text = "sp"
"""


def _tiny_wav(path) -> None:
    import struct

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44_100)
        # 2 seconds: the fixture intervals reach 1.8 s.
        frames = b"".join(struct.pack("<h", int(12000 * (i % 50) / 50)) for i in range(88_200))
        handle.writeframes(frames)


def corpus(tmp_path, stem="arctic_a0001", speaker_name="ABA") -> Path:
    root = tmp_path / "l2arctic"
    speaker = root / speaker_name
    (speaker / "wav").mkdir(parents=True)
    (speaker / "transcript").mkdir()
    (speaker / "annotation").mkdir()
    _tiny_wav(speaker / "wav" / f"{stem}.wav")
    (speaker / "transcript" / f"{stem}.txt").write_text("then this\n", encoding="utf-8")
    (speaker / "annotation" / f"{stem}.TextGrid").write_text(ANNOTATION, encoding="utf-8")
    return root


def _speaker_with_role(role: str) -> str:
    """A speaker whose three-way split role is role under seed 42.

    The split is speaker-disjoint, so every utterance of that speaker lands
    in the same role - stems no longer matter for split selection.
    """
    for speaker, l1 in SPEAKER_L1.items():
        utterance = Utterance(
            f"l2arctic-{speaker}-x", Path("x"), "", l1, (), corpus="l2arctic", speaker=speaker
        )
        if assign_split(utterance, 42) == role:
            return speaker
    raise AssertionError(f"no speaker with role {role!r}")


def _train_stem() -> str:
    return "arctic_a0001"


def _test_speaker() -> str:
    return _speaker_with_role("test")


def test_exports_only_alphabet_classes(tmp_path):
    out = tmp_path / "out"
    manifest = export(
        ExportOptions(
            corpus(tmp_path, _train_stem(), _speaker_with_role("train")), out, val_fraction=0.0
        )
    )
    rows = list(csv.DictReader((out / "labels.csv").open(encoding="utf-8")))
    labels = sorted(r["label"] for r in rows)
    # DH correct -> ð; DH,Z,s -> z; Z -> z; V -> v; D -> d.
    assert labels == ["d", "v", "z", "z", "ð"]
    assert manifest["class_counts"] == {"ð": 1, "z": 2, "d": 1, "v": 1}
    assert manifest["skipped"]["realized /t/"] == 1
    assert manifest["skipped"]["deleted"] == 1


def test_held_out_utterances_are_never_exported(tmp_path):
    out = tmp_path / "out"
    manifest = export(
        ExportOptions(corpus(tmp_path, _train_stem(), _test_speaker()), out, val_fraction=0.0)
    )
    assert manifest["class_counts"] == {"ð": 0, "z": 0, "d": 0, "v": 0}
    assert sum(1 for _ in (out / "labels.csv").open()) == 1  # header only


def test_calibration_pool_utterances_are_never_exported(tmp_path):
    """The cal carve is the harness's threshold-fitting pool: like the test
    split, it must never leak into training data (Stage 1 lockstep)."""
    cal_speaker = next(
        s
        for s, l1 in SPEAKER_L1.items()
        if assign_split(
            Utterance(f"l2arctic-{s}-x", Path("x"), "", l1, (), corpus="l2arctic", speaker=s),
            42,
        )
        == "cal"
    )
    out = tmp_path / "out"
    manifest = export(
        ExportOptions(corpus(tmp_path, speaker_name=cal_speaker), out, val_fraction=0.0)
    )
    assert manifest["class_counts"] == {"ð": 0, "z": 0, "d": 0, "v": 0}
    assert sum(1 for _ in (out / "labels.csv").open()) == 1  # header only


def test_val_assignment_is_per_utterance(tmp_path):
    out = tmp_path / "out"
    manifest = export(
        ExportOptions(
            corpus(tmp_path, _train_stem(), _speaker_with_role("train")),
            out,
            val_fraction=1.0,
        )
    )
    assert manifest["split_counts"] == {"train": 0, "val": 5}
    rows = list(csv.DictReader((out / "labels.csv").open(encoding="utf-8")))
    assert all(r["split"] == "val" for r in rows)


def test_max_per_class_caps_every_class(tmp_path):
    out = tmp_path / "out"
    manifest = export(
        ExportOptions(
            corpus(tmp_path, _train_stem(), _speaker_with_role("train")),
            out,
            val_fraction=0.0,
            max_per_class=1,
        )
    )
    assert manifest["class_counts"] == {"ð": 1, "z": 1, "d": 1, "v": 1}
    rows = list(csv.DictReader((out / "labels.csv").open(encoding="utf-8")))
    assert len(rows) == 4


def test_exported_wavs_are_16k_mono_with_real_energy(tmp_path):
    """The fixture audio is a sawtooth, so exported segments must carry real
    energy: the int16 scale bug shipped silent training data once already."""
    import numpy as np

    out = tmp_path / "out"
    export(
        ExportOptions(
            corpus(tmp_path, _train_stem(), _speaker_with_role("train")), out, val_fraction=0.0
        )
    )
    rows = list(csv.DictReader((out / "labels.csv").open(encoding="utf-8")))
    assert len(rows) == 5
    with wave.open(str(out / rows[0]["filename"]), "rb") as handle:
        assert handle.getframerate() == 16_000
        assert handle.getnchannels() == 1
        assert handle.getnframes() > 0
        frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    assert float(np.abs(frames).mean()) > 100  # not silence


def test_manifest_records_provenance(tmp_path):
    out = tmp_path / "out"
    export(
        ExportOptions(
            corpus(tmp_path, _train_stem(), _speaker_with_role("train")), out, val_fraction=0.0
        )
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus"] == "L2-ARCTIC"
    assert manifest["split_seed"] == 42
    assert "held-out" in manifest["note"]
