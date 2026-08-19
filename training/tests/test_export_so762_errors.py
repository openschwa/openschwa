"""so762 error export: expert-bracketed tokens, realized-phone labels."""

import csv

import numpy as np
from openschwa_engine.alignment import AlignedPhone
from openschwa_engine.models.phone_set import PhoneMap

import openschwa_training.export_so762_errors as err

IDX = {"ð": 2, "z": 3, "d": 4, "v": 5}


def test_realized_phone_reads_the_dominant_mass():
    probs = np.zeros((4, 6))
    probs[:, 2] = 0.05  # ð
    probs[:, 3] = 0.85  # z
    probs[:, 4] = 0.05  # d
    probs[:, 5] = 0.05  # v
    logp = np.log(probs).astype(np.float32)
    assert err._realized_phone(logp, np.arange(4), IDX) == "z"


def test_realized_phone_requires_min_class_mass():
    probs = np.zeros((2, 6))
    probs[:, 2] = 0.45
    probs[:, 3] = 0.55
    logp = np.log(probs).astype(np.float32)
    assert err._realized_phone(logp, np.arange(2), IDX, min_class_mass=0.6) is None
    assert err._realized_phone(logp, np.arange(2), IDX, min_class_mass=0.5) == "z"


def test_realized_phone_reports_target_when_the_acoustics_are_target():
    # Experts heard an error the acoustics do not show: the caller skips it.
    probs = np.zeros((3, 6))
    probs[:, 2] = 0.9
    logp = np.log(probs).astype(np.float32)
    assert err._realized_phone(logp, np.arange(3), IDX) == "ð"


def test_empty_frames_return_none():
    assert err._realized_phone(np.zeros((0, 6), np.float32), [], IDX) is None


def test_error_exporter_keeps_bracketed_tokens_with_realized_labels(tmp_path, monkeypatch):
    from test_export_so762 import corpus

    class FakeRegistry:
        def spec(self, _):
            return object()

        def phone_map(self, _):
            return PhoneMap(
                model_id="fake",
                table_name="dhz_en",
                token_of={"ð": "ð", "z": "z", "d": "d", "v": "v"},
                index_of=IDX,
                blank_index=0,
            )

    monkeypatch.setattr(err, "ModelRegistry", lambda model_dir: FakeRegistry())
    probs = np.zeros((4, 6))
    probs[:, 3] = 0.8  # z-dominant
    probs[:, 2] = 0.2
    logp = np.log(probs).astype(np.float32)
    monkeypatch.setattr(
        err,
        "_align_with_posteriors",
        lambda u, p, r, s: (
            {
                0: AlignedPhone(
                    index=0,
                    label="ð",
                    start_s=0.1,
                    end_s=0.25,
                    gop=-0.1,
                    confidence=0.9,
                    frame_indices=(0, 1, 2, 3),
                )
            },
            logp,
        ),
    )
    out = tmp_path / "errs"
    manifest = err.export_so762_errors(
        err.ErrorExportOptions(so762_root=corpus(tmp_path), out_dir=out)
    )
    rows = list(csv.DictReader((out / "labels.csv").open(encoding="utf-8")))
    # Token 0 of 000010035 is expert-bracketed -> kept with the realized z label;
    # token 0 of 000010011 is clean -> skipped.
    assert len(rows) == 1
    assert rows[0]["label"] == "z"
    assert rows[0]["l1"] == "mandarin"
    assert manifest["rows"] == 1