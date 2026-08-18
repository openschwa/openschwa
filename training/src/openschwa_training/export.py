"""Export labeled L2-ARCTIC segments for contrast fine-tuning.

Only the harness's TRAIN split is exported: the held-out split is the exam and
must stay blind. Tokens of each of the four target phones become short 16 kHz
WAVs plus rows in labels.csv; labels are the 4-class CTC alphabet {ð, z, d, v}.
Correct tokens of a target feed their own phone's class, and substituted /ð/
tokens feed the realized phone's class. Tokens whose realization is outside
the alphabet (t, θ, s, ..., deletions, unknowns) are skipped: training on them
would be training on mislabels.

The output is corpus-derived audio, so it never enters git (training/data/ is
ignored); manifest.json records the provenance of every run.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openschwa_engine.audio import MODEL_SAMPLE_RATE, decode_wav, resample_to_model_rate
from openschwa_eval.datasets import L2Arctic, PhoneToken, Utterance
from openschwa_eval.harness import assign_split

ALPHABET = ("ð", "z", "d", "v")
PAD_S = 0.05  # coarticulation context on each side of the annotated interval


class ExportError(RuntimeError):
    """The export cannot proceed without corrupting the dataset."""


@dataclass(frozen=True)
class ExportOptions:
    l2arctic_root: Path
    out_dir: Path
    split_seed: int = 42
    val_fraction: float = 0.15
    pad_s: float = PAD_S
    max_per_class: int | None = None


def _class_label(token: PhoneToken) -> str | None:
    """The CTC target for a token, or None when it must be skipped.

    For a correct token of any target phone the realization IS the target, so
    it feeds that phone's class. A substituted token feeds the REALIZED phone's
    class: the model must learn what each of the four phones sounds like, and a
    /ð/ realized as /z/ is, acoustically, a /z/. Out-of-alphabet realizations
    and deletions are skipped - training on them would be training on mislabels.
    """
    if token.label == "correct":
        return token.phone
    if token.label == "substituted" and token.substituted_with in ALPHABET:
        return token.substituted_with
    return None  # deletions, out-of-alphabet realizations, unknowns


def _utterance_is_val(utterance: Utterance, seed: int, val_fraction: float) -> bool:
    """Whole-utterance val assignment: segments of one utterance must never
    straddle the train/val boundary, or the val score would flatter us."""
    rng = random.Random(f"{seed}:val:{utterance.utterance_id}")
    return rng.random() < val_fraction


def _cap_per_class(rows: list[dict[str, object]], cap: int, seed: int) -> list[dict[str, object]]:
    """Deterministic per-class downsampling for class balance."""
    by_class: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_class.setdefault(str(row["label"]), []).append(row)
    kept: list[dict[str, object]] = []
    for label in sorted(by_class):
        group = sorted(by_class[label], key=lambda row: str(row["filename"]))
        rng = random.Random(f"{seed}:cap:{label}")
        kept.extend(rng.sample(group, min(cap, len(group))))
    return kept


def _recount(rows: list[dict[str, object]], alphabet: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {label: 0 for label in alphabet}
    for row in rows:
        counts[str(row["label"])] += 1
    return counts


def _write_segment_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(MODEL_SAMPLE_RATE)
        handle.writeframes(np.clip(samples, -1.0, 1.0).astype("<i2").tobytes())


def export(options: ExportOptions) -> dict[str, object]:
    """Export the train split; returns the manifest (also written to disk)."""
    audio_dir = options.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    adapter = L2Arctic(options.l2arctic_root)

    skipped: dict[str, int] = {}
    counts: dict[str, int] = {label: 0 for label in ALPHABET}
    split_counts: dict[str, int] = {"train": 0, "val": 0}
    rows: list[dict[str, object]] = []

    # Every class needs acoustic examples of ITS phone: correct tokens of each
    # of the four target phones feed their own class, and substituted /ð/
    # tokens feed the realized phone's class (a /ð/ heard as /z/ IS a /z/).
    # Without this, the v class would have zero examples: L2-ARCTIC's
    # annotators never realized /ð/ as /v/.
    for target in ALPHABET:
        for utterance in adapter.utterances(target):
            if assign_split(utterance, options.split_seed) == "test":
                continue  # the exam: never exported, never trained on
            is_val = _utterance_is_val(utterance, options.split_seed, options.val_fraction)
            decoded = None
            for token in utterance.tokens(target):
                label = _class_label(token)
                if label is None:
                    reason = (
                        token.label
                        if token.label != "substituted"
                        else (f"realized /{token.substituted_with}/")
                    )
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue
                if token.start_s is None or token.end_s is None:
                    raise ExportError(
                        f"{utterance.utterance_id}: token {token.index} has no interval "
                        "- refusing to guess boundaries"
                    )
                if decoded is None:
                    decoded = decode_wav(utterance.audio_path.read_bytes())
                start = max(0.0, token.start_s - options.pad_s)
                end = min(decoded.duration_s, token.end_s + options.pad_s)
                if end - start <= 0.01:
                    raise ExportError(
                        f"{utterance.utterance_id}: token {token.index} interval is empty"
                    )
                samples = decoded.samples[
                    int(start * decoded.sample_rate) : int(end * decoded.sample_rate)
                ]
                samples_16k = resample_to_model_rate(samples, decoded.sample_rate)
                stem = f"{utterance.utterance_id}_{token.index}"
                _write_segment_wav(audio_dir / f"{stem}.wav", samples_16k)
                split = "val" if is_val else "train"
                rows.append(
                    {
                        "filename": f"audio/{stem}.wav",
                        "label": label,
                        "l1": utterance.l1,
                        "utterance_id": utterance.utterance_id,
                        "token_index": token.index,
                        "target_phone": target,
                        "start_s": round(token.start_s, 4),
                        "end_s": round(token.end_s, 4),
                        "duration_s": round(len(samples_16k) / MODEL_SAMPLE_RATE, 4),
                        "split": split,
                    }
                )
                counts[label] += 1
                split_counts[split] += 1

    if options.max_per_class is not None:
        rows = _cap_per_class(rows, options.max_per_class, options.split_seed)
        counts = _recount(rows, ALPHABET)

    fieldnames = [
        "filename",
        "label",
        "l1",
        "utterance_id",
        "token_index",
        "target_phone",
        "start_s",
        "end_s",
        "duration_s",
        "split",
    ]
    rows.sort(key=lambda row: (str(row["split"]), str(row["filename"])))
    with (options.out_dir / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest: dict[str, object] = {
        "corpus": "L2-ARCTIC",
        "alphabet": list(ALPHABET),
        "split_seed": options.split_seed,
        "val_fraction": options.val_fraction,
        "pad_s": options.pad_s,
        "max_per_class": options.max_per_class,
        "class_counts": counts,
        "split_counts": split_counts,
        "skipped": skipped,
        "note": "train split only; the eval harness's held-out split was never exported",
    }
    (options.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2arctic", required=True, help="local L2-ARCTIC corpus root")
    parser.add_argument(
        "--out", required=True, help="export directory (audio/ + labels.csv + manifest.json)"
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--pad-s", type=float, default=PAD_S)
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()
    manifest = export(
        ExportOptions(
            l2arctic_root=Path(args.l2arctic),
            out_dir=Path(args.out),
            split_seed=args.split_seed,
            val_fraction=args.val_fraction,
            pad_s=args.pad_s,
            max_per_class=args.max_per_class,
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
