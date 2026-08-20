"""Export labeled L2-ARCTIC segments for contrast fine-tuning.

Only the harness's TRAIN split is exported: the cal pool feeds the threshold
fit and the test pool is the exam - both stay blind. The open-set alphabet is
{ð, z, d, other}: correct tokens of the drilled phones feed their own class,
correct tokens of a widened source set (θ, s, t, f, l, v) feed "other" so the
model learns what NOT-ð/z/d sounds like, substituted /ð/ tokens feed the
realized phone's class, and every other /ð/ realization (t, θ, l, s, f, h, …)
folds into "other" instead of being skipped - that is the recall ceiling the
closed 4-class head could never represent. /ð/ deletions stay out: their
annotated interval is unreliable.

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

ALPHABET = ("ð", "z", "d", "other")
#: The closed-set alternative (the recipe the v22 run used): non-z/d
#: realizations are skipped instead of folded.
CLOSED_ALPHABET = ("ð", "z", "d", "v")
#: Correct tokens of these phones feed the "other" class: what the model must
#: learn to tell apart from ð/z/d.
OTHER_SOURCE_PHONES = ("θ", "s", "t", "f", "l", "v")
#: The phones whose correct tokens feed their own class.
TARGET_PHONES = ("ð", "z", "d")
PAD_S = 0.10  # coarticulation context on each side of the annotated interval


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
    #: "open" folds non-z/d realizations into "other"; "closed" is the
    #: v22-era {ð, z, d, v} alphabet with out-of-alphabet realizations skipped.
    alphabet: str = "open"


def _class_label(token: PhoneToken, alphabet: tuple[str, ...]) -> str | None:
    """The target for a token, or None when it must be skipped.

    Correct tokens of the drilled phones feed their own class. In the open
    alphabet, correct tokens of the widened source set feed "other", and every
    non-z/d /ð/ realization folds into "other"; in the closed alphabet both
    are skipped (the v22-era semantics). Substituted non-ð tokens are not
    contrast evidence and are skipped, as are deletions and unknowns.
    """
    open_set = "other" in alphabet
    if token.label == "correct":
        if token.phone in TARGET_PHONES:
            return token.phone
        if open_set and token.phone in OTHER_SOURCE_PHONES:
            return "other"
        return None
    if token.label == "substituted" and token.phone == "ð":
        if token.substituted_with in ("z", "d"):
            return token.substituted_with
        if open_set and token.substituted_with:
            return "other"
    return None  # deletions, unknowns, non-ð-slot substitutions


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
        # Scale before casting: float samples in [-1, 1] truncate to 0 as
        # int16, which is exactly the silent-export bug the v1 runs trained on.
        handle.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())


def export(options: ExportOptions) -> dict[str, object]:
    """Export the train split; returns the manifest (also written to disk)."""
    alphabet = ALPHABET if options.alphabet == "open" else CLOSED_ALPHABET
    other_sources = OTHER_SOURCE_PHONES if options.alphabet == "open" else ()
    audio_dir = options.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    adapter = L2Arctic(options.l2arctic_root)

    skipped: dict[str, int] = {}
    counts: dict[str, int] = {label: 0 for label in alphabet}
    split_counts: dict[str, int] = {"train": 0, "val": 0}
    rows: list[dict[str, object]] = []

    # Every class needs acoustic examples: correct tokens of the drilled
    # phones feed their own class, correct tokens of the widened source set
    # feed "other", and substituted /ð/ tokens feed the realized phone's
    # class (a /ð/ heard as /z/ IS a /z/).
    for target in (*TARGET_PHONES, *other_sources):
        for utterance in adapter.utterances(target):
            # Only the harness's TRAIN split is exported: the calibration
            # pool feeds the threshold fit and the test pool is the exam.
            if assign_split(utterance, options.split_seed) != "train":
                continue
            is_val = _utterance_is_val(utterance, options.split_seed, options.val_fraction)
            decoded = None
            for token in utterance.tokens(target):
                label = _class_label(token, alphabet)
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
        # /ð/-slot rows are the exam-shaped evidence (correct AND substituted
        # tokens of /ð/): they are never capped. Canonical tokens of the other
        # classes (correct z/d/θ/s/t/f/l/v) are curriculum - cap them hard.
        slot_rows = [row for row in rows if row["target_phone"] == "ð"]
        curriculum_rows = [row for row in rows if row["target_phone"] != "ð"]
        rows = slot_rows + _cap_per_class(
            curriculum_rows, options.max_per_class, options.split_seed
        )
        counts = _recount(rows, alphabet)

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
        "alphabet": list(alphabet),
        "split_seed": options.split_seed,
        "val_fraction": options.val_fraction,
        "pad_s": options.pad_s,
        "max_per_class": options.max_per_class,
        "class_counts": counts,
        "split_counts": split_counts,
        "skipped": skipped,
        "label_policy": (
            "open-set {ð, z, d, other}: ð-slot substitutions fold non-z/d realizations "
            "into other; canonical tokens capped by max_per_class; ð-slot rows never capped"
        ),
        "note": "train split only; the cal and test splits were never exported",
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
    parser.add_argument("--alphabet", choices=["open", "closed"], default="open")
    args = parser.parse_args()
    manifest = export(
        ExportOptions(
            l2arctic_root=Path(args.l2arctic),
            out_dir=Path(args.out),
            split_seed=args.split_seed,
            val_fraction=args.val_fraction,
            pad_s=args.pad_s,
            max_per_class=args.max_per_class,
            alphabet=args.alphabet,
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
