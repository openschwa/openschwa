"""Export speechocean762 train-split /ð/ ERROR segments (expert-bracketed).

The correct-only exporter skips every token an expert bracketed - roughly 220
real, expert-confirmed Mandarin /ð/ errors the judge has never seen in
training. This exporter keeps exactly those tokens and labels each with the
phone the charsiu aligner actually hears (z/d/v), so the 4-class classifier
finally learns what a Mandarin /ð/ substitution sounds like, not just what a
correct Mandarin /ð/ sounds like.

Tokens whose acoustics the aligner reads as /ð/ anyway are skipped: experts
heard an error the acoustics do not show, and a wrong label would teach the
judge the opposite lesson. The so762 TEST split is never touched.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openschwa_engine.alignment import AlignedPhone
from openschwa_engine.audio import MODEL_SAMPLE_RATE, decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.models.registry import ModelRegistry
from openschwa_eval.datasets import SpeechOcean762
from openschwa_eval.harness import assign_split

from openschwa_training.export import _utterance_is_val, _write_segment_wav
from openschwa_training.export_so762 import PAD_S, _align_with_posteriors

#: Open-set realized classes: any non-ð/z/d realization folds into "other".
REALIZED_CLASSES = ("ð", "z", "d", "other")


@dataclass(frozen=True)
class ErrorExportOptions:
    so762_root: Path
    out_dir: Path
    model_dir: Path | None = None  # None -> Settings().model_dir
    split_seed: int = 42
    val_fraction: float = 0.15
    pad_s: float = PAD_S
    #: The realized class must hold at least this share of the closed-set
    #: frame mass over the token, else the label is ambiguous and the row is
    #: skipped rather than teaching noise.
    min_class_mass: float = 0.6


def _realized_phone(
    log_probs: np.ndarray,
    frame_indices: np.ndarray | list[int] | tuple[int, ...],
    closed_index_of: dict[str, int | list[int]],
    min_class_mass: float = 0.6,
) -> str | None:
    """The phone the aligner hears over the token's frames, or None.

    Mass-weighted over the open-set classes {ð, z, d, other}: the "other"
    entry may map to a LIST of vocabulary indices, and its mass is the sum of
    the per-frame maxima over those indices (the single dominant non-ð/z/d
    realization). The realized class must hold >= min_class_mass of the
    closed-set mass.
    """
    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.size == 0 or log_probs is None:
        return None
    names = list(closed_index_of)
    per_class = []
    for name in names:
        idx = closed_index_of[name]
        cols = [idx] if isinstance(idx, int) else list(idx)
        if not cols:
            per_class.append(np.full(log_probs.shape[0], -1e9))
            continue
        if len(cols) > 1:
            per_class.append(log_probs[frames][:, cols].max(axis=1))
        else:
            per_class.append(log_probs[frames][:, cols[0]])
    logp = np.stack(per_class, axis=1).astype(np.float64)
    shifted = logp - logp.max(axis=1, keepdims=True)
    mass = np.exp(shifted).mean(axis=0)
    mass /= mass.sum() + 1e-12
    winner = int(np.argmax(mass))
    if mass[winner] < min_class_mass:
        return None
    return names[winner]


def _closed_map(phone_map, log_probs: np.ndarray) -> dict[str, int | list[int]]:
    """{ð, z, d, other} -> vocabulary index / indices for the open-set vote.

    "other" covers every vocabulary index that is not blank, ð, z or d: its
    per-frame value is the max over those indices (the single dominant
    non-ð/z/d realization).
    """
    used = {phone_map.index_of[c] for c in ("ð", "z", "d")}
    used.add(phone_map.blank_index)
    other = [i for i in range(log_probs.shape[1]) if i not in used]
    return {
        "ð": phone_map.index_of["ð"],
        "z": phone_map.index_of["z"],
        "d": phone_map.index_of["d"],
        "other": other,
    }


def export_so762_errors(options: ErrorExportOptions) -> dict[str, object]:
    audio_dir = options.out_dir / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    registry = ModelRegistry(options.model_dir or settings.model_dir)
    adapter = SpeechOcean762(options.so762_root)
    charsiu_map = registry.phone_map(registry.spec(settings.alignment_model))

    skipped: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    for utterance in adapter.utterances("ð"):
        # Only the harness's TRAIN split is exported: the cal carve and the
        # native test partition are the exam's fitting and held-out pools.
        if assign_split(utterance, options.split_seed) != "train":
            continue
        is_val = _utterance_is_val(utterance, options.split_seed, options.val_fraction)
        decoded = decode_wav(utterance.audio_path.read_bytes())
        prepared = prepare(decoded.samples, decoded.sample_rate)
        aligned, log_probs = _align_with_posteriors(utterance, prepared, registry, settings)
        if not aligned or log_probs is None:
            skipped["alignment failed"] = skipped.get("alignment failed", 0) + 1
            continue
        for token in utterance.tokens("ð"):
            if token.label == "correct" and token.expert_error_votes in (0, None):
                continue  # not an expert-bracketed error
            phone: AlignedPhone | None = aligned.get(token.index)
            if phone is None:
                skipped["token not aligned"] = skipped.get("token not aligned", 0) + 1
                continue
            realized = _realized_phone(
                log_probs,
                phone.frame_indices,
                _closed_map(charsiu_map, log_probs),
                options.min_class_mass,
            )
            if realized is None or realized == "ð":
                skipped["aligner heard ð"] = skipped.get("aligner heard ð", 0) + 1
                continue
            start = max(0, int((phone.start_s - options.pad_s) * MODEL_SAMPLE_RATE))
            end = min(
                prepared.samples_16k.size, int((phone.end_s + options.pad_s) * MODEL_SAMPLE_RATE)
            )
            segment = prepared.samples_16k[start:end]
            if segment.size < int(0.03 * MODEL_SAMPLE_RATE):
                skipped["too short"] = skipped.get("too short", 0) + 1
                continue
            stem = f"so762err-{utterance.utterance_id}_{token.index}"
            _write_segment_wav(audio_dir / f"{stem}.wav", segment)
            rows.append(
                {
                    "filename": f"audio/{stem}.wav",
                    "label": realized,
                    "l1": "mandarin",
                    "utterance_id": f"so762err-{utterance.utterance_id}",
                    "token_index": token.index,
                    "target_phone": "ð",
                    "expert_error_votes": token.expert_error_votes,
                    "start_s": round(phone.start_s, 4),
                    "end_s": round(phone.end_s, 4),
                    "duration_s": round(len(segment) / MODEL_SAMPLE_RATE, 4),
                    "split": "val" if is_val else "train",
                }
            )

    rows.sort(key=lambda row: (str(row["split"]), str(row["filename"])))
    fieldnames = [
        "filename",
        "label",
        "l1",
        "utterance_id",
        "token_index",
        "target_phone",
        "expert_error_votes",
        "start_s",
        "end_s",
        "duration_s",
        "split",
    ]
    with (options.out_dir / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest: dict[str, object] = {
        "corpus": "speechocean762 (train split only)",
        "intervals": "charshu aligner, raw-audio slice (exam-style extraction)",
        "label_policy": (
            "expert-bracketed /ð/ errors, labeled with the charsiu aligner's realized phone; "
            "ambiguous tokens (aligner hears ð, or no class >= min_class_mass) are skipped"
        ),
        "split_seed": options.split_seed,
        "val_fraction": options.val_fraction,
        "pad_s": options.pad_s,
        "min_class_mass": options.min_class_mass,
        "rows": len(rows),
        "split_counts": {
            split: sum(1 for row in rows if row["split"] == split) for split in ("train", "val")
        },
        "skipped": skipped,
        "note": "the so762 TEST split was never exported - it stays the exam's held-out",
    }
    (options.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so762", required=True, help="local speechocean762 corpus root")
    parser.add_argument("--out", required=True, help="export directory")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--pad-s", type=float, default=PAD_S)
    parser.add_argument("--min-class-mass", type=float, default=0.6)
    args = parser.parse_args()
    manifest = export_so762_errors(
        ErrorExportOptions(
            so762_root=Path(args.so762),
            out_dir=Path(args.out),
            split_seed=args.split_seed,
            val_fraction=args.val_fraction,
            pad_s=args.pad_s,
            min_class_mass=args.min_class_mass,
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()