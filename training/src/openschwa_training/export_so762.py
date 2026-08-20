"""Export speechocean762 train-split /ð/ segments with aligner-derived intervals.

so762 carries expert phone labels but no timestamps. The charsiu aligner
supplies the intervals - the same boundaries the exam's extraction uses - and
only DEFINITELY correct tokens are kept (all five experts agree, zero
brackets), so the judge learns what Mandarin /ð/ sounds like without label
noise. The so762 TEST split is never touched: it stays the exam's held-out
pool.

This is the fix for the v2 exam finding: the judge scored 0.69 AUC on
L2-ARCTIC held-out and 0.96 on so762 held-out separately, but 0.55 pooled -
because it had never seen Mandarin /ð/ and ranked correct Mandarin tokens as
substitutions. Teaching it what correct Mandarin /ð/ sounds like realigns the
pooled distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openschwa_engine.alignment import AlignedPhone, align_exercise
from openschwa_engine.audio import MODEL_SAMPLE_RATE, decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.content.loader import Exercise, PhoneSpec
from openschwa_engine.models.registry import ModelRegistry
from openschwa_eval.datasets import SpeechOcean762
from openschwa_eval.harness import assign_split

from openschwa_training.export import _utterance_is_val, _write_segment_wav

PAD_S = 0.10


class ExportError(RuntimeError):
    """The export cannot proceed without corrupting the dataset."""


@dataclass(frozen=True)
class So762Options:
    so762_root: Path
    out_dir: Path
    model_dir: Path | None = None  # None -> Settings().model_dir
    split_seed: int = 42
    val_fraction: float = 0.15
    max_tokens: int | None = 1500  # cap the ð class: keep the classes balanced
    pad_s: float = PAD_S


def _align_with_posteriors(
    utterance, prepared, registry, settings
) -> tuple[dict[int, AlignedPhone], np.ndarray | None]:
    """Align, returning the phones by index plus the frame log-posteriors.

    The posteriors are the aligner's raw per-frame log-probabilities over its
    vocabulary - what the error exporter uses to read off the *realized*
    phone for an expert-bracketed token.
    """
    phones = tuple(
        PhoneSpec(index=t.index, ph=t.phone, focus=False, confusions=()) for t in utterance.phones
    )
    exercise = Exercise(
        id=f"so762-export-{utterance.utterance_id}",
        pack_id="so762-export",
        type="word",
        title="",
        lang="en",
        text=utterance.transcript,
        ipa="",
        phones=phones,
        source_path=Path("so762-export"),
    )
    spec = registry.spec(settings.alignment_model)
    model_dir = registry.require_ready(spec)
    phone_map = registry.phone_map(spec)
    from openschwa_engine.alignment.acoustic import load

    outcome = align_exercise(
        prepared,
        exercise.phone_labels,
        phone_map,
        load(model_dir),
        min_confidence=0.0,
        low_confidence=0.0,
    )
    if not outcome.ok:
        return {}, None
    log_probs = outcome.posteriors.log_probs if outcome.posteriors is not None else None
    return {phone.index: phone for phone in outcome.phones}, log_probs


def _align_for_export(utterance, prepared, registry, settings) -> dict[int, AlignedPhone]:
    """Align the utterance and index the aligned phones by sequence position."""
    aligned, _posteriors = _align_with_posteriors(utterance, prepared, registry, settings)
    return aligned


def export_so762(options: So762Options) -> dict[str, object]:
    audio_dir = options.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    registry = ModelRegistry(options.model_dir or settings.model_dir)
    adapter = SpeechOcean762(options.so762_root)

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
        aligned = _align_for_export(utterance, prepared, registry, settings)
        if not aligned:
            skipped["alignment failed"] = skipped.get("alignment failed", 0) + 1
            continue
        for token in utterance.tokens("ð"):
            if token.label != "correct":
                skipped["expert-bracketed"] = skipped.get("expert-bracketed", 0) + 1
                continue
            if token.expert_error_votes not in (0, None):
                skipped["expert-bracketed"] = skipped.get("expert-bracketed", 0) + 1
                continue
            phone = aligned.get(token.index)
            if phone is None:
                skipped["token not aligned"] = skipped.get("token not aligned", 0) + 1
                continue
            start = max(0, int((phone.start_s - options.pad_s) * MODEL_SAMPLE_RATE))
            end = min(
                prepared.samples_16k.size, int((phone.end_s + options.pad_s) * MODEL_SAMPLE_RATE)
            )
            segment = prepared.samples_16k[start:end]
            if segment.size < int(0.03 * MODEL_SAMPLE_RATE):
                skipped["too short"] = skipped.get("too short", 0) + 1
                continue
            stem = f"so762-{utterance.utterance_id}_{token.index}"
            _write_segment_wav(audio_dir / f"{stem}.wav", segment)
            rows.append(
                {
                    "filename": f"audio/{stem}.wav",
                    "label": "ð",
                    "l1": "mandarin",
                    "utterance_id": f"so762-{utterance.utterance_id}",
                    "token_index": token.index,
                    "target_phone": "ð",
                    "start_s": round(phone.start_s, 4),
                    "end_s": round(phone.end_s, 4),
                    "duration_s": round(len(segment) / MODEL_SAMPLE_RATE, 4),
                    "split": "val" if is_val else "train",
                }
            )

    if options.max_tokens is not None and len(rows) > options.max_tokens:
        rng = random.Random(f"{options.split_seed}:so762-cap")
        rows = rng.sample(rows, options.max_tokens)

    rows.sort(key=lambda row: (str(row["split"]), str(row["filename"])))
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
    with (options.out_dir / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest: dict[str, object] = {
        "corpus": "speechocean762 (train split only)",
        "intervals": "charshu aligner, raw-audio slice (exam-style extraction)",
        "label_policy": "5/5 experts agree the phone is correct",
        "split_seed": options.split_seed,
        "val_fraction": options.val_fraction,
        "pad_s": options.pad_s,
        "max_tokens": options.max_tokens,
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
    parser.add_argument("--max-tokens", type=int, default=1500)
    args = parser.parse_args()
    manifest = export_so762(
        So762Options(
            so762_root=Path(args.so762),
            out_dir=Path(args.out),
            split_seed=args.split_seed,
            val_fraction=args.val_fraction,
            max_tokens=args.max_tokens,
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
