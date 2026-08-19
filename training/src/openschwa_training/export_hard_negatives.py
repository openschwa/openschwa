"""Hard-negative mining: re-export the correct tokens the judge misreads.

The v4-v6 exams showed the precision wall: a handful of *correct* tokens
(mostly Mandarin /ð/) score at the very top of the substitution range, and
one pooled line can never separate them from true errors. These tokens are
the model's blind spot, so they go back into training - labeled correct,
extracted exactly the way the exam sees them - until the model stops
confusing them with errors.

Selection: the top-K correct tokens of the TRAIN pool by the exam's raw
contrast score. The test pool is never touched: it stays the exam's
held-out.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from openschwa_engine.alignment import AlignedPhone, align_exercise
from openschwa_engine.audio import MODEL_SAMPLE_RATE, decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.content.loader import Exercise, PhoneSpec
from openschwa_engine.models.registry import ModelRegistry
from openschwa_eval.datasets import L2Arctic, SpeechOcean762

from openschwa_training.export import _write_segment_wav
from openschwa_training.export_so762 import PAD_S


@dataclass(frozen=True)
class HardNegOptions:
    checkpoint: Path  # exam checkpoint jsonl (the scorer's records)
    l2arctic_root: Path
    so762_root: Path
    out_dir: Path
    model_dir: Path | None = None
    top_k: int = 200
    l1: str | None = None  # mine only this language group (e.g. "mandarin")
    pad_s: float = PAD_S


def select_hard_negatives(
    records: list[dict], top_k: int, l1: str | None = None
) -> list[dict]:
    """The top-K train-pool CORRECT tokens by contrast score.

    With l1 set, only that language group is mined: the Mandarin wall is the
    one that breaks the pooled bar, so a Mandarin-only round targets exactly
    the tokens whose correct label the judge keeps refusing to believe.
    """
    candidates = [
        r
        for r in records
        if r.get("split") == "train"
        and r.get("label") == "correct"
        and r.get("score") is not None
        and (l1 is None or r.get("l1") == l1)
    ]
    candidates.sort(key=lambda r: r["score"], reverse=True)
    return candidates[:top_k]


def _align_phones(utterance, prepared, registry, settings) -> dict[int, AlignedPhone]:
    phones = tuple(
        PhoneSpec(index=t.index, ph=t.phone, focus=False, confusions=()) for t in utterance.phones
    )
    exercise = Exercise(
        id=f"hardneg-{utterance.utterance_id}",
        pack_id="hardneg",
        type="word",
        title="",
        lang="en",
        text=utterance.transcript,
        ipa="",
        phones=phones,
        source_path=Path("hardneg"),
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
        return {}
    return {phone.index: phone for phone in outcome.phones}


def export_hard_negatives(options: HardNegOptions) -> dict[str, object]:
    records = [
        json.loads(line) for line in options.checkpoint.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    picks = select_hard_negatives(records, options.top_k, l1=options.l1)
    keys = {(r["utterance_id"], r["token_index"]) for r in picks}
    settings = Settings()
    registry = ModelRegistry(options.model_dir or settings.model_dir)
    adapters = [
        L2Arctic(options.l2arctic_root),
        SpeechOcean762(options.so762_root),
    ]

    audio_dir = options.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    skipped: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for adapter in adapters:
        for utterance in adapter.utterances("ð"):
            wanted = [
                t for t in utterance.tokens("ð")
                if (utterance.utterance_id, t.index) in keys
                and (utterance.utterance_id, t.index) not in seen
            ]
            if not wanted:
                continue
            decoded = decode_wav(utterance.audio_path.read_bytes())
            prepared = prepare(decoded.samples, decoded.sample_rate)
            aligned = _align_phones(utterance, prepared, registry, settings)
            for token in wanted:
                phone = aligned.get(token.index)
                if phone is None:
                    skipped["token not aligned"] = skipped.get("token not aligned", 0) + 1
                    continue
                start = max(0, int((phone.start_s - options.pad_s) * MODEL_SAMPLE_RATE))
                end = min(
                    prepared.samples_16k.size,
                    int((phone.end_s + options.pad_s) * MODEL_SAMPLE_RATE),
                )
                segment = prepared.samples_16k[start:end]
                if segment.size < int(0.03 * MODEL_SAMPLE_RATE):
                    skipped["too short"] = skipped.get("too short", 0) + 1
                    continue
                stem = f"hardneg-{utterance.utterance_id}_{token.index}"
                _write_segment_wav(audio_dir / f"{stem}.wav", segment)
                rows.append(
                    {
                        "filename": f"audio/{stem}.wav",
                        "label": "ð",
                        "l1": utterance.l1,
                        "utterance_id": utterance.utterance_id,
                        "token_index": token.index,
                        "target_phone": "ð",
                        "start_s": round(phone.start_s, 4),
                        "end_s": round(phone.end_s, 4),
                        "duration_s": round(len(segment) / MODEL_SAMPLE_RATE, 4),
                        "split": "train",
                    }
                )
                seen.add((utterance.utterance_id, token.index))

    rows.sort(key=lambda row: (str(row["split"]), str(row["filename"])))
    fieldnames = [
        "filename", "label", "l1", "utterance_id", "token_index", "target_phone",
        "start_s", "end_s", "duration_s", "split",
    ]
    with (options.out_dir / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest: dict[str, object] = {
        "label_policy": "top-K correct tokens by the exam contrast score (hard negatives)",
        "l1": options.l1,
        "top_k": options.top_k,
        "pad_s": options.pad_s,
        "rows": len(rows),
        "skipped": skipped,
        "note": "only the exam TRAIN pool is mined; the test pool is untouched",
    }
    (options.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="exam checkpoint jsonl")
    parser.add_argument("--l2arctic", required=True)
    parser.add_argument("--so762", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument(
        "--l1", default=None, help="mine only this L1 group (e.g. mandarin)"
    )
    args = parser.parse_args()
    manifest = export_hard_negatives(
        HardNegOptions(
            checkpoint=Path(args.checkpoint),
            l2arctic_root=Path(args.l2arctic),
            so762_root=Path(args.so762),
            out_dir=Path(args.out),
            top_k=args.top_k,
            l1=args.l1,
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()