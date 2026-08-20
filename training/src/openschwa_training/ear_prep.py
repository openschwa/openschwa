"""Prepare the ear's training data: Common Voice EN -> 16 kHz segments + phone labels.

The ear (Phase 1) is a frozen XLS-R-300M + fresh CTC head over the charsiu
phone inventory (stressless ARPABET, the vocabulary the repo already commits
for alignment). This script streams the Common Voice EN train split (CC0),
keeps validated clips, converts each sentence to a phone-token sequence with
g2p_en (CMU dictionary + rules, stress digits stripped), and writes 16 kHz
mono wavs plus a manifest. The output is corpus-derived audio, so it never
enters git (training/data/ is ignored).

Val split discipline: the hold-out is by CLIENT (speaker), matching the
repo's speaker-disjoint rule - no speaker's voice may leak from train into
the ear's own validation.

Usage (laptop):
    uv run python -m openschwa_training.ear_prep \
        --out data/ear-cv --hours 60
Resumable: existing clips with a completed manifest row are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("openschwa-training")

#: charsiu's stressless ARPABET tokens - the ear's head vocabulary. Same set
#: as engine/models/vocab/charsiu-en-w2v2-ctc.json minus [SIL]/[UNK]/[PAD].
PHONE_TOKENS = tuple(
    [
        "NG",
        "F",
        "M",
        "AE",
        "R",
        "UW",
        "N",
        "IY",
        "AW",
        "V",
        "UH",
        "OW",
        "AA",
        "ER",
        "HH",
        "Z",
        "K",
        "CH",
        "W",
        "EY",
        "ZH",
        "T",
        "EH",
        "Y",
        "AH",
        "B",
        "P",
        "TH",
        "DH",
        "AO",
        "G",
        "L",
        "JH",
        "OY",
        "SH",
        "D",
        "AY",
        "S",
        "IH",
    ]
)
TOKEN_INDEX = {token: i + 1 for i, token in enumerate(PHONE_TOKENS)}  # 0 = [PAD]
TOKEN_SET = frozenset(PHONE_TOKENS)


def vocab() -> dict[str, int]:
    """The ear's committed vocabulary: [PAD] blank at 0, phones at 1..39."""
    return {"[PAD]": 0, **TOKEN_INDEX}


def sentence_to_tokens(sentence: str, g2p: object) -> list[str]:
    """Words -> stressless ARPABET tokens via g2p_en; raises on any OOV phone.

    g2p_en's rule-based fallback can produce non-CMU phones for rare words;
    a token outside the ear's vocabulary would silently poison the CTC target,
    so it fails loud instead (the caller skips the sentence).
    """
    tokens: list[str] = []
    for word in sentence.split():
        for phone in g2p(word):  # type: ignore[operator]
            token = "".join(ch for ch in phone if not ch.isdigit())
            if token not in TOKEN_SET:
                raise ValueError(f"unmappable phone {phone!r} in word {word!r}")
            tokens.append(token)
    return tokens


def build_manifest_row(
    clip_id: str, client_id: str, sentence: str, tokens: list[str], wav_path: Path
) -> dict[str, object]:
    return {
        "id": clip_id,
        "client_id": client_id,
        "text": sentence,
        "tokens": " ".join(tokens),
        "audio_path": str(wav_path),
    }


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())


def prep(
    out_dir: Path,
    hours: float,
    *,
    min_duration_s: float = 1.0,
    max_duration_s: float = 10.0,
    max_clips: int | None = None,
) -> dict[str, object]:
    """Stream Common Voice EN, keep validated clips, write wavs + manifest."""
    from datasets import Audio, load_dataset  # noqa: PLC0415 - the ear env only
    from g2p_en.g2p import G2p  # noqa: PLC0415

    audio_dir = out_dir / "audio"
    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["id"])
        log.info("resuming: %d clips already prepared", len(done_ids))

    dataset = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "en",
        split="train",
        streaming=True,
    ).cast_column("audio", Audio(sampling_rate=16_000))

    g2p = G2p()
    total_s = 0.0
    kept = 0
    skipped_oov = 0
    skipped_unvalidated = 0
    seen = 0
    for row in dataset:
        seen += 1
        if max_clips is not None and kept >= max_clips:
            break
        if total_s >= hours * 3600:
            break
        if seen % 1000 == 0:
            log.info("seen %d, kept %d (%.1f h)", seen, kept, total_s / 3600)
        clip_id = str(row.get("path") or row.get("audio", {}).get("path") or f"cv-{seen}")
        if clip_id in done_ids:
            continue
        client_id = str(row.get("client_id") or "unknown")
        sentence = str(row.get("sentence") or "").strip()
        if not sentence:
            continue
        if int(row.get("up_votes") or 0) <= int(row.get("down_votes") or 0):
            skipped_unvalidated += 1
            continue
        audio = row["audio"]["array"]
        sample_rate = int(row["audio"]["sampling_rate"])
        duration = len(audio) / sample_rate
        if not min_duration_s <= duration <= max_duration_s:
            continue
        try:
            tokens = sentence_to_tokens(sentence, g2p)
        except (ValueError, IndexError) as exc:
            skipped_oov += 1
            if skipped_oov < 10:
                log.debug("skip '%s': %s", sentence[:60], exc)
            continue
        wav_path = audio_dir / f"{clip_id}.wav"
        _write_wav(wav_path, audio, sample_rate)
        row_out = build_manifest_row(clip_id, client_id, sentence, tokens, wav_path)
        new_file = not manifest_path.is_file() or manifest_path.stat().st_size == 0
        writer = csv.DictWriter(
            manifest_path.open("a", encoding="utf-8"), fieldnames=list(row_out.keys())
        )
        if new_file:
            writer.writeheader()
        writer.writerow(row_out)
        done_ids.add(clip_id)
        kept += 1
        total_s += duration

    summary = {
        "hours": round(total_s / 3600, 2),
        "clips": kept,
        "skipped_oov": skipped_oov,
        "skipped_unvalidated": skipped_unvalidated,
        "seen": seen,
    }
    (out_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("done: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="output dir (data/ear-cv)")
    parser.add_argument("--hours", type=float, default=60.0, help="target audio hours")
    parser.add_argument("--max-clips", type=int, default=None, help="hard clip cap (smoke runs)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prep(args.out, args.hours, max_clips=args.max_clips)


if __name__ == "__main__":
    main()
