"""Prepare the ear's training data: transcript-only speech -> 16 kHz segments + phones.

The ear (Phase 1) is a frozen XLS-R-300M + fresh CTC head over the charsiu
phone inventory (stressless ARPABET, the vocabulary the repo already commits
for alignment). This script streams license-clean English speech with
transcripts - LibriSpeech (CC-BY-4.0) and VoxPopuli (CC0); Common Voice left
HF in Oct 2025 for Mozilla Data Collective - converts each sentence to a
phone-token sequence with g2p_en (CMU dictionary + rules, stress digits
stripped), and writes 16 kHz mono wavs plus a manifest. The output is
corpus-derived audio, so it never enters git (training/data/ is ignored).

Val split discipline: the hold-out is by SPEAKER, matching the repo's
speaker-disjoint rule - no speaker's voice may leak from train into the
ear's own validation.

Usage (laptop):
    uv run python -m openschwa_training.ear_prep \
        --out data/ear-cv --hours 100 --source librispeech --source voxpopuli
Resumable: existing clips with a completed manifest row are skipped.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
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
    sources: tuple[str, ...] = ("librispeech",),
    *,
    min_duration_s: float = 1.0,
    max_duration_s: float = 30.0,
    max_clips: int | None = None,
) -> dict[str, object]:
    """Stream transcript-only English speech, write wavs + manifest.

    Sources (all license-clean for training shippable weights):
    - librispeech: openslr/librispeech_asr, config clean, train.100 (CC-BY-4.0)
    - voxpopuli:   facebook/voxpopuli, config en, train (CC0)
    Common Voice is listed but unusable since Oct 2025 (moved to Mozilla Data
    Collective; the HF repos are walled off).
    """
    from g2p_en.g2p import G2p  # noqa: PLC0415

    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["id"])
        log.info("resuming: %d clips already prepared", len(done_ids))

    g2p = G2p()
    total_s = 0.0
    kept = 0
    skipped_oov = 0
    seen = 0
    for source in sources:
        if total_s >= hours * 3600:
            break
        total_s, kept, skipped_oov, seen = _prep_source(
            source,
            out_dir,
            g2p,
            done_ids,
            total_s,
            kept,
            skipped_oov,
            seen,
            hours,
            min_duration_s,
            max_duration_s,
            max_clips,
        )

    summary = {
        "hours": round(total_s / 3600, 2),
        "clips": kept,
        "skipped_oov": skipped_oov,
        "seen": seen,
    }
    (out_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("done: %s", summary)
    return summary


def _stream(source: str) -> object:
    """The streaming dataset for a source, audio kept as raw bytes.

    datasets 3.x decodes audio through torchcodec, which has no wheels for
    the pinned torch 2.5.1 - so rows carry encoded bytes and _decode_audio
    does the decoding with PyAV (bundled FFmpeg: FLAC for LibriSpeech, Opus
    for VoxPopuli).
    """
    from datasets import Audio, load_dataset  # noqa: PLC0415 - the ear env only

    if source == "librispeech":
        return load_dataset(
            "openslr/librispeech_asr", "clean", split="train.100", streaming=True
        ).cast_column("audio", Audio(decode=False))
    if source == "voxpopuli":
        return load_dataset("facebook/voxpopuli", "en", split="train", streaming=True).cast_column(
            "audio", Audio(decode=False)
        )
    raise ValueError(
        f"unknown source {source!r} - available: librispeech, voxpopuli "
        "(Common Voice left HF in Oct 2025 for Mozilla Data Collective)"
    )


def _decode_audio(audio: dict[str, object]) -> tuple[np.ndarray, int]:
    """Encoded audio bytes -> (float32 mono samples, sample rate) at 16 kHz."""
    import av  # noqa: PLC0415 - the ear env only

    container = av.open(io.BytesIO(bytes(audio["bytes"])))
    resampler = av.AudioResampler(format="flt", layout="mono", rate=16_000)
    chunks: list[np.ndarray] = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray().reshape(-1))
    if chunks:
        return np.concatenate(chunks).astype(np.float32), 16_000
    return np.zeros(0, dtype=np.float32), 16_000


def _row_fields(source: str, row: dict[str, object]) -> tuple[str, str, str] | None:
    """(clip_id, client_id, sentence) for a source's row; None = skip."""
    if source == "librispeech":
        sentence = str(row.get("text") or "").strip()
        if not sentence:
            return None
        stem = Path(str(row.get("file") or row.get("id") or "")).stem
        return (
            f"ls-{stem}" if stem else f"ls-{row.get('id') or ''}",
            str(row.get("speaker_id") or "unknown"),
            sentence,
        )
    if source == "voxpopuli":
        sentence = str(row.get("normalized_text") or "").strip()
        if not sentence:
            return None
        audio_path = Path(str(row.get("audio", {}).get("path") or ""))
        stem = audio_path.stem or str(row.get("id") or "")
        return (
            f"vp-{stem}",
            str(row.get("speaker_id") or "unknown"),
            sentence,
        )
    return None


def _prep_source(
    source: str,
    out_dir: Path,
    g2p: object,
    done_ids: set[str],
    total_s: float,
    kept: int,
    skipped_oov: int,
    seen: int,
    hours: float,
    min_duration_s: float,
    max_duration_s: float,
    max_clips: int | None,
) -> tuple[float, int, int, int]:
    log.info("source %s: streaming...", source)
    audio_dir = out_dir / "audio"
    manifest_path = out_dir / "manifest.jsonl"
    for row in _stream(source):
        if max_clips is not None and kept >= max_clips:
            break
        if total_s >= hours * 3600:
            break
        seen += 1
        if seen % 1000 == 0:
            log.info("source %s: seen %d, kept %d (%.1f h)", source, seen, kept, total_s / 3600)
        fields = _row_fields(source, row)
        if fields is None:
            continue
        clip_id, client_id, sentence = fields
        if clip_id in done_ids:
            continue
        audio, sample_rate = _decode_audio(row["audio"])
        if audio.size == 0:
            continue
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
    return total_s, kept, skipped_oov, seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="output dir (data/ear-cv)")
    parser.add_argument("--hours", type=float, default=100.0, help="target audio hours")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        choices=["librispeech", "voxpopuli"],
        help="streaming source; repeatable, streamed in order (default: librispeech)",
    )
    parser.add_argument("--max-clips", type=int, default=None, help="hard clip cap (smoke runs)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prep(args.out, args.hours, tuple(args.source or ["librispeech"]), max_clips=args.max_clips)
    # datasets 3.x streaming keeps a background thread that crashes at
    # interpreter teardown (PyGILState_Release after the work is done and the
    # manifest is flushed). Bypass finalization: the data is on disk by now.
    os._exit(0)


if __name__ == "__main__":
    main()
