"""Train the ear (Phase 1): frozen XLS-R-300M + fresh CTC head, on cached features.

The mirror needs an ear with the FULL phone vocabulary, trained on
transcript-only mass (Common Voice, CC0) - connected speech the judge line
could never use. The base is facebook/wav2vec2-xls-r-300m (Apache-2.0),
frozen forever; only a 40-class CTC head (charsiu's stressless ARPABET
inventory + [PAD] blank) trains.

Three stages, resumable:
- extract: run each prepared clip through the frozen encoder once (bf16),
  cache the final hidden states in shards (features + per-clip index);
- train:   CTC head on the cached features - no encoder forward, so epochs
  are minutes and hyperparameter sweeps are cheap; val split is by CLIENT
  (speaker-disjoint, the repo's discipline);
- export:  assemble the engine-loadable model dir (XLS-R base + trained head,
  config/vocab/preprocessor) in the exact layout the registry expects.

Usage (laptop):
    uv run python -m openschwa_training.ear_train \
        --data data/ear-cv --base-model ../.models/wav2vec2-xls-r-300m \
        --out runs/ear-v1 --stage extract
    ... then --stage train, then --stage export
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import wave
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from openschwa_training.ear_prep import TOKEN_INDEX, vocab

log = logging.getLogger("openschwa-training")

VOCAB = vocab()  # {[PAD]: 0, phones: 1..39}
NUM_CLASSES = len(VOCAB)  # 40
BLANK = 0
SHARD_CLIPS = 2000
FEATURE_DIM = 1024
BATCH_MAX_CLIPS = 8
# Cap a batch at ~15 s of audio: attention memory grows with B x T^2, and
# 8 x (750 encoder frames)^2 x 16 heads x 24 layers x bf16 ~= 3.5 GB -
# comfortably inside the 4060's 8 GB. Bigger batches thrash and slow down.
BATCH_MAX_SAMPLES = 240_000


@dataclass(frozen=True)
class ShardIndex:
    """One feature shard: concatenated features + per-clip spans."""

    path: Path
    clips: tuple[dict[str, object], ...]  # id, start, length, tokens, split


@dataclass(frozen=True)
class ClipRef:
    """A clip inside its shard: enough to slice the feature array."""

    shard: ShardIndex
    position: int  # index into shard.clips

    @property
    def clip(self) -> dict[str, object]:
        return self.shard.clips[self.position]


def _client_split(client_id: str) -> str:
    """Speaker-disjoint val: a stable ~5% hash bucket of clients."""
    bucket = int(hashlib.md5(client_id.encode("utf-8")).hexdigest()[0:8], 16) % 100
    return "val" if bucket < 5 else "train"


def load_manifest(data_dir: Path) -> list[dict[str, object]]:
    from openschwa_training.ear_prep import _manifest_rows  # noqa: PLC0415

    manifest = data_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest}: run ear_prep first")
    return _manifest_rows(manifest)


def extract(
    data_dir: Path, base_model_dir: Path, out_dir: Path, *, max_clips: int | None = None
) -> dict[str, object]:
    """Frozen-encoder pass: cache last hidden states, sharded, resumable.

    Clips are batched (pad to the batch's longest, cap by total samples) -
    single-sample forwards cost ~1 s each on the 4060 and would make 28k
    clips an eight-hour stage; batched, the same stage is well under an hour.
    """
    from transformers import Wav2Vec2Model  # noqa: PLC0415 - the ear env only

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = Wav2Vec2Model.from_pretrained(str(base_model_dir)).to(device).eval()
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(data_dir)
    if max_clips is not None:
        rows = rows[:max_clips]
    done: set[str] = set()
    for index_path in feat_dir.glob("shard-*.index.jsonl"):
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    log.info("%d clips total, %d already extracted", len(rows), len(done))

    shard: list[dict[str, object]] = []
    flat: list[np.ndarray] = []
    shard_number = 0
    frames_total = 0
    shard_frames = 0  # offsets are PER-SHARD: each .npy starts at row 0
    extracted = 0
    pending: list[dict[str, object]] = []
    pending_samples = 0
    for row in rows:
        clip_id = str(row["id"])
        if clip_id in done:
            continue
        wav_path = Path(str(row["audio_path"]))
        if not wav_path.is_file():
            log.warning("missing audio %s - skipped", wav_path)
            continue
        with wave.open(str(wav_path), "rb") as handle:
            samples = (
                np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(
                    np.float32
                )
                / 32768.0
            )
        pending.append(
            {
                "id": clip_id,
                "client_id": str(row["client_id"]),
                "tokens": str(row["tokens"]),
                "samples": samples,
            }
        )
        pending_samples += samples.size
        if len(pending) >= BATCH_MAX_CLIPS or pending_samples >= BATCH_MAX_SAMPLES:
            frames_total, shard_frames = _flush_batch(
                encoder, device, pending, shard, flat, frames_total, shard_frames, feat_dir
            )
            extracted += len(pending)
            pending = []
            pending_samples = 0
        if extracted % 200 == 0 and extracted:
            log.info(
                "extracted %d clips (%.1f h of features)", extracted, frames_total * 0.02 / 3600
            )
    if pending:
        frames_total, shard_frames = _flush_batch(
            encoder, device, pending, shard, flat, frames_total, shard_frames, feat_dir
        )
        extracted += len(pending)
    if shard:
        _write_shard(feat_dir, shard_number, shard, flat)
        shard_number += 1
    return {"shards": shard_number, "extracted": extracted}


def _flush_batch(
    encoder: object,
    device: str,
    pending: list[dict[str, object]],
    shard: list[dict[str, object]],
    flat: list[np.ndarray],
    frames_total: int,
    shard_frames: int,
    feat_dir: Path,
) -> tuple[int, int]:
    """One batched encoder forward: pad, run, unbatch into shard entries.

    The attention mask is NOT optional: without it every clip's frames attend
    across the whole padded batch, so the cached features carry the other
    clips' audio. The head then trains on batch-contaminated features while
    inference feeds clean single clips - the exported ear decodes garbage
    (raw top-1 4% on the first exam was exactly this).

    Offsets are PER-SHARD (`shard_frames`): each shard's .npy starts at row 0,
    and a clip's `start` is its offset inside ITS shard. A global accumulator
    here poisoned every shard after the first - clips sliced past the array
    end, CTCLoss got zero-length inputs, and training died with
    'invalid configuration argument' on random batches.
    """
    max_len = max(int(p["samples"].size) for p in pending)
    padded = torch.zeros(len(pending), max_len, dtype=torch.float32)
    mask = torch.zeros(len(pending), max_len, dtype=torch.long)
    for position, entry in enumerate(pending):
        samples = entry["samples"]
        padded[position, : samples.size] = torch.from_numpy(samples)
        mask[position, : samples.size] = 1
    padded = padded.to(device)
    mask = mask.to(device)
    autocast = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with torch.inference_mode(), autocast:
        hidden = encoder(padded, attention_mask=mask).last_hidden_state  # [B, T, 1024]
    valid = [int(encoder._get_feat_extract_output_lengths(p["samples"].size)) for p in pending]
    for position, entry in enumerate(pending):
        feat = hidden[position, : valid[position]].float().cpu().numpy()
        # bf16 encoder output can exceed fp16's range (65504); casting those
        # values to float16 silently yields inf, the head trains on garbage,
        # and the loss plateaus at the random-prediction level. Clamp first.
        feat = np.clip(feat, -65_504.0, 65_504.0).astype(np.float16)
        flat.append(feat)
        shard.append(
            {
                "id": str(entry["id"]),
                "client_id": str(entry["client_id"]),
                "start": shard_frames,
                "length": int(feat.shape[0]),
                "tokens": str(entry["tokens"]),
                "split": _client_split(str(entry["client_id"])),
            }
        )
        shard_frames += feat.shape[0]
        frames_total += feat.shape[0]
    while len(shard) >= SHARD_CLIPS:
        head = shard[:SHARD_CLIPS]
        write_flat = flat[:SHARD_CLIPS]
        del shard[:SHARD_CLIPS]
        del flat[:SHARD_CLIPS]
        number = len(list(feat_dir.glob("shard-*.npy")))
        _write_shard(feat_dir, number, head, write_flat)
        shard_frames = 0
    return frames_total, shard_frames


def _write_shard(
    feat_dir: Path, number: int, clips: list[dict[str, object]], flat: list[np.ndarray]
) -> None:
    stacked = np.concatenate(flat, axis=0).astype(np.float16)
    array_path = feat_dir / f"shard-{number:04d}.npy"
    index_path = feat_dir / f"shard-{number:04d}.index.jsonl"
    np.save(array_path, stacked)
    with index_path.open("w", encoding="utf-8") as handle:
        for clip in clips:
            handle.write(json.dumps(clip) + "\n")
    log.info("shard %d: %d clips, %d frames", number, len(clips), stacked.shape[0])


def load_shards(out_dir: Path) -> list[ShardIndex]:
    shards: list[ShardIndex] = []
    for index_path in sorted((out_dir / "features").glob("shard-*.index.jsonl")):
        clips = tuple(
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        # shard-0000.index.jsonl -> shard-0000.npy. with_suffix would replace
        # only the LAST suffix and yield a nonexistent shard-0000.index.npy -
        # which silently dropped every shard and made train() see "no shards".
        array_path = index_path.with_name(index_path.name.replace(".index.jsonl", ".npy"))
        if clips and array_path.is_file():
            shards.append(ShardIndex(path=array_path, clips=clips))
    return shards


def _clip_batch(
    picks: list[ClipRef], arrays: dict[Path, np.ndarray]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Padded feature batch + lengths + token targets (padded with BLANK)."""
    feats: list[np.ndarray] = []
    for pick in picks:
        clip = pick.clip
        array = arrays[pick.shard.path]
        feats.append(array[clip["start"] : clip["start"] + clip["length"]])
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    max_len = int(lengths.max())
    padded = torch.zeros(len(feats), max_len, FEATURE_DIM, dtype=torch.float32)
    for position, feat in enumerate(feats):
        padded[position, : feat.shape[0]] = torch.from_numpy(feat.astype(np.float32))
    token_lists = [[TOKEN_INDEX[t] for t in pick.clip["tokens"].split()] for pick in picks]
    max_tokens = max(len(tokens) for tokens in token_lists)
    targets = torch.full((len(feats), max_tokens), BLANK, dtype=torch.long)
    target_lengths = torch.tensor([len(tokens) for tokens in token_lists], dtype=torch.long)
    for position, tokens in enumerate(token_lists):
        targets[position, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return padded, lengths, targets, target_lengths


def train(
    out_dir: Path, *, epochs: int = 12, batch_size: int = 64, lr: float = 1e-3, seed: int = 42
) -> dict[str, object]:
    """CTC head on cached features; speaker-disjoint val, resumable."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = torch.nn.Linear(FEATURE_DIM, NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    checkpoint = out_dir / "head-last.pt"
    start_epoch = 0
    history: list[dict[str, object]] = []
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device)
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        history = state["history"]
        log.info("resumed from epoch %d", start_epoch)

    shards = load_shards(out_dir)
    if not shards:
        raise RuntimeError(f"{out_dir}/features has no shards - run --stage extract first")

    def _ctc_valid(ref: ClipRef) -> bool:
        """CTC requires more input frames than target tokens per item; a
        short fast clip with a long sentence violates it and CTCLoss dies
        with 'invalid configuration argument' mid-epoch. Zero-length
        features (a corrupt shard offset) die the same way - never let them
        reach the kernel."""
        length = int(ref.clip["length"])
        return length > 0 and len(ref.clip["tokens"].split()) < length

    train_refs: list[ClipRef] = []
    val_refs: list[ClipRef] = []
    for shard in shards:
        for position, clip in enumerate(shard.clips):
            ref = ClipRef(shard=shard, position=position)
            (val_refs if clip["split"] == "val" else train_refs).append(ref)
    train_refs = [ref for ref in train_refs if _ctc_valid(ref)]
    val_refs = [ref for ref in val_refs if _ctc_valid(ref)]
    log.info("train %d / val %d clips", len(train_refs), len(val_refs))

    total_steps = max(1, epochs * (len(train_refs) + batch_size - 1) // batch_size)
    warmup = max(1, total_steps // 10)

    def schedule(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    # mmap: features stay on disk and page in per batch (60 h of float16
    # features is ~22 GB - far beyond the laptop's RAM).
    arrays = {shard.path: np.load(shard.path, mmap_mode="r") for shard in shards}

    step = 0
    for epoch in range(start_epoch, epochs):
        head.train()
        total_loss = 0.0
        processed = 0
        order = torch.randperm(len(train_refs)).tolist()
        for batch_start in range(0, len(order), batch_size):
            picks = [train_refs[i] for i in order[batch_start : batch_start + batch_size]]
            padded, lengths, targets, target_lengths = _clip_batch(picks, arrays)
            logits = head(padded.to(device))
            loss = ctc(
                logits.log_softmax(2).transpose(0, 1),
                targets.to(device),
                lengths,
                target_lengths,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
            processed += 1
            step += 1
        val_acc = _val_accuracy(head, arrays, val_refs, device)
        history.append(
            {
                "epoch": epoch,
                "loss": round(total_loss / max(processed, 1), 4),
                "val_accuracy": round(val_acc, 4),
            }
        )
        log.info("epoch %d: %s", epoch, history[-1])
        torch.save(
            {
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "history": history,
            },
            checkpoint,
        )
        (out_dir / "head-summary.json").write_text(
            json.dumps({"history": history}, indent=2), encoding="utf-8"
        )
    return {"history": history}


def _decode_ctc(preds: list[int] | object) -> list[int]:
    """Greedy CTC decode: collapse repeats, drop blanks."""
    decoded: list[int] = []
    previous: int | None = None
    for token in preds:  # type: ignore[union-attr]
        if token != BLANK and token != previous:
            decoded.append(int(token))
        previous = int(token)
    return decoded


def _val_accuracy(
    head: torch.nn.Module,
    arrays: dict[Path, np.ndarray],
    val_refs: list[ClipRef],
    device: str,
) -> float:
    """Greedy-decoded sequence accuracy on the val clips (exam-shaped proxy).

    Per-frame argmax compared position-wise to the token list is meaningless
    under CTC (blank-dominated alignments); the decoded sequence must match
    the target sequence exactly.
    """
    head.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for ref in val_refs:
            clip = ref.clip
            array = arrays[ref.shard.path]
            feats = array[clip["start"] : clip["start"] + clip["length"]]
            logits = head(torch.from_numpy(feats.astype(np.float32)).unsqueeze(0).to(device))
            preds = logits.argmax(-1)[0].cpu().tolist()
            targets = [TOKEN_INDEX[t] for t in clip["tokens"].split()]
            total += 1
            correct += 1 if _decode_ctc(preds) == targets else 0
    return correct / total if total else 0.0


def export_model(head_state: dict[str, torch.Tensor], base_model_dir: Path, out_dir: Path) -> None:
    """Assemble the engine-loadable ear: XLS-R base + trained CTC head."""
    from safetensors.torch import save_file  # noqa: PLC0415 - the ear env only
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC  # noqa: PLC0415

    model = Wav2Vec2ForCTC.from_pretrained(str(base_model_dir))
    with torch.no_grad():
        model.lm_head = torch.nn.Linear(model.config.hidden_size, NUM_CLASSES)
        model.lm_head.weight.copy_(head_state["weight"])
        model.lm_head.bias.copy_(head_state["bias"])
    model.config.vocab_size = NUM_CLASSES
    model.config.pad_token_id = BLANK
    model.config.ctc_loss_reduction = "mean"
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), model_dir / "model.safetensors")
    config_dict = Wav2Vec2Config.from_pretrained(str(base_model_dir)).to_dict()
    config_dict["vocab_size"] = NUM_CLASSES
    config_dict["pad_token_id"] = BLANK
    config_dict["ctc_loss_reduction"] = "mean"
    config_dict["architectures"] = ["Wav2Vec2ForCTC"]
    (model_dir / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    preprocessor = base_model_dir / "preprocessor_config.json"
    if not preprocessor.is_file():
        raise FileNotFoundError(f"{preprocessor}: the base model dir must include it")
    shutil.copyfile(preprocessor, model_dir / "preprocessor_config.json")
    (model_dir / "vocab.json").write_text(json.dumps(VOCAB, indent=2), encoding="utf-8")
    log.info("exported ear to %s", model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="ear_prep output dir")
    parser.add_argument("--base-model", required=True, type=Path, help="xls-r-300m model dir")
    parser.add_argument("--out", required=True, type=Path, help="run dir (runs/ear-v1)")
    parser.add_argument("--stage", required=True, choices=["extract", "train", "export"])
    parser.add_argument("--max-clips", type=int, default=None, help="smoke cap for extract")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "extract":
        extract(args.data, args.base_model, args.out, max_clips=args.max_clips)
    elif args.stage == "train":
        train(args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    else:
        checkpoint = args.out / "head-last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{checkpoint}: run --stage train first")
        state = torch.load(checkpoint, map_location="cpu")
        export_model(state["head"], args.base_model, args.out)


if __name__ == "__main__":
    main()
