"""Fine-tune the 4-class /ð/ contrast judge (Option 3, Step 2).

Loads the segment dataset exported by export.py, replaces the charsiu base's
CTC head with a fresh {blank, ð, z, d, v} head, and fine-tunes. Two phases:
the base stays frozen while the head learns (default), then everything
unfreezes at a lower learning rate. The output directory is assembled in the
exact layout the engine's model registry expects (config, preprocessor,
vocab.json, pytorch_model.bin), so Step 3's manifest entry can load it
without new pipeline code.

Runs on CPU (smoke tests) and CUDA (the laptop). Every decision is seeded;
checkpoints are resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import wave
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("openschwa-training")

ALPHABET = ("ð", "z", "d", "v")
#: vocab.json layout: [PAD]=blank, [UNK], then the four phones.
VOCAB = {"[PAD]": 0, "[UNK]": 1, **{phone: i + 2 for i, phone in enumerate(ALPHABET)}}


@dataclass(frozen=True)
class TrainOptions:
    data_dir: Path  # the export dir: labels.csv + audio/
    base_model_dir: Path  # assembled charsiu dir (config/preprocessor/weights/vocab)
    out_dir: Path
    epochs: int = 8
    freeze_epochs: int = 4
    batch_size: int = 32
    lr_head: float = 5e-4
    lr_full: float = 1e-5
    seed: int = 42
    max_steps: int | None = None  # smoke runs
    max_segment_s: float = 0.5  # pad/truncate ceiling
    num_workers: int = 0


def load_dataset(data_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = list(csv.DictReader((data_dir / "labels.csv").open(encoding="utf-8")))
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    if not train or not val:
        raise ValueError(
            f"{data_dir}/labels.csv needs both train and val rows, got "
            f"{len(train)}/{len(val)} - re-run export.py without --val-fraction 0"
        )
    return train, val


def read_segment(data_dir: Path, row: dict[str, object]) -> np.ndarray:
    """16 kHz float32 samples for one segment, padded to the ceiling length."""
    with wave.open(str(data_dir / row["filename"]), "rb") as handle:
        assert handle.getframerate() == 16_000, row["filename"]
        assert handle.getnchannels() == 1, row["filename"]
        samples = (
            np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32)
            / 32768.0
        )
    return samples


def class_weight(train: list[dict[str, object]]) -> list[float]:
    counts = {label: 0 for label in ALPHABET}
    for row in train:
        counts[str(row["label"])] += 1
    total = sum(counts.values())
    return [total / (len(ALPHABET) * counts[label]) for label in ALPHABET]


def _freeze_base(model, freeze: bool) -> None:
    for name, param in model.named_parameters():
        if "lm_head" not in name:
            param.requires_grad = not freeze


def build_model(base_model_dir: Path, device: str):
    """A Wav2Vec2ForCTC with a fresh 4-class head over the charsiu base."""
    from transformers import Wav2Vec2ForCTC  # noqa: PLC0415

    model = Wav2Vec2ForCTC.from_pretrained(
        str(base_model_dir),
        num_labels=len(VOCAB),
        ignore_mismatched_sizes=True,
        pad_token_id=VOCAB["[PAD]"],
        vocab_size=len(VOCAB),
    )
    model.config.ctc_loss_reduction = "mean"
    # SpecAugment's fixed mask length breaks on our 60-300 ms segments (a
    # mask longer than the sequence raises). Disabled for v1; the frozen-base
    # phase and per-class weights are the regularization instead.
    model.config.apply_spec_augment = False
    return model.to(device)


def collate(batch: list[tuple[np.ndarray, int]], device: str, max_samples: int):
    """Pad a batch of (samples, label) to a tensor pair."""
    import torch  # noqa: PLC0415 - ml extra

    audio = []
    labels = []
    lengths = []
    for samples, label in batch:
        trimmed = samples[:max_samples]
        audio.append(torch.from_numpy(trimmed).float())
        labels.append(label)
        lengths.append(len(trimmed))
    padded = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True).to(device)
    return padded, torch.tensor(labels, dtype=torch.long, device=device), lengths


def val_metrics(
    model, val: list[dict[str, object]], data_dir: Path, options: TrainOptions, device: str
):
    """Per-class accuracy + macro F1 over the val split.

    Batched: the per-row loop was the dominant CPU cost (one forward pass per
    ~100 ms segment), and batching cuts val time roughly 10x.
    """
    import torch  # noqa: PLC0415 - ml extra

    model.eval()
    expected: list[int] = []
    predicted: list[int] = []
    max_samples = int(options.max_segment_s * 16_000)
    with torch.inference_mode():
        for start in range(0, len(val), options.batch_size):
            chunk = val[start : start + options.batch_size]
            batch = [(read_segment(data_dir, row), VOCAB[str(row["label"])]) for row in chunk]
            audio, labels, _lengths = collate(batch, device, max_samples)
            logits = model(audio).logits
            probs = torch.softmax(logits, dim=-1)
            # The non-blank class with the most total mass over frames.
            class_mass = probs[:, :, 2:].sum(dim=1)
            expected.extend(int(label.item()) - 2 for label in labels)
            predicted.extend(int(value.item()) for value in torch.argmax(class_mass, dim=1))
    per_class = {}
    for name, index in [(phone, i) for i, phone in enumerate(ALPHABET)]:
        hits = sum(1 for e, p in zip(expected, predicted, strict=True) if p == index and e == index)
        total = sum(1 for e in expected if e == index)
        per_class[name] = {"acc": round(hits / total, 4) if total else None, "n": total}
    correct = sum(1 for e, p in zip(expected, predicted, strict=True) if e == p)
    accuracy = correct / len(expected)
    f1s = []
    for index in range(len(ALPHABET)):
        tp = sum(1 for e, p in zip(expected, predicted, strict=True) if e == index and p == index)
        fp = sum(1 for e, p in zip(expected, predicted, strict=True) if e != index and p == index)
        fn = sum(1 for e, p in zip(expected, predicted, strict=True) if e == index and p != index)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return accuracy, float(np.mean(f1s)), per_class


def balanced_batches(
    rows: list[dict[str, object]], batch_size: int, rng: random.Random
) -> list[list[dict[str, object]]]:
    """Per-class balanced batches.

    Round-robin from shuffled per-class pools (with replacement): every batch
    carries roughly equal counts of each phone. Without this, the biggest
    class (d: ~28% of rows) is the shortest path to a low loss, and the head
    collapses to predicting it - exactly what the v1 CPU run did.
    """
    by_class: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_class.setdefault(str(row["label"]), []).append(row)
    pools = {label: pool[:] for label, pool in by_class.items()}
    for pool in pools.values():
        rng.shuffle(pool)
    per_class = max(1, batch_size // len(ALPHABET))
    batches: list[list[dict[str, object]]] = []
    for _ in range((len(rows) + batch_size - 1) // batch_size):
        batch: list[dict[str, object]] = []
        for label in ALPHABET:
            pool = pools[label]
            if len(pool) < per_class:  # refill with replacement
                pool.extend(by_class[label])
                rng.shuffle(pool)
            batch.extend(pool[:per_class])
            del pool[:per_class]
        rng.shuffle(batch)
        batches.append(batch[:batch_size])
    return batches


def train(options: TrainOptions) -> dict[str, object]:
    """Run (or resume) the fine-tune; returns the final metrics summary."""
    import torch  # noqa: PLC0415 - ml extra
    from torch.nn import functional as F  # noqa: PLC0415

    torch.manual_seed(options.seed)
    random.seed(options.seed)
    np.random.seed(options.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device: %s", device)

    train_rows, val_rows = load_dataset(options.data_dir)
    weights = class_weight(train_rows)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    max_samples = int(options.max_segment_s * 16_000)

    model = build_model(options.base_model_dir, device)
    options.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = options.out_dir / "last.pt"

    start_epoch = 0
    best_f1 = -1.0
    history: list[dict[str, object]] = []
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"]
        best_f1 = state.get("best_f1", -1.0)
        history = state.get("history", [])
        log.info("resumed from %s at epoch %d", checkpoint, start_epoch)

    step = 0
    for epoch in range(start_epoch, options.epochs):
        freeze = epoch < options.freeze_epochs
        _freeze_base(model, freeze)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=options.lr_head if freeze else options.lr_full, weight_decay=0.01
        )
        model.train()
        rng = random.Random(options.seed * 1000 + epoch)
        total_loss = 0.0
        processed = 0
        for chunk in balanced_batches(train_rows, options.batch_size, rng):
            batch = [
                (read_segment(options.data_dir, row), VOCAB[str(row["label"])]) for row in chunk
            ]
            audio, labels, lengths = collate(batch, device, max_samples)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
            )
            with context:
                logits = model(audio).logits
            log_probs = F.log_softmax(logits.float(), dim=-1)  # CTC loss runs in fp32
            input_lengths = torch.tensor(
                [logits.shape[1]] * len(labels), dtype=torch.long, device=device
            )
            target_lengths = torch.ones_like(labels)
            per_element = F.ctc_loss(
                log_probs.transpose(0, 1),
                labels,
                input_lengths,
                target_lengths,
                blank=VOCAB["[PAD]"],
                reduction="none",
            )
            # torch's ctc_loss has no per-class weight argument: apply the
            # class weights to the per-element losses instead.
            loss = (per_element * weights_tensor[labels - 2]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            processed += 1
            step += 1
            if options.max_steps is not None and step >= options.max_steps:
                break
        accuracy, f1, per_class = val_metrics(model, val_rows, options.data_dir, options, device)
        history.append(
            {
                "epoch": epoch,
                "freeze": freeze,
                "loss": round(total_loss / max(processed, 1), 4),
                "val_accuracy": round(accuracy, 4),
                "val_f1": round(f1, 4),
                "per_class": per_class,
            }
        )
        log.info("epoch %d: %s", epoch, history[-1])
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "best_f1": best_f1,
                "history": history,
            },
            checkpoint,
        )
        if f1 > best_f1:
            best_f1 = f1
            export_model(model, options.out_dir / "model", options.base_model_dir)
        if options.max_steps is not None and step >= options.max_steps:
            break

    summary = {
        "best_val_f1": round(best_f1, 4),
        "history": history,
        "vocab": VOCAB,
        "class_weights": weights,
    }
    (options.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def export_model(model, out_dir: Path, base_model_dir: Path) -> None:
    """Assemble the engine-loadable layout: config, preprocessor, vocab, weights."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    preprocessor = base_model_dir / "preprocessor_config.json"
    if not preprocessor.is_file():
        raise FileNotFoundError(
            f"{preprocessor}: the base model dir must include preprocessor_config.json"
        )
    shutil.copyfile(preprocessor, out_dir / "preprocessor_config.json")
    (out_dir / "vocab.json").write_text(json.dumps(VOCAB), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="export dir with labels.csv + audio/")
    parser.add_argument("--base-model", required=True, help="assembled charsiu model dir")
    parser.add_argument("--out", required=True, help="checkpoints + final model dir")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--freeze-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--lr-full", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None, help="smoke runs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = train(
        TrainOptions(
            data_dir=Path(args.data),
            base_model_dir=Path(args.base_model),
            out_dir=Path(args.out),
            epochs=args.epochs,
            freeze_epochs=args.freeze_epochs,
            batch_size=args.batch_size,
            lr_head=args.lr_head,
            lr_full=args.lr_full,
            seed=args.seed,
            max_steps=args.max_steps,
        )
    )
    log.info("done; best val F1 %s", summary["best_val_f1"])


if __name__ == "__main__":
    main()
