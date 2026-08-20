"""Fine-tune the 4-class /ð/ contrast classifier with 10-dim DSP Feature Fusion.

Fuses Wav2Vec 2.0 acoustic representations with a 10-dimensional deterministic
DSP feature vector (spectral ratio >4kHz, stop closure dip, burst sharpness,
burst contrast, sibilance prominence, voicing continuity, spectral centroid,
ZCR, rolloff, flatness) and fine-tunes using Cross-Entropy loss with target
class boosting.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import shutil
import subprocess
import wave
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from openschwa_engine.measurements.features import extract_acoustic_features
from torch import nn

log = logging.getLogger("openschwa-training")

ALPHABET = ("ð", "z", "d", "v")
#: vocab.json layout for 4-class sequence classifier
VOCAB = {phone: i for i, phone in enumerate(ALPHABET)}


class PhoneContrastClassifier(nn.Module):
    """Wav2Vec2 + DSP Acoustic Feature Fusion Classifier."""

    def __init__(
        self,
        base_model_or_dir: str | Path | nn.Module,
        num_classes: int = 4,
        num_features: int = 10,
    ) -> None:
        super().__init__()
        from transformers import Wav2Vec2Model  # noqa: PLC0415

        if isinstance(base_model_or_dir, (str, Path)):
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(str(base_model_or_dir))
        else:
            self.wav2vec2 = base_model_or_dir
        self.num_classes = num_classes
        self.num_features = num_features
        hidden_size = self.wav2vec2.config.hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2 + num_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
        self.config = self.wav2vec2.config
        self.config.architectures = ["PhoneContrastClassifier"]
        self.config.num_labels = num_classes

    def freeze_feature_encoder(self) -> None:
        if hasattr(self.wav2vec2, "feature_extractor"):
            self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [batch_size, time, hidden_size]

        if attention_mask is not None:
            feat_lengths = self.wav2vec2._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            max_time = hidden.shape[1]
            frame_mask = (
                torch.arange(max_time, device=hidden.device).unsqueeze(0)
                < feat_lengths.unsqueeze(1)
            ).float().unsqueeze(-1)
            mean_p = (hidden * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp(min=1.0)
            masked_h = hidden.clone()
            masked_h[frame_mask.squeeze(-1) == 0] = -1e9
            max_p, _ = masked_h.max(dim=1)
        else:
            mean_p = hidden.mean(dim=1)
            max_p, _ = hidden.max(dim=1)

        pooled = torch.cat([mean_p, max_p], dim=-1)
        if features is not None:
            fused = torch.cat([pooled, features], dim=-1)
        else:
            zero_feat = torch.zeros(pooled.shape[0], self.num_features, device=pooled.device)
            fused = torch.cat([pooled, zero_feat], dim=-1)

        return self.fusion(fused)


@dataclass(frozen=True)
class TrainOptions:
    data_dirs: list[Path]  # export dirs to merge: labels.csv + audio/ each
    base_model_dir: Path  # assembled charsiu dir (config/preprocessor/weights/vocab)
    out_dir: Path
    epochs: int = 12
    freeze_epochs: int = 4
    batch_size: int = 32
    lr_head: float = 1e-3
    lr_full: float = 2e-5
    seed: int = 42
    max_steps: int | None = None  # smoke runs
    max_segment_s: float = 0.5  # pad/truncate ceiling
    use_amp: bool = True  # bf16 autocast on CUDA; disable for fp32 experiments
    label_smoothing: float = 0.05
    target_boost: float = 1.0
    #: Repeat every hard-negative row this many times in the train pool: the
    #: precision wall is made of exactly these tokens, and the balanced
    #: sampler's per-class weight alone underserves them.
    hardneg_mult: int = 1
    #: Speed/noise augmentation of non-target (error) segments: the recall
    #: wall is error-confidence starvation, and the error classes are the
    #: smallest pools we have.
    augment: bool = False
    #: Fraction of total steps spent in linear LR warmup (the rest is a cosine
    #: decay). One optimizer lives across the whole run now; the warmup keeps
    #: the first steps from burning the schedule.
    warmup_frac: float = 0.1


def load_dataset(data_dirs: list[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Merge every export dir's labels.csv; each row remembers its source."""
    rows: list[dict[str, object]] = []
    for data_dir in data_dirs:
        for row in csv.DictReader((data_dir / "labels.csv").open(encoding="utf-8")):
            row["_source"] = str(data_dir)
            rows.append(row)
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    if not train or not val:
        raise ValueError(
            f"{[str(d) for d in data_dirs]} needs both train and val rows, got "
            f"{len(train)}/{len(val)} - re-run the exports without --val-fraction 0"
        )
    return train, val


def read_segment_and_features(row: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """16 kHz float32 samples and 10-dim DSP feature vector for one segment."""
    with wave.open(str(Path(row["_source"]) / row["filename"]), "rb") as handle:
        assert handle.getframerate() == 16_000, row["filename"]
        assert handle.getnchannels() == 1, row["filename"]
        samples = (
            np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32)
            / 32768.0
        )
    features = extract_acoustic_features(samples)
    return samples, features


def augment_segment(samples: np.ndarray, rng: random.Random) -> np.ndarray:
    """Speed-stretch and/or noise a segment; the length is preserved.

    Fricatives survive mild resampling stretch and noise, which is the point:
    the error classes (z/d/v) are the smallest pools, and their acoustic
    variety is exactly what the recall wall is missing.
    """
    samples = samples.copy()
    if rng.random() < 0.6:  # speed stretch, 0.9x-1.1x
        factor = rng.uniform(0.9, 1.1)
        n = len(samples)
        stretched = np.interp(np.linspace(0, n - 1, int(n / factor)), np.arange(n), samples)
        samples = (
            stretched[:n]
            if stretched.size >= n
            else np.pad(stretched, (0, n - stretched.size))
        )
    if rng.random() < 0.6:  # gaussian noise at 15-30 dB SNR
        signal_rms = float(np.sqrt(np.mean(samples**2) + 1e-9))
        noise_rms = signal_rms / (10 ** (rng.uniform(15.0, 30.0) / 20.0))
        noise = np.array([rng.gauss(0.0, noise_rms) for _ in range(samples.size)])
        samples = samples + noise.reshape(samples.shape)
    return samples.astype(np.float32)


def class_weight(train: list[dict[str, object]], target_boost: float = 1.0) -> list[float]:
    counts = {label: 0 for label in ALPHABET}
    for row in train:
        counts[str(row["label"])] += 1
    total = sum(counts.values())
    weights = [total / (len(ALPHABET) * counts[label]) for label in ALPHABET]
    weights[0] *= target_boost
    return weights


def _freeze_base(model: PhoneContrastClassifier, freeze: bool) -> None:
    for name, param in model.named_parameters():
        # The conv feature encoder stays frozen FOREVER: freeze_feature_encoder
        # froze it at build time, and the old code unfroze it again at the
        # unfreeze epoch - undoing the design's regularization on every run.
        if "fusion" not in name and "feature_extractor" not in name:
            param.requires_grad = not freeze


def build_model(base_model_dir: Path, device: str) -> PhoneContrastClassifier:
    """A PhoneContrastClassifier over the charsiu base with 10-dim DSP fusion."""
    model = PhoneContrastClassifier(base_model_dir, num_classes=len(VOCAB), num_features=10)
    model.freeze_feature_encoder()
    model.config.apply_spec_augment = False
    return model.to(device)


def collate(
    batch: list[tuple[np.ndarray, np.ndarray, int]], device: str, max_samples: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    """Pad a batch of (samples, dsp_features, label) to tensors."""
    audio = []
    dsp_feats = []
    labels = []
    lengths = []
    for samples, feat, label in batch:
        trimmed = samples[:max_samples]
        audio.append(torch.from_numpy(trimmed).float())
        dsp_feats.append(torch.from_numpy(feat).float())
        labels.append(label)
        lengths.append(len(trimmed))
    padded = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0.0).to(device)
    dsp_tensor = torch.stack(dsp_feats).to(device)
    max_length = padded.shape[1]
    mask = (
        torch.arange(max_length, device=device).unsqueeze(0)
        < torch.tensor(lengths, device=device).unsqueeze(1)
    ).long()
    return padded, dsp_tensor, torch.tensor(labels, dtype=torch.long, device=device), lengths, mask


def val_metrics(
    model: PhoneContrastClassifier,
    val: list[dict[str, object]],
    options: TrainOptions,
    device: str,
) -> tuple[float, float, dict[str, dict[str, object]]]:
    """Per-class accuracy + macro F1 over the val split."""
    model.eval()
    expected: list[int] = []
    predicted: list[int] = []
    max_samples = int(options.max_segment_s * 16_000)
    with torch.inference_mode():
        for start in range(0, len(val), options.batch_size):
            chunk = val[start : start + options.batch_size]
            batch = [
                (*read_segment_and_features(row), VOCAB[str(row["label"])]) for row in chunk
            ]
            audio, dsp_feats, labels, _lengths, mask = collate(batch, device, max_samples)
            logits = model(audio, attention_mask=mask, features=dsp_feats)
            preds = torch.argmax(logits, dim=-1)
            expected.extend(labels.cpu().tolist())
            predicted.extend(preds.cpu().tolist())

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


def _tie_corrected_auc(pairs: list[tuple[float, int]]) -> float:
    """Rank-sum AUC with tie correction (same maths as the eval harness)."""
    if not pairs:
        return float("nan")
    n_pos = sum(1 for _, label in pairs if label)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    ordered = sorted(pairs)
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end + 1) / 2
        for position in range(index, end + 1):
            if ordered[position][1]:
                rank_sum += average_rank
        index = end + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def val_dh_slot_metrics(
    model: PhoneContrastClassifier, val: list[dict[str, object]], options: TrainOptions, device: str
) -> tuple[float, float]:
    """Exam-shaped /ð/-slot metrics over the val split.

    Only rows whose target_phone is /ð/ count (canonical z/d/other rows are
    curriculum, not exam proxy): positives are /ð/ realizations of anything
    other than /ð/, and the score mirrors the engine's contrast score
    log(p_best_other / p_ð). Returns (tie-corrected AUC, precision at
    recall >= 0.4 on the same ranking).
    """
    max_samples = int(options.max_segment_s * 16_000)
    dh_rows = [row for row in val if row.get("target_phone") == "ð"]
    if not dh_rows:
        return float("nan"), 0.0
    scores: list[float] = []
    labels: list[int] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dh_rows), options.batch_size):
            chunk = dh_rows[start : start + options.batch_size]
            batch = [
                (*read_segment_and_features(row), VOCAB[str(row["label"])]) for row in chunk
            ]
            audio, dsp_feats, _labels, _lengths, mask = collate(batch, device, max_samples)
            logits = model(audio, attention_mask=mask, features=dsp_feats).float()
            probs = torch.softmax(logits, dim=-1)
            dh_index = VOCAB["ð"]
            others = [i for i in range(len(VOCAB)) if i != dh_index]
            p_other = probs[:, others].max(dim=1).values
            p_dh = probs[:, dh_index].clamp(min=1e-9)
            scores.extend(torch.log(p_other / p_dh).cpu().tolist())
            labels.extend(1 if str(row["label"]) != "ð" else 0 for row in chunk)
    auc = _tie_corrected_auc(list(zip(scores, labels, strict=True)))
    ordered = sorted(zip(scores, labels, strict=True), reverse=True)
    n_pos = sum(labels)
    best_precision = 0.0
    tp = 0
    fp = 0
    for _, label in ordered:
        tp += label
        fp += 1 - label
        if tp >= 0.4 * n_pos and n_pos:
            best_precision = tp / (tp + fp)
            break
    return auc, best_precision


def balanced_batches(
    rows: list[dict[str, object]], batch_size: int, rng: random.Random
) -> list[list[dict[str, object]]]:
    """Per-class balanced batches."""
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
    from torch.nn import functional as F  # noqa: PLC0415

    torch.manual_seed(options.seed)
    random.seed(options.seed)
    np.random.seed(options.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device: %s", device)

    train_rows, val_rows = load_dataset(options.data_dirs)
    if options.hardneg_mult > 1:
        hard = [row for row in train_rows if "hardneg" in str(row["_source"])]
        train_rows = train_rows + hard * (options.hardneg_mult - 1)
        log.info("hard negatives x%d (%d rows)", options.hardneg_mult, len(hard))
    weights = class_weight(train_rows, target_boost=options.target_boost)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    max_samples = int(options.max_segment_s * 16_000)

    model = build_model(options.base_model_dir, device)
    options.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = options.out_dir / "last.pt"

    # Per-run provenance: the exact recipe, the code revision, and the data
    # dirs this run consumed. Slot reuse made v13-v23 unreproducible; this is
    # the fix, and it is written BEFORE any training happens.
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
        check=False,
    ).stdout.strip()
    run_config = {
        "git_sha": git_sha,
        "data_dirs": [str(d) for d in options.data_dirs],
        "epochs": options.epochs,
        "freeze_epochs": options.freeze_epochs,
        "batch_size": options.batch_size,
        "lr_head": options.lr_head,
        "lr_full": options.lr_full,
        "label_smoothing": options.label_smoothing,
        "target_boost": options.target_boost,
        "hardneg_mult": options.hardneg_mult,
        "augment": options.augment,
        "warmup_frac": options.warmup_frac,
        "seed": options.seed,
        "max_segment_s": options.max_segment_s,
    }
    (options.out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    start_epoch = 0
    best_f1 = -1.0
    best_select = float("nan")  # /ð/-slot val AUC when available, else val F1
    history: list[dict[str, object]] = []
    step = 0
    optimizer_state = None
    scheduler_state = None
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"]
        best_f1 = state.get("best_f1", -1.0)
        best_select = state.get("best_select", float("nan"))
        history = state.get("history", [])
        step = state.get("step", 0)
        optimizer_state = state.get("optimizer")
        scheduler_state = state.get("scheduler")
        log.info("resumed from %s at epoch %d", checkpoint, start_epoch)

    # ONE optimizer for the whole run (the old code rebuilt AdamW inside the
    # epoch loop - moments reset every epoch, twelve cold restarts per run).
    # Two param groups: the fusion head at lr_head, the transformer base at 0
    # while frozen and lr_full afterwards; the conv feature encoder is never
    # trained. Warmup+cosine rides the whole step budget.
    head_params = [p for n, p in model.named_parameters() if "fusion" in n]
    base_params = [
        p
        for n, p in model.named_parameters()
        if "fusion" not in n and "feature_extractor" not in n
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": options.lr_head, "weight_decay": 0.01},
            {"params": base_params, "lr": 0.0, "weight_decay": 0.01},
        ],
        lr=options.lr_head,
        weight_decay=0.01,
    )
    batches_per_epoch = len(
        balanced_batches(train_rows, options.batch_size, random.Random(options.seed))
    )
    total_steps = max(1, options.epochs * batches_per_epoch)
    warmup_steps = max(1, int(total_steps * options.warmup_frac))

    def _schedule_at(step_index: int) -> float:
        if step_index < warmup_steps:
            return step_index / warmup_steps
        progress = (step_index - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [_schedule_at, _schedule_at])
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    for epoch in range(start_epoch, options.epochs):
        freeze = epoch < options.freeze_epochs
        _freeze_base(model, freeze)
        optimizer.param_groups[0]["lr"] = options.lr_head
        optimizer.param_groups[1]["lr"] = 0.0 if freeze else options.lr_full
        model.train()
        rng = random.Random(options.seed * 1000 + epoch)
        total_loss = 0.0
        processed = 0
        for chunk in balanced_batches(train_rows, options.batch_size, rng):
            batch = []
            for row in chunk:
                samples, feats = read_segment_and_features(row)
                if options.augment and str(row["label"]) != "ð" and rng.random() < 0.5:
                    samples = augment_segment(samples, rng)
                    feats = extract_acoustic_features(samples)
                batch.append((samples, feats, VOCAB[str(row["label"])]))
            audio, dsp_feats, labels, _lengths, mask = collate(batch, device, max_samples)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device == "cuda" and options.use_amp
                else nullcontext()
            )
            with context:
                logits = model(audio, attention_mask=mask, features=dsp_feats)
            loss = F.cross_entropy(
                logits.float(),
                labels,
                weight=weights_tensor,
                label_smoothing=options.label_smoothing,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
            processed += 1
            step += 1
            if options.max_steps is not None and step >= options.max_steps:
                break
        accuracy, f1, per_class = val_metrics(model, val_rows, options, device)
        dh_auc, dh_p_at_r04 = val_dh_slot_metrics(model, val_rows, options, device)
        history.append(
            {
                "epoch": epoch,
                "freeze": freeze,
                "loss": round(total_loss / max(processed, 1), 4),
                "val_accuracy": round(accuracy, 4),
                "val_f1": round(f1, 4),
                "val_dh_auc": round(dh_auc, 4) if not math.isnan(dh_auc) else None,
                "per_class": per_class,
            }
        )
        log.info("epoch %d: %s", epoch, history[-1])
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "step": step,
                "best_f1": best_f1,
                "best_select": best_select,
                "history": history,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            checkpoint,
        )
        # Exam-shaped model selection: the /ð/-slot AUC on the val split, the
        # number the exam actually measures. Canonical z/d/other rows are
        # curriculum and never select the model. When the val split has no
        # /ð/-slot positives (smoke fixtures), fall back to val F1.
        select = dh_auc if not math.isnan(dh_auc) else f1
        if math.isnan(best_select) or select > best_select:
            best_select = select
            best_f1 = f1
            export_model(model, options.out_dir / "model", options.base_model_dir)
        if options.max_steps is not None and step >= options.max_steps:
            break

    summary = {
        "best_select": round(best_select, 4) if not math.isnan(best_select) else None,
        "best_val_f1": round(best_f1, 4),
        "history": history,
        "vocab": VOCAB,
        "class_weights": weights,
    }
    (options.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def export_model(model: PhoneContrastClassifier, out_dir: Path, base_model_dir: Path) -> None:
    """Assemble the engine-loadable layout: config, preprocessor, vocab, weights."""
    from safetensors.torch import save_file  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    config_dict = model.wav2vec2.config.to_dict()
    config_dict["architectures"] = ["PhoneContrastClassifier"]
    config_dict["num_labels"] = len(VOCAB)
    config_dict["id2label"] = {i: p for p, i in VOCAB.items()}
    config_dict["label2id"] = VOCAB
    config_dict["num_features"] = model.num_features
    (out_dir / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    save_file(model.state_dict(), out_dir / "model.safetensors")
    preprocessor = base_model_dir / "preprocessor_config.json"
    if not preprocessor.is_file():
        raise FileNotFoundError(
            f"{preprocessor}: the base model dir must include preprocessor_config.json"
        )
    shutil.copyfile(preprocessor, out_dir / "preprocessor_config.json")
    (out_dir / "vocab.json").write_text(json.dumps(VOCAB), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        action="append",
        help="export dir with labels.csv + audio/ (repeat to merge datasets)",
    )
    parser.add_argument("--base-model", required=True, help="assembled charsiu model dir")
    parser.add_argument("--out", required=True, help="checkpoints + final model dir")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--freeze-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-full", type=float, default=2e-5)
    parser.add_argument("--target-boost", type=float, default=1.5)
    #: CE label smoothing. The 0.90-precision bar lives in the high-confidence
    #: tail; smoothing compresses exactly that tail, so a bar-chasing run
    #: passes --label-smoothing 0.
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    parser.add_argument(
        "--hardneg-mult", type=int, default=1, help="repeat hard-negative rows N times"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="speed/noise augmentation of error segments",
    )
    parser.add_argument("--warmup-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None, help="smoke runs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = train(
        TrainOptions(
            data_dirs=[Path(d) for d in args.data],
            base_model_dir=Path(args.base_model),
            out_dir=Path(args.out),
            epochs=args.epochs,
            freeze_epochs=args.freeze_epochs,
            batch_size=args.batch_size,
            lr_head=args.lr_head,
            lr_full=args.lr_full,
            target_boost=args.target_boost,
            label_smoothing=args.label_smoothing,
            hardneg_mult=args.hardneg_mult,
            augment=args.augment,
            warmup_frac=args.warmup_frac,
            seed=args.seed,
            max_steps=args.max_steps,
            use_amp=not args.fp32,
        )
    )
    log.info("done; best val F1 %s", summary["best_val_f1"])


if __name__ == "__main__":
    main()
