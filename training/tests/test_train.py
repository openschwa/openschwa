"""Training smoke test: two steps on a tiny synthetic model and dataset.

CPU-only by design (CI-compatible); the real run happens on the laptop's GPU.
"""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openschwa_training.train import VOCAB, TrainOptions, balanced_batches, train  # noqa: E402


def test_balanced_batches_carry_all_four_classes():
    import random as random_module

    rows = [
        {"label": ["ð", "z", "d", "v"][index % 4], "filename": f"u{index}.wav"}
        for index in range(12)
    ]
    batches = balanced_batches(rows, 8, random_module.Random(1))
    assert len(batches) == 2
    for batch in batches:
        assert {row["label"] for row in batch} == {"ð", "z", "d", "v"}


def tiny_base(tmp_path: Path) -> Path:
    from transformers import Wav2Vec2Config, Wav2Vec2Model  # noqa: PLC0415

    config = Wav2Vec2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_conv_layers=2,
        conv_dim=[32, 32],
        conv_kernel=[3, 3],
        conv_stride=[2, 2],
        feat_extract_norm="layer",
        num_feat_extract_layers=2,
    )
    model = Wav2Vec2Model(config)
    base = tmp_path / "base"
    model.save_pretrained(str(base))
    (base / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "do_normalize": True,
                "feature_extractor_type": "Wav2Vec2FeatureExtractor",
                "feature_size": 1,
                "padding_side": "right",
                "padding_value": 0,
                "return_attention_mask": True,
                "sampling_rate": 16000,
            }
        ),
        encoding="utf-8",
    )
    return base


def tiny_dataset(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    audio = data / "audio"
    audio.mkdir(parents=True)
    rng = np.random.RandomState(0)
    rows = []
    for index in range(8):
        samples = (rng.normal(0, 0.02, 1280) * 32767).astype("<i2")
        with wave.open(str(audio / f"u{index}.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(samples.tobytes())
        label = ["ð", "z", "d", "v"][index % 4]
        split = "train" if index < 4 else "val"
        rows.append(
            {
                "filename": f"audio/u{index}.wav",
                "label": label,
                "l1": "test",
                "utterance_id": f"u{index}",
                "token_index": 0,
                "target_phone": label,
                "start_s": 0.0,
                "end_s": 0.08,
                "duration_s": 0.08,
                "split": split,
            }
        )
    with (data / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return data


def test_smoke_training_runs_end_to_end(tmp_path):
    base = tiny_base(tmp_path)
    data = tiny_dataset(tmp_path)
    out = tmp_path / "out"
    summary = train(
        TrainOptions(
            data_dirs=[data],
            base_model_dir=base,
            out_dir=out,
            epochs=2,
            freeze_epochs=1,
            batch_size=4,
            max_steps=2,
            max_segment_s=0.1,
        )
    )
    assert "history" in summary
    assert (out / "model" / "config.json").is_file()
    assert (out / "model" / "preprocessor_config.json").is_file()
    vocab = json.loads((out / "model" / "vocab.json").read_text(encoding="utf-8"))
    assert vocab == VOCAB
    assert (out / "model" / "pytorch_model.bin").is_file() or (
        out / "model" / "model.safetensors"
    ).is_file()
