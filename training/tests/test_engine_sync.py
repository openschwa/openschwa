"""Training/engine classifier lockstep: the two copies of
PhoneContrastClassifier must produce identical probabilities, or a training
run would optimize a geometry the engine never uses (the silent-corruption
failure mode the registry phone tables guard against on the vocab side).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from openschwa_engine.measurements.features import extract_acoustic_features

from openschwa_training.train import VOCAB, PhoneContrastClassifier, export_model


def _tiny_base(tmp_path: Path) -> Path:
    from transformers import Wav2Vec2Config

    base = tmp_path / "base"
    base.mkdir()
    config = Wav2Vec2Config(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_conv_pos_embeddings=64,
        num_conv_pos_embedding_groups=2,
        conv_dim=(32, 32, 32, 32, 32, 32, 32),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        feat_extract_norm="layer",
    )
    (base / "config.json").write_text(config.to_json_string(), encoding="utf-8")
    (base / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "feature_size": 1,
                "sampling_rate": 16000,
                "do_normalize": True,
                "return_attention_mask": True,
            }
        ),
        encoding="utf-8",
    )
    return base


def test_training_and_engine_classifiers_produce_identical_logits(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    base = _tiny_base(tmp_path)
    config = Wav2Vec2Config.from_pretrained(str(base))
    model = PhoneContrastClassifier(
        Wav2Vec2Model(config), num_classes=len(VOCAB), num_features=10
    )
    model.eval()
    out = tmp_path / "model"
    export_model(model, out, base)

    from openschwa_engine.alignment.acoustic import AcousticModel

    engine = AcousticModel(out)
    rng = np.random.RandomState(0)
    audio = (rng.randn(4000) * 0.1).astype(np.float32)  # 0.25 s
    feats = extract_acoustic_features(audio)
    with torch.inference_mode():
        logits = model(
            torch.from_numpy(audio).unsqueeze(0),
            attention_mask=torch.ones(1, 4000),
            features=torch.from_numpy(feats).float().unsqueeze(0),
        )
    train_probs = torch.softmax(logits.float(), dim=-1).numpy()
    engine_probs = np.exp(engine.posteriors(audio).log_probs)
    assert engine_probs.shape == train_probs.shape
    # Not bit-exact: the engine normalizes the waveform via the feature
    # extractor in numpy, training via the model's internal torch path -
    # float32 ordering drift. The lockstep contract is geometry, not bits.
    assert np.allclose(engine_probs, train_probs, atol=2e-3)