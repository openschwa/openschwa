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
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    base = tmp_path / "base"
    base.mkdir()
    config = Wav2Vec2Config(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=24,  # the ear slices hidden_states[12:21]
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
    # Weights too: export_model() loads the base with from_pretrained, which
    # requires them even though the test only exercises the head geometry.
    from safetensors.torch import save_file

    save_file(Wav2Vec2Model(config).state_dict(), base / "model.safetensors")
    return base


def test_training_and_engine_classifiers_produce_identical_logits(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    base = _tiny_base(tmp_path)
    config = Wav2Vec2Config.from_pretrained(str(base))
    model = PhoneContrastClassifier(Wav2Vec2Model(config), num_classes=len(VOCAB), num_features=10)
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


def test_ear_export_loads_in_the_engine_with_identical_logits(tmp_path):
    """The EarCTC lockstep: the exported ear must produce the same logits in
    the engine (mean of hidden layers 12-20 + head) as the training recipe.
    A silent mismatch here is the exact failure that produced the 4% exam."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    base = _tiny_base(tmp_path)
    config = Wav2Vec2Config.from_pretrained(str(base))
    # The SAME base weights the export loads from disk - a fresh random
    # encoder here would compare different models.
    encoder = Wav2Vec2Model.from_pretrained(str(base)).eval()
    head = torch.nn.Linear(config.hidden_size, 40)
    with torch.no_grad():
        head.weight.normal_(0, 0.1)
        head.bias.normal_(0, 0.1)

    from openschwa_training.ear_train import export_model

    out = tmp_path / "ear"
    export_model({"weight": head.weight, "bias": head.bias}, base, out)

    from openschwa_engine.alignment.acoustic import AcousticModel

    engine = AcousticModel(out / "model")  # export_model writes out/model/
    rng = np.random.RandomState(0)
    audio = (rng.randn(4000) * 0.1).astype(np.float32)  # 0.25 s
    # The engine normalizes through Wav2Vec2FeatureExtractor; the training
    # path must feed the SAME normalized input or the parity check compares
    # different waveforms.
    from transformers import Wav2Vec2FeatureExtractor

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(base))
    normalized = extractor(audio, sampling_rate=16_000, return_tensors="pt").input_values
    with torch.inference_mode():
        hidden = encoder(
            normalized,
            attention_mask=torch.ones_like(normalized),
            output_hidden_states=True,
        ).hidden_states
        features = torch.stack(hidden[12:21], dim=0).mean(dim=0)
        train_logits = head(features)
    train_probs = torch.softmax(train_logits.float(), dim=-1).numpy()
    engine_probs = np.exp(engine.posteriors(audio).log_probs)
    # The engine returns per-frame posteriors [frames, vocab]; the training
    # path carries a batch dim.
    assert engine_probs.shape == train_probs.shape[1:]
    assert np.allclose(engine_probs, train_probs[0], atol=2e-3)
