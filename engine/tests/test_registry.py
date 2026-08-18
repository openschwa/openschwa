"""Model registry: readiness checks and local-model guards."""

import pytest

from openschwa_engine.models.registry import MANIFEST, ModelError, ModelRegistry

FIXED = ("config.json", "preprocessor_config.json", "vocab.json")


def _model_dir(tmp_path):
    return tmp_path / "models"


@pytest.mark.parametrize("weight_file", ["pytorch_model.bin", "model.safetensors"])
def test_is_ready_accepts_either_weight_format(tmp_path, weight_file):
    """transformers 5.x saves safetensors by default; the fine-tuned Option 3
    model uses it, so readiness must accept both serializations."""
    registry = ModelRegistry(_model_dir(tmp_path))
    spec = MANIFEST["dh-contrast-v1"]
    base = registry.local_dir(spec)
    base.mkdir(parents=True)
    for name in FIXED:
        (base / name).write_text("{}", encoding="utf-8")
    (base / weight_file).write_bytes(b"x")
    assert registry.is_ready(spec)


def test_a_local_model_cannot_be_pulled(tmp_path):
    """dh-contrast-v1 is built by training/, not a hub download: pulling it
    must fail with instructions, not with a network error."""
    registry = ModelRegistry(_model_dir(tmp_path))
    with pytest.raises(ModelError, match="built locally"):
        list(registry.pull(MANIFEST["dh-contrast-v1"]))
