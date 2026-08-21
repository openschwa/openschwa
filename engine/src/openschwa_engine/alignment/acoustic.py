"""Acoustic model loading and phoneme posteriors.

Everything torch-shaped is confined to this module so the rest of the pipeline
stays importable without the `ml` extra. The network is a lazily-created
singleton per model directory: loading a 1.2 GB CTC model takes seconds, and the
engine runs single-worker uvicorn precisely so one copy can be shared.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from openschwa_engine.audio import MODEL_SAMPLE_RATE
from openschwa_engine.models.registry import ModelError

log = logging.getLogger(__name__)

_cache: dict[Path, "AcousticModel"] = {}
_cache_lock = threading.Lock()


@dataclass(frozen=True)
class Posteriors:
    """Frame-wise log posteriors over the model vocabulary."""

    log_probs: npt.NDArray[np.float32]  # [frames, vocab]
    hop_s: float

    @property
    def frames(self) -> int:
        return int(self.log_probs.shape[0])


class AcousticModel:
    """Wav2Vec2-style CTC or Sequence Classification phoneme model over 16 kHz mono float32."""

    def __init__(self, model_dir: Path) -> None:
        try:
            import json  # noqa: PLC0415

            import torch  # noqa: PLC0415 - lazy `ml` extra
            from transformers import (  # noqa: PLC0415 - lazy `ml` extra
                AutoModelForAudioClassification,
                Wav2Vec2FeatureExtractor,
                Wav2Vec2ForCTC,
            )
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            # The cause is included deliberately: in a packaged build torch is
            # present but can still fail to import (a pruned submodule, a
            # missing dylib), and a bare "not installed" sends you hunting for
            # the wrong problem.
            raise ModelError(
                f"could not load torch/transformers ({exc}) — "
                "run `uv sync --extra ml` in engine/ if this is a source checkout"
            ) from exc

        self._torch = torch
        # CUDA when the machine has it: the eval harness and the training
        # exporters run the same engine code, and the laptop's GPU turns the
        # multi-hour corpus passes into minutes. Everything downstream stays
        # numpy on the CPU - only the forward pass moves.
        self._cuda = torch.cuda.is_available()
        log.info("loading acoustic model from %s (cuda=%s)", model_dir, self._cuda)
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir))

        config_path = model_dir / "config.json"
        model_type = "ctc"
        if config_path.is_file():
            try:
                config_data = json.loads(config_path.read_text(encoding="utf-8"))
                architectures = config_data.get("architectures", [])
                if "PhoneContrastClassifier" in architectures:
                    model_type = "fusion"
                elif "EarCTC" in architectures:
                    model_type = "ear"
                elif any(
                    "SequenceClassification" in a or "AudioClassification" in a
                    for a in architectures
                ):
                    model_type = "seq"
            except Exception:
                pass

        self._model_type = model_type
        if model_type == "fusion":
            from safetensors.torch import load_file
            from torch import nn
            from transformers import Wav2Vec2Config, Wav2Vec2Model

            class _PhoneContrastClassifier(nn.Module):
                def __init__(self, dir_path: Path, num_cls: int = 4, num_feat: int = 10) -> None:
                    super().__init__()
                    config = Wav2Vec2Config.from_pretrained(str(dir_path))
                    num_feat = int(getattr(config, "num_features", num_feat))
                    self.wav2vec2 = Wav2Vec2Model(config)
                    self.num_classes = num_cls
                    self.num_features = num_feat
                    hidden_size = config.hidden_size
                    self.fusion = nn.Sequential(
                        nn.Linear(hidden_size * 2 + num_feat, 512),
                        nn.LayerNorm(512),
                        nn.GELU(),
                        nn.Dropout(0.15),
                        nn.Linear(512, 256),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(256, num_cls),
                    )
                    self.config = config
                    self.config.architectures = ["PhoneContrastClassifier"]
                    self.config.num_labels = num_cls

                def forward(
                    self,
                    input_values: torch.Tensor,
                    attention_mask: torch.Tensor | None = None,
                    features: torch.Tensor | None = None,
                ) -> torch.Tensor:
                    outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
                    hidden = outputs.last_hidden_state
                    if attention_mask is not None:
                        feat_lengths = self.wav2vec2._get_feat_extract_output_lengths(
                            attention_mask.sum(dim=1).long()  # type: ignore[arg-type]
                        )
                        max_time = hidden.shape[1]
                        frame_mask = (
                            (
                                torch.arange(max_time, device=hidden.device).unsqueeze(0)
                                < feat_lengths.unsqueeze(1)
                            )
                            .float()
                            .unsqueeze(-1)
                        )
                        mean_p = (hidden * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp(
                            min=1.0
                        )
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
                        zero_feat = torch.zeros(
                            pooled.shape[0], self.num_features, device=pooled.device
                        )
                        fused = torch.cat([pooled, zero_feat], dim=-1)
                    return self.fusion(fused)

            model = _PhoneContrastClassifier(model_dir)
            safetensors_file = model_dir / "model.safetensors"
            if safetensors_file.is_file():
                model.load_state_dict(load_file(str(safetensors_file)))
            elif (model_dir / "pytorch_model.bin").is_file():
                model.load_state_dict(
                    torch.load(
                        model_dir / "pytorch_model.bin", map_location="cpu", weights_only=False
                    )
                )
            self._model: Any = model
        elif model_type == "ear":
            from safetensors.torch import load_file
            from torch import nn
            from transformers import Wav2Vec2Config, Wav2Vec2Model

            class _EarCTC(nn.Module):
                """The Phase 1 ear: frozen XLS-R + linear head over the mean of
                middle-layer hidden states (layers 12-20 - phone identity lives
                there; the final layer is tuned for the SSL objective)."""

                def __init__(self, dir_path: Path) -> None:
                    super().__init__()
                    config = Wav2Vec2Config.from_pretrained(str(dir_path))
                    self.wav2vec2 = Wav2Vec2Model(config)
                    layer_span = getattr(config, "ear_layers", None)
                    self.layers = (
                        (int(layer_span[0]), int(layer_span[1]))
                        if layer_span is not None
                        else (12, 21)
                    )
                    self.head = nn.Linear(config.hidden_size, config.vocab_size)
                    self.config = config
                    self.config.architectures = ["EarCTC"]

                def forward(
                    self,
                    input_values: torch.Tensor,
                    attention_mask: torch.Tensor | None = None,
                ) -> torch.Tensor:
                    outputs = self.wav2vec2(
                        input_values,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    features = torch.stack(
                        outputs.hidden_states[self.layers[0] : self.layers[1]], dim=0
                    ).mean(dim=0)
                    return self.head(features)

            model = _EarCTC(model_dir)
            safetensors_file = model_dir / "model.safetensors"
            if safetensors_file.is_file():
                model.load_state_dict(load_file(str(safetensors_file)))
            elif (model_dir / "pytorch_model.bin").is_file():
                model.load_state_dict(
                    torch.load(
                        model_dir / "pytorch_model.bin", map_location="cpu", weights_only=False
                    )
                )
            self._model: Any = model
        elif model_type == "seq":
            self._model = AutoModelForAudioClassification.from_pretrained(str(model_dir))
        else:
            self._model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))
        self._model.eval()
        if self._cuda:
            self._model = self._model.cuda()
        cfg = getattr(self._model, "config", None)
        v_size = getattr(cfg, "vocab_size", None) or getattr(cfg, "num_labels", 0)
        self.vocab_size = int(v_size or 0)

    def posteriors(self, samples_16k: npt.NDArray[np.float32]) -> Posteriors:
        if samples_16k.size == 0:
            raise ModelError("no audio to analyse")
        torch = self._torch

        inputs: Any = self._extractor(
            samples_16k,
            sampling_rate=MODEL_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if self._cuda:
            inputs["input_values"] = inputs["input_values"].cuda()
            inputs["attention_mask"] = inputs["attention_mask"].cuda()
        with torch.inference_mode():
            if self._model_type == "fusion":
                from openschwa_engine.measurements.features import extract_acoustic_features

                dsp_feat = extract_acoustic_features(samples_16k)
                feat_tensor = torch.from_numpy(dsp_feat).float().unsqueeze(0)
                if self._cuda:
                    feat_tensor = feat_tensor.cuda()
                logits = self._model(
                    inputs.input_values,
                    attention_mask=inputs.attention_mask,
                    features=feat_tensor,
                )[0]
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                log_probs = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            elif self._model_type == "ear":
                logits = self._model(
                    inputs.input_values, attention_mask=inputs.attention_mask
                )[0]
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                log_probs = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            elif self._model_type == "seq":
                logits = self._model(
                    inputs.input_values, attention_mask=inputs.attention_mask
                ).logits[0]
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                log_probs = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            else:
                logits = self._model(
                    inputs.input_values, attention_mask=inputs.attention_mask
                ).logits[0]
                log_probs = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

        frames = int(log_probs.shape[0])
        # Derive the hop from the actual frame count rather than assuming the
        # conv stack's 20 ms stride — the two disagree at the edges, and every
        # phone boundary the UI draws depends on this number.
        hop_s = (samples_16k.size / MODEL_SAMPLE_RATE) / max(frames, 1)
        return Posteriors(log_probs=log_probs.astype(np.float32), hop_s=hop_s)


def load(model_dir: Path) -> AcousticModel:
    """Get the shared model for `model_dir`, loading it on first use."""
    with _cache_lock:
        model = _cache.get(model_dir)
        if model is None:
            model = AcousticModel(model_dir)
            _cache[model_dir] = model
        return model
