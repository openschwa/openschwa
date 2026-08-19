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
    """Wav2Vec2-style CTC phoneme model over 16 kHz mono float32."""

    def __init__(self, model_dir: Path) -> None:
        try:
            import torch  # noqa: PLC0415 - lazy `ml` extra
            from transformers import (  # noqa: PLC0415 - lazy `ml` extra
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
        # Any: transformers' wrapped __call__ defeats the inferred type once
        # .cuda() reassigns the module, and mypy then rejects the forward call.
        self._model: Any = Wav2Vec2ForCTC.from_pretrained(str(model_dir))
        self._model.eval()
        if self._cuda:
            # mypy misreads transformers' wrapped .cuda as a __call__.
            self._model = self._model.cuda()  # type: ignore[call-arg]
        self.vocab_size = int(self._model.config.vocab_size or 0)

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
            logits = self._model(inputs.input_values, attention_mask=inputs.attention_mask).logits[
                0
            ]
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
