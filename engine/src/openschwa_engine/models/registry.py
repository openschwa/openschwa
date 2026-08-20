"""Model manifest, cache state, and resumable downloads.

Weights are pinned by commit sha, never by branch: an upstream retrain must not
silently change what a learner is scored against. Downloads land in
`settings.model_dir` (override with `OPENSCHWA_MODEL_DIR`) so the Tauri sidecar
and hosted deployments can point elsewhere.

The heavy imports (`torch`, `transformers`, `huggingface_hub`) live in the
engine's `ml` extra and are imported lazily inside methods. Everything the API
needs to *describe* model state — is it downloaded, how big is it — works
without them, so an engine with no ML stack still starts, still serves
exercises, and still returns a schema-valid "retry" instead of crashing.
"""

import json
import logging
import queue
import shutil
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from types import MappingProxyType
from typing import Any

from openschwa_engine.models.phone_set import PhoneMap

log = logging.getLogger(__name__)

VOCAB_DIR = Path(__file__).parent / "vocab"

#: Files pulled from the hub for a model to count as downloaded. Weights
#: specifically accept both serializations (see WEIGHT_FILES): transformers 5.x
#: saves model.safetensors by default, and the Option 3 fine-tune uses it.
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin", "vocab.json")

#: Either weight format satisfies is_ready(); the rest of REQUIRED_FILES must
#: all be present.
WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors")


class ModelError(RuntimeError):
    """Model cannot be used. Callers turn this into a "retry", never a verdict."""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    repo_id: str
    revision: str
    phone_table: str
    vocab_snapshot: str
    download_bytes: int
    license: str
    note: str
    #: Files the HF repo does not ship but the runtime needs, as
    #: (local filename, committed source under models/vocab/). Copied after pull.
    extra_files: tuple[tuple[str, str], ...] = ()
    #: "aligner" models carry the full phone inventory and align exercises;
    #: "contrast" models are closed-set judges (Option 3) and only ever score
    #: the focus interval - the pipeline never aligns with them.
    role: str = "aligner"


MANIFEST: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "wav2vec2-espeak-cv-ft": ModelSpec(
            id="wav2vec2-espeak-cv-ft",
            repo_id="facebook/wav2vec2-lv-60-espeak-cv-ft",
            # Pinned commit, not `main` — see module docstring.
            revision="ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4",
            phone_table="espeak_en",
            vocab_snapshot="wav2vec2-espeak-cv-ft.json",
            download_bytes=1_263_540_000,
            license="Apache-2.0",
            note=(
                "M0 aligner, kept as a bake-off candidate: multilingual CTC over "
                "eSpeak IPA phonemes. The M1 bake-off (eval/reports/) found neither "
                "candidate discriminative for the /ð/ contrast; charsiu won the "
                "alignment-sanity / size / latency criteria and became the default."
            ),
        ),
        "charsiu-en-w2v2-ctc": ModelSpec(
            id="charsiu-en-w2v2-ctc",
            repo_id="charsiu/en_w2v2_ctc_libris_and_cv",
            # Pinned commit, not 'main' - see module docstring.
            revision="70f5061463f2927a27236d7e9d309cf0fd5282b3",
            phone_table="charsiu_en",
            vocab_snapshot="charsiu-en-w2v2-ctc.json",
            download_bytes=377_706_220,
            license="unknown (charsiu)",
            note=(
                "M1 bake-off candidate: wav2vec2-base CTC fine-tuned on LibriSpeech "
                "and Common Voice, stressless ARPABET phone vocabulary."
            ),
            # The HF repo ships only config + weights; the preprocessor config
            # and the vocabulary come from committed snapshots.
            extra_files=(
                ("preprocessor_config.json", "charsiu-en-w2v2-ctc-preprocessor.json"),
                ("vocab.json", "charsiu-en-w2v2-ctc.json"),
            ),
        ),
        # The Option 3 contrast judge: fine-tuned locally (training/) from the
        # charsiu base. Its weights never live on a hub - the training run's
        # out/model/ directory is dropped into the model dir wholesale, and
        # is_ready() validates the layout. pull() refuses with instructions.
        "dh-contrast-v1": ModelSpec(
            id="dh-contrast-v1",
            repo_id="local",
            revision="local",
            phone_table="dhz_en",
            vocab_snapshot="dh-contrast-v1.json",
            download_bytes=0,
            license="Apache-2.0 (fine-tuned locally from the charsiu base)",
            note=(
                "Option 3 closed-set judge for /ð/ vs {z, d, v}: a charsiu-base "
                "wav2vec2 with a fresh 4-class CTC head, fine-tuned on the "
                "L2-ARCTIC train split (training/). Vocabulary is exactly "
                "{blank, ð, z, d, v} - it cannot align, only judge."
            ),
            role="contrast",
        ),
        "dh-contrast-v2": ModelSpec(
            id="dh-contrast-v2",
            repo_id="local",
            revision="local",
            phone_table="dhz_en",
            vocab_snapshot="dh-contrast-v2.json",
            download_bytes=0,
            license="Apache-2.0 (fine-tuned locally from the charsiu base)",
            note=(
                "Option 1 closed-set sequence classifier for /ð/ vs {z, d, v}: "
                "a charsiu-base wav2vec2 with a 4-class pooled classification head, "
                "fine-tuned with Cross-Entropy loss on L2-ARCTIC + speechocean762 train splits."
            ),
            role="contrast",
        ),
        "dh-contrast-v3": ModelSpec(
            id="dh-contrast-v3",
            repo_id="local",
            revision="local",
            phone_table="dhz_en",
            vocab_snapshot="dh-contrast-v3.json",
            download_bytes=0,
            license="Apache-2.0 (fine-tuned locally from the charsiu base)",
            note=(
                "Wav2Vec2 + DSP Acoustic Feature Fusion Classifier for /ð/ vs {z, d, v}: "
                "incorporating high-frequency sibilance ratio, stop closure/burst cues, "
                "and 100ms expanded context window."
            ),
            role="contrast",
        ),
        "dh-contrast-v4": ModelSpec(
            id="dh-contrast-v4",
            repo_id="local",
            revision="local",
            phone_table="dhz_en",
            vocab_snapshot="dh-contrast-v4.json",
            download_bytes=0,
            license="Apache-2.0 (fine-tuned locally from the charsiu base)",
            note=(
                "Refined 10-dim DSP Acoustic Feature Fusion Classifier for /ð/ vs {z, d, v}: "
                "with burst contrast ratio, sibilance prominence, and voicing continuity."
            ),
            role="contrast",
        ),
        "dh-contrast-v5": ModelSpec(
            id="dh-contrast-v5",
            repo_id="local",
            revision="local",
            phone_table="dhz_en",
            vocab_snapshot="dh-contrast-v5.json",
            download_bytes=0,
            license="Apache-2.0 (fine-tuned locally from the charsiu base)",
            note=(
                "Target-Boosted 10-dim DSP Acoustic Feature Fusion Classifier for "
                "/ð/ vs {z, d, v}: calibrated to meet the >=90% precision shipping bar."
            ),
            role="contrast",
        ),
        "dh-contrast-v6": ModelSpec(
            id="dh-contrast-v6",
            repo_id="local",
            revision="local",
            phone_table="dhz_en",
            vocab_snapshot="dh-contrast-v6.json",
            download_bytes=0,
            license="Apache-2.0 (fine-tuned locally from the charsiu base)",
            note=("Optimized Precision-Gated 10-dim DSP Fusion Classifier for /ð/ vs {z, d, v}."),
            role="contrast",
        ),
        "dh-contrast-v7": ModelSpec(
            id="dh-contrast-v7",
            repo_id="local",
            revision="local",
            phone_table="dhz_en",
            vocab_snapshot="dh-contrast-v7.json",
            download_bytes=0,
            license="Apache-2.0 (fine-tuned locally from the charsiu base)",
            note=(
                "12-dim Core/Boundary DSP Feature Fusion Classifier for /ð/ vs {z, d, v}: "
                "isolates coarticulation bleed and eliminates continuous speech false alarms."
            ),
            role="contrast",
        ),
        "dh-contrast-v8": ModelSpec(
            id="dh-contrast-v8",
            repo_id="local",
            revision="local",
            phone_table="dhz_en",
            vocab_snapshot="dh-contrast-v8.json",
            download_bytes=0,
            license="Apache-2.0 (fine-tuned locally from the charsiu base)",
            note=("Balanced 10-dim DSP Acoustic Feature Fusion Classifier for /ð/ vs {z, d, v}."),
            role="contrast",
        ),
        "dh-contrast-v9": ModelSpec(
            id="dh-contrast-v9",
            repo_id="local",
            revision="local",
            phone_table="dhz_en",
            vocab_snapshot="dh-contrast-v9.json",
            download_bytes=0,
            license="Apache-2.0 (fine-tuned locally from the charsiu base)",
            note=(
                "Dual Mean+Max Temporal Pooling 10-dim DSP Fusion Classifier for /ð/ vs {z, d, v}: "
                "balanced loss weighting for robust human practice discrimination."
            ),
            role="contrast",
        ),
        "dh-contrast-v12": ModelSpec(
            id="dh-contrast-v12",
            repo_id="local",
            revision="local",
            phone_table="dhz_en",
            vocab_snapshot="dh-contrast-v12.json",
            download_bytes=0,
            license="Apache-2.0 (fine-tuned locally from the charsiu base)",
            note=(
                "Stage 3 r3: the v22 closed-set {ð, z, d, v} data recipe on the "
                "honest three-way split with the fixed training recipe."
            ),
            role="contrast",
        ),
        "dh-contrast-v11": ModelSpec(
            id="dh-contrast-v11",
            repo_id="local",
            revision="local",
            phone_table="dhz_open_en",
            vocab_snapshot="dh-contrast-v11.json",
            download_bytes=0,
            license="Apache-2.0 (fine-tuned locally from the charsiu base)",
            note=(
                "Stage 3 open-set judge: {ð, z, d, other} with the fixed recipe "
                "(single optimizer with warmup+cosine, forever-frozen conv encoder, "
                "exam-shaped selection, per-run provenance)."
            ),
            role="contrast",
        ),
        "dh-contrast-v10": ModelSpec(
            id="dh-contrast-v10",
            repo_id="local",
            revision="local",
            phone_table="dhz_en",
            vocab_snapshot="dh-contrast-v10.json",
            download_bytes=0,
            license="Apache-2.0 (fine-tuned locally from the charsiu base)",
            note=(
                "Hard-negative trained 10-dim DSP Fusion Classifier for /ð/ vs {z, d, v}: "
                "expert-error examples + mined hard negatives + no label smoothing."
            ),
            role="contrast",
        ),
        "tinyschwa-v1": ModelSpec(
            id="tinyschwa-v1",
            repo_id="local",
            revision="local",
            phone_table="dhz_en",
            vocab_snapshot="tinyschwa-v1.json",
            download_bytes=0,
            license="Apache-2.0 (fine-tuned locally from facebook/wav2vec2-xls-r-300m)",
            note=(
                "TinySchwa v1: 300M Cross-Lingual XLS-R Foundation Model with Dual Mean+Max "
                "Temporal Pooling and 10-dim DSP Acoustic Feature Fusion for /ð/ vs {z, d, v}."
            ),
            role="contrast",
        ),
        "ear-xlsr-v1": ModelSpec(
            id="ear-xlsr-v1",
            repo_id="local",
            revision="local",
            phone_table="charsiu_en",
            vocab_snapshot="ear-xlsr-v1.json",
            download_bytes=0,
            license="Apache-2.0 (XLS-R-300M base; head trained on CC-BY/CC0 transcript speech)",
            note=(
                "The ear (Phase 1 mirror): frozen XLS-R-300M + a CTC phone head over the "
                "charsiu stressless-ARPABET inventory, trained on LibriSpeech train.100 "
                "(CC-BY-4.0) transcripts. A full-vocabulary hearing model - it scores the "
                "focus segment only (role=contrast); the charsiu aligner keeps aligning."
            ),
            role="contrast",
        ),
    }
)


@dataclass(frozen=True)
class ModelStatus:
    spec: ModelSpec
    state: str  # ready | missing | downloading
    runtime_available: bool
    path: Path | None


def ml_runtime_available() -> bool:
    """True when the `ml` extra is installed (`uv sync --extra ml`)."""
    return all(find_spec(mod) is not None for mod in ("torch", "transformers"))


class ModelRegistry:
    """Owns model files on disk. Does not own the loaded network — that is
    `alignment.acoustic`, which caches it behind this registry's paths."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._downloading: set[str] = set()
        self._lock = threading.Lock()

    def spec(self, model_id: str) -> ModelSpec:
        try:
            return MANIFEST[model_id]
        except KeyError as exc:
            raise ModelError(f"unknown model '{model_id}'") from exc

    def local_dir(self, spec: ModelSpec) -> Path:
        return self.model_dir / spec.id

    def is_ready(self, spec: ModelSpec) -> bool:
        base = self.local_dir(spec)
        fixed = [name for name in REQUIRED_FILES if name not in WEIGHT_FILES]
        return all((base / name).is_file() for name in fixed) and any(
            (base / name).is_file() for name in WEIGHT_FILES
        )

    def status(self, spec: ModelSpec) -> ModelStatus:
        with self._lock:
            downloading = spec.id in self._downloading
        ready = self.is_ready(spec)
        return ModelStatus(
            spec=spec,
            state="downloading" if downloading else ("ready" if ready else "missing"),
            runtime_available=ml_runtime_available(),
            path=self.local_dir(spec) if ready else None,
        )

    def catalog(self) -> list[ModelStatus]:
        return [self.status(spec) for spec in MANIFEST.values()]

    def require_ready(self, spec: ModelSpec) -> Path:
        if not self.is_ready(spec):
            raise ModelError(
                f"model '{spec.id}' is not downloaded — POST /v1/models/pull "
                f"({spec.download_bytes / 1e9:.1f} GB)"
            )
        if not ml_runtime_available():
            raise ModelError(
                "the engine's `ml` extra is not installed — run `uv sync --extra ml` in engine/"
            )
        return self.local_dir(spec)

    # -- vocabulary + phone mapping -------------------------------------------------

    def _snapshot_vocab(self, spec: ModelSpec) -> dict[str, int]:
        path = VOCAB_DIR / spec.vocab_snapshot
        if not path.is_file():
            raise ModelError(f"missing committed vocabulary snapshot {path}")
        vocab: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))
        return vocab

    def vocab(self, spec: ModelSpec) -> dict[str, int]:
        """Downloaded vocabulary if present, else the committed snapshot.

        A downloaded vocabulary that disagrees with the snapshot means the pin
        moved or the cache is corrupt; either way every phone index could now be
        wrong, so it is a hard error rather than a warning.
        """
        snapshot = self._snapshot_vocab(spec)
        downloaded_path = self.local_dir(spec) / "vocab.json"
        if not downloaded_path.is_file():
            return snapshot
        downloaded: dict[str, int] = json.loads(downloaded_path.read_text(encoding="utf-8"))
        if downloaded != snapshot:
            raise ModelError(
                f"vocabulary for '{spec.id}' does not match the committed snapshot "
                f"({VOCAB_DIR / spec.vocab_snapshot}); phone indices cannot be trusted"
            )
        return downloaded

    def phone_map(self, spec: ModelSpec) -> PhoneMap:
        return PhoneMap.build(spec.id, spec.phone_table, self.vocab(spec))

    # -- download -------------------------------------------------------------------

    def pull(self, spec: ModelSpec) -> Iterator[dict[str, Any]]:
        """Download the model, yielding progress events for NDJSON streaming.

        Resumable: `huggingface_hub` keeps partial files and continues from
        wherever an interrupted run stopped.
        """
        if spec.repo_id == "local":
            raise ModelError(
                f"model '{spec.id}' is built locally (training/) - copy the training "
                f"run's out/model/ directory to {self.local_dir(spec)} and restart"
            )
        if find_spec("huggingface_hub") is None:
            raise ModelError(
                "huggingface_hub is not installed — run `uv sync --extra ml` in engine/"
            )
        with self._lock:
            if spec.id in self._downloading:
                raise ModelError(f"model '{spec.id}' is already downloading")
            self._downloading.add(spec.id)
        try:
            yield from self._pull(spec)
        finally:
            with self._lock:
                self._downloading.discard(spec.id)

    def _pull(self, spec: ModelSpec) -> Iterator[dict[str, Any]]:
        from huggingface_hub import snapshot_download  # noqa: PLC0415 - lazy `ml` extra
        from tqdm.auto import tqdm as base_tqdm  # noqa: PLC0415 - lazy `ml` extra

        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        target = self.local_dir(spec)

        class _ReportingTqdm(base_tqdm):  # type: ignore[misc]
            """huggingface_hub drives download progress through tqdm; hooking it
            is how we get byte-level updates without reimplementing caching."""

            def update(self, n: float | None = 1) -> bool | None:
                result = super().update(n)
                if self.total:
                    events.put(
                        {
                            "model_id": spec.id,
                            "bytes_done": int(self.n),
                            "bytes_total": int(self.total),
                        }
                    )
                return result

        error: list[BaseException] = []

        def run() -> None:
            try:
                snapshot_download(
                    repo_id=spec.repo_id,
                    revision=spec.revision,
                    local_dir=str(target),
                    allow_patterns=list(REQUIRED_FILES),
                    tqdm_class=_ReportingTqdm,
                )
            except BaseException as exc:  # surfaced to the client below
                error.append(exc)
            finally:
                events.put(None)

        worker = threading.Thread(target=run, name=f"pull-{spec.id}", daemon=True)
        worker.start()

        yield {"model_id": spec.id, "bytes_done": 0, "bytes_total": spec.download_bytes}
        while True:
            event = events.get()
            if event is None:
                break
            yield event
        worker.join()

        if error:
            log.exception("download of %s failed", spec.id, exc_info=error[0])
            yield {"model_id": spec.id, "error": str(error[0])}
            return

        # Files the upstream repo does not ship come from committed snapshots.
        for local_name, source_name in spec.extra_files:
            source = VOCAB_DIR / source_name
            if not source.is_file():
                raise ModelError(f"{spec.id}: missing committed extra file {source}")
            shutil.copyfile(source, target / local_name)

        # Reading the vocabulary validates it against the committed snapshot.
        self.vocab(spec)
        yield {"model_id": spec.id, "state": "ready", "done": True}
