"""Engine settings. Every value can be overridden via OPENSCHWA_* env vars —
required for the Tauri sidecar (custom model dir) and hosted deployments."""

import sys
from pathlib import Path

import platformdirs
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resource_root() -> Path:
    """Directory holding bundled resources: content packs and the built UI.

    In a checkout that is the repo root. In a PyInstaller bundle the same
    subtree is shipped under `sys._MEIPASS`, so keeping the layout identical
    means nothing below this line has to know whether it is frozen.
    """
    bundle = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and bundle:
        return Path(bundle)
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _resource_root()


def _default_model_dir() -> Path:
    repo_models = REPO_ROOT / ".models"
    if repo_models.is_dir():
        return repo_models
    return Path(platformdirs.user_cache_dir("openschwa")) / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENSCHWA_")

    host: str = "127.0.0.1"  # localhost only; never bind wider by default
    port: int = 8577  # fixed unusual default; server auto-increments on conflict
    model_dir: Path = _default_model_dir()
    content_dir: Path = REPO_ROOT / "content" / "packs"
    content_schema_path: Path = REPO_ROOT / "content" / "schema" / "exercise.schema.json"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    #: Built SPA. When present the engine serves it at `/`, which is what makes
    #: the packaged app and the hosted deployment a single origin — and why the
    #: UI addresses the API relatively rather than at a fixed port.
    ui_dir: Path = REPO_ROOT / "ui" / "dist"

    #: Open a browser once the server is listening. The desktop entry point
    #: turns this on; the bare engine leaves it off.
    open_browser: bool = False

    #: Alignment model id in models.registry.MANIFEST. The M1 bake-off
    #: (eval/reports/m1-bakeoff-*.md) found neither candidate able to carry the
    #: /ð/ contrast, but for the alignment job itself charsiu won every
    #: measurable criterion: higher alignment confidence on L2 speech (0.90 vs
    #: 0.82), a 3.3x smaller download, and 2.6x lower latency. This setting is
    #: still how a candidate gets swapped without touching pipeline code.
    alignment_model: str = "charsiu-en-w2v2-ctc"

    #: VAD backend behind audio.preprocess.detect_speech: "auto" prefers
    #: silero-vad when the ml extra can load it and falls back to the energy
    #: detector; "silero"/"energy" force one path.
    vad_backend: str = "auto"

    #: Dedicated contrast model (Option 3): a manifest id whose vocabulary is
    #: exactly the drilled closed set. When set and downloaded, the focus
    #: interval is scored by it rather than by the aligner's posteriors; when
    #: unset or unavailable, the engine falls back to aligner-based contrast
    #: scoring. Null means "use the alignment model" (the M0/M1 behavior).
    contrast_model_id: str | None = "tinyschwa-v1"

    #: Context padding on each side of the aligned focus phone when the
    #: contrast judge scores it. The judge was trained with 0.10; a window
    #: experiment (coarticulation false positives) can override it via
    #: OPENSCHWA_FOCUS_PAD_S without a retrain.
    focus_pad_s: float = 0.10

    #: Load the acoustic model at startup rather than on the first recording.
    #: Disable in tests and tooling that never analyse audio.
    warm_model_on_start: bool = True

    #: Analysis refuses to judge below these. Placeholders until the M1 eval
    #: harness produces calibrated thresholds in scoring/calibration.yaml —
    #: they gate "did alignment work at all", never a pronunciation verdict.
    min_alignment_confidence: float = 0.30
    low_alignment_confidence: float = 0.55
