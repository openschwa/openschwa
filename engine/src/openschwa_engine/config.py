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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENSCHWA_")

    host: str = "127.0.0.1"  # localhost only; never bind wider by default
    port: int = 8577  # fixed unusual default; server auto-increments on conflict
    model_dir: Path = Path(platformdirs.user_cache_dir("openschwa")) / "models"
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

    #: Alignment model id in models.registry.MANIFEST. Provisional for M0 — the
    #: M1 bake-off (docs/architecture.md §6) picks the shipping model; this setting is how a
    #: candidate gets swapped in without touching pipeline code.
    alignment_model: str = "wav2vec2-espeak-cv-ft"

    #: Load the acoustic model at startup rather than on the first recording.
    #: Disable in tests and tooling that never analyse audio.
    warm_model_on_start: bool = True

    #: Analysis refuses to judge below these. Placeholders until the M1 eval
    #: harness produces calibrated thresholds in scoring/calibration.yaml —
    #: they gate "did alignment work at all", never a pronunciation verdict.
    min_alignment_confidence: float = 0.30
    low_alignment_confidence: float = 0.55
