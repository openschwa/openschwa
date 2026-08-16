"""FastAPI app factory + console entrypoint.

App-factory pattern so tests, the Tauri sidecar, and hosted deployments can
each configure the app differently. The bound port is printed to stdout at
startup — Tauri discovers the engine that way (with /v1/health as the poll
target).

Content packs are loaded and validated during `create_app`: a malformed pack
should stop the engine coming up, not surface as a broken drill later.
"""

import logging
import socket
import threading
import time
import webbrowser

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from openschwa_engine import __version__
from openschwa_engine.api import analyze, exercises, health, models
from openschwa_engine.config import Settings
from openschwa_engine.content import load_library
from openschwa_engine.models.registry import ModelError, ModelRegistry, ml_runtime_available

log = logging.getLogger(__name__)

#: How many ports to try past the configured one before giving up.
PORT_SCAN_RANGE = 20


def _warm_model(registry: ModelRegistry, model_id: str) -> None:
    """Load the acoustic model in the background, if it is already downloaded.

    Reading 1.2 GB of weights takes tens of seconds, and lazily paying that on
    the first recording is the worst possible moment — the learner has just
    spoken and is waiting. Startup is unaffected: a request arriving mid-warmup
    simply waits on the same lock instead of loading a second copy.
    """
    from openschwa_engine.alignment import acoustic  # noqa: PLC0415 - lazy `ml` extra

    try:
        spec = registry.spec(model_id)
        acoustic.load(registry.require_ready(spec))
        log.info("acoustic model %s ready", model_id)
    except ModelError as exc:
        # Expected when the engine is running without weights or without the
        # `ml` extra — /v1/health already reports it, so one line will do.
        log.info("alignment disabled: %s", exc)
    except Exception:  # a warmup failure must never stop the engine serving
        log.warning("could not pre-load %s; it will load on first use", model_id, exc_info=True)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="OpenSchwa engine", version=__version__)
    app.state.settings = settings
    app.state.library = load_library(settings.content_dir, settings.content_schema_path)
    app.state.registry = ModelRegistry(settings.model_dir)

    if (
        settings.warm_model_on_start
        and ml_runtime_available()
        and app.state.registry.is_ready(app.state.registry.spec(settings.alignment_model))
    ):
        threading.Thread(
            target=_warm_model,
            args=(app.state.registry, settings.alignment_model),
            name="warm-model",
            daemon=True,
        ).start()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in (health.router, exercises.router, analyze.router, models.router):
        app.include_router(router, prefix="/v1")

    # Mounted last, so the /v1 routes above always win the match. Absent in a
    # plain checkout until `npm run build`, which is the normal dev case — the
    # Vite server owns the UI then.
    if settings.ui_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.ui_dir, html=True), name="ui")
        log.info("serving UI from %s", settings.ui_dir)

    return app


def find_free_port(host: str, preferred: int, attempts: int = PORT_SCAN_RANGE) -> int:
    """First free port at or after `preferred`.

    The engine binds loopback only, but a second copy (or an unrelated service)
    on the default port would otherwise be a confusing hard failure. Whatever is
    chosen is printed to stdout for the desktop shell to pick up.
    """
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise SystemExit(f"no free port in {preferred}-{preferred + attempts - 1} — set OPENSCHWA_PORT")


def _open_browser_when_listening(url: str, host: str, port: int, timeout_s: float = 30.0) -> None:
    """Wait for the socket to accept, then open a browser.

    Opening immediately races the server and lands the user on a connection
    error; polling the port is the difference between "the app opened" and "the
    app looked broken on first launch".
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            if probe.connect_ex((host, port)) == 0:
                webbrowser.open(url)
                return
        time.sleep(0.1)
    log.warning("server did not start within %.0fs; open %s yourself", timeout_s, url)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings()
    port = find_free_port(settings.host, settings.port)
    app = create_app(settings)
    exercise_count = len(app.state.library.exercises)
    url = f"http://{settings.host}:{port}"
    print(f"openschwa-engine listening on {url}")
    print(f"loaded {exercise_count} exercises from {settings.content_dir}")

    if settings.open_browser:
        threading.Thread(
            target=_open_browser_when_listening,
            args=(url, settings.host, port),
            name="open-browser",
            daemon=True,
        ).start()

    # Single worker: the acoustic model is a per-process singleton and a second
    # copy would double a 1.2 GB resident footprint for no throughput gain.
    uvicorn.run(app, host=settings.host, port=port, workers=1)


if __name__ == "__main__":
    main()
