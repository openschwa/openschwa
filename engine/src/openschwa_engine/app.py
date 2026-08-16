"""Desktop entry point: serve the UI and open a browser at it.

The difference from `openschwa-engine` is presentation, not architecture — the
same engine, the same static SPA, one origin. This is the shape the architecture
calls for (docs/architecture.md §3, "Serving the UI"): Tauri later wraps exactly
this, and the hosted deployment runs exactly this without the browser-opening.

`OPENSCHWA_*` environment variables still override everything, so a user who
wants a different port or model directory sets it the same way here.
"""

import os

from openschwa_engine.server import main as serve


def main() -> None:
    # Defaults, not overrides: an explicitly-set variable still wins.
    os.environ.setdefault("OPENSCHWA_OPEN_BROWSER", "true")
    serve()


if __name__ == "__main__":
    main()
