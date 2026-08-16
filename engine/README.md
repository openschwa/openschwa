# openschwa-engine

The OpenSchwa analysis engine: a local FastAPI service that scores scripted
pronunciation drills and returns judgments, annotations, and confidence-gated
feedback as a versioned `AnalysisResult` (see `../schemas/`).

```bash
uv sync --extra ml        # install, including torch + the acoustic model runtime
uv run pytest             # tests, including the schema-drift gate
uv run openschwa-engine   # serve on http://127.0.0.1:8577
```

`uv sync` without `--extra ml` gives a working engine that serves exercises and
measures audio, duration, and F0 — but cannot align phones, so every analysis
comes back with `alignment.status = "failed"` and a single `retry`. That is the
configuration CI runs, because the degraded path is the one that must never
break.

The acoustic model itself (~1.3 GB) is downloaded separately, into a
platformdirs cache outside the repo (`OPENSCHWA_MODEL_DIR` to relocate):

```bash
just models    # or use the app's first-run download panel
```

Every setting is overridable via `OPENSCHWA_*` environment variables — see
`config.py`.

Architecture and module contracts: [docs/architecture.md](../docs/architecture.md).
