# OpenSchwa task runner (https://github.com/casey/just)

default:
    @just --list

# First-time setup: engine deps (including the ML extra) + UI deps.
# The `ml` extra is ~800 MB installed; without it the engine still runs and
# measures audio, but cannot align phones.
setup:
    cd engine && uv sync --extra ml
    cd ui && npm install

# Download the alignment model (~1.3 GB, resumable, cached outside the repo).
# The app can also do this from its first-run panel.
models:
    cd engine && uv run python -c "from openschwa_engine.config import Settings; from openschwa_engine.models.registry import ModelRegistry; s=Settings(); r=ModelRegistry(s.model_dir); spec=r.spec(s.alignment_model); [print(f\"{e.get('bytes_done',0)/1e6:.0f} MB\") for e in r.pull(spec) if e.get('done')]; print('ready')"

# Run engine + UI dev servers together (Ctrl-C stops both)
dev:
    (cd engine && uv run openschwa-engine) & (cd ui && npm run dev) & wait

# All tests and checks, every side (engine, ui, eval)
test:
    cd engine && uv run python -m pytest -q
    cd engine && uv run ruff check . && uv run ruff format --check . && uv run python -m mypy
    cd ui && npm run check
    cd ui && npm test
    cd eval && uv run python -m pytest -q
    cd eval && uv run ruff check . && uv run ruff format --check .
    cd training && uv run python -m pytest -q
    cd training && uv run ruff check src tests

# Regenerate the contract artifacts (JSON Schema + TS types) from the pydantic models
schema:
    cd engine && uv run python -m openschwa_engine.schemas.export
    cd ui && npm run gen:types

# Regenerate the logo assets from assets/brand/schwa-source.png.
# Not a CI gate — these are pictures, not a contract. See assets/brand/README.md.
brand:
    uv run --with pillow python assets/brand/build-icons.py

# Build the desktop bundle for THIS machine (unsigned, ~590 MB, macOS/Linux/Windows
# native — it is not cross-platform). Signed installers are M4.
package:
    cd ui && npm run build
    cd engine && uv sync --extra ml --group package
    cd engine && uv run pyinstaller --noconfirm --clean \
        --distpath ../dist --workpath ../dist/.build packaging/openschwa.spec
    @echo
    @echo "built dist/openschwa/openschwa — run it to launch the app"

# Offline evaluation harness — see eval/README.md. Model download: `just models`.
eval *ARGS:
    cd eval && uv run python run_eval.py {{ARGS}}
