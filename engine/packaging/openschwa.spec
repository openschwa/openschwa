# PyInstaller spec for the OpenSchwa desktop bundle.
#
# The packaging spike the roadmap scheduled for M2, pulled forward. Build with:
#   just package        (from the repo root — it builds the UI first)
#
# onedir, not onefile: onefile unpacks ~2 GB to a temp directory on every
# launch, which turns a 3-second start into a 30-second one. onedir also keeps
# torch's dylibs where the loader expects them.
#
# What is deliberately NOT bundled: the acoustic models. The default aligner
# is now the 0.38 GB charsiu CTC model (the 1.26 GB espeak model remains an
# option); both live in a platformdirs cache shared with the dev install, and
# are licensed separately. The app downloads the chosen one on first run.

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

SPEC_DIR = Path(SPECPATH)
REPO_ROOT = SPEC_DIR.parents[1]

# Resources keep their repo-relative layout inside the bundle, so
# config._resource_root() resolves identically frozen or not.
datas = [
    (str(REPO_ROOT / "content"), "content"),
    (str(REPO_ROOT / "ui" / "dist"), "ui/dist"),
]
# The committed vocabulary snapshots ship inside the package; so does
# scoring/calibration.yaml when one exists (a future contrast passing the bar).
datas += collect_data_files("openschwa_engine")

hiddenimports = []

# transformers resolves model classes through a lazy module registry, so static
# analysis finds almost none of it. Only the wav2vec2 family is needed, and
# collecting the whole library would add hundreds of MB of unrelated
# architectures.
hiddenimports += collect_submodules("transformers.models.wav2vec2")
hiddenimports += [
    "transformers.models.auto",
    "transformers.models.auto.configuration_auto",
    "transformers.models.auto.feature_extraction_auto",
    "transformers.models.auto.modeling_auto",
]

# uvicorn likewise picks its loop and protocol implementations by name at
# runtime. Without these the frozen server starts and then cannot serve.
hiddenimports += collect_submodules("uvicorn")

a = Analysis(
    [str(SPEC_DIR / "entry.py")],
    pathex=[str(REPO_ROOT / "engine" / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Pruned: pulled in transitively, never used at runtime, and expensive.
    #
    # Nothing under `torch.*` belongs here. torch's __init__ imports its own
    # subpackages eagerly, so excluding one — torch.distributed and
    # torch.testing were the tempting pair — breaks `import torch` outright,
    # and the failure surfaces at runtime as "alignment unavailable" rather
    # than as a build error.
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "notebook",
        "pytest",
        "mypy",
        "ruff",
        "torchvision",
        "torchaudio",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="openschwa",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX corrupts signed dylibs on macOS
    console=True,  # keeps the log visible; a windowed shell arrives with Tauri
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="openschwa",
)
