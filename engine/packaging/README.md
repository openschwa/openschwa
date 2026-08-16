# Desktop packaging

The PyInstaller + torch spike the roadmap scheduled for M2, done early. Build a
bundle for the machine you are sitting at:

```bash
just package     # builds the UI, then freezes the engine around it
./dist/openschwa/openschwa
```

The app serves its own UI and opens a browser at it. One process, one origin —
the same shape Tauri will wrap in M4 and the same shape a hosted deployment
runs, minus the browser-opening.

## What this is not

**Not cross-platform.** PyInstaller freezes the interpreter and native
extensions of the machine that runs it. A macOS arm64 build runs on macOS arm64
and nowhere else; each target needs its own build host.

**Not signed or notarized.** macOS will refuse the first launch from Finder
until it is allowed through Gatekeeper, and Windows will show SmartScreen.
Signed installers are M4 — see the risk table in `docs/architecture.md`.

**Not self-contained.** The acoustic model (~1.3 GB) is *not* in the bundle. It
is downloaded on first run into a platformdirs cache shared with the dev
install, so a developer who already has it pays nothing. Bundling it would
triple the artifact and fold an Apache-2.0 model into an AGPL distribution
that has no need to carry it.

## Size

~590 MB, of which torch is ~400 MB and parselmouth ~30 MB. That is close to the
floor for this approach: excluding `tkinter`, `matplotlib`, and the unused
`torchvision`/`torchaudio` is worth a fair amount, but torch itself is not
negotiable while inference runs through it. **ONNX Runtime export is the
pre-planned escape hatch** if the size becomes a problem — it is why the
acoustic model sits behind a single wrapper module.

## Traps found building this

**Never exclude a `torch.*` submodule.** torch's `__init__` imports its own
subpackages eagerly, so pruning `torch.distributed` or `torch.testing` breaks
`import torch` entirely. The failure is quiet: the bundle builds, the app
starts, and alignment is simply unavailable at runtime.

**transformers needs explicit hidden imports.** It resolves model classes
through a lazy registry that static analysis cannot follow. Only the wav2vec2
family is collected — `collect_submodules("transformers")` would add hundreds
of megabytes of unrelated architectures.

**onedir, not onefile.** onefile unpacks the whole bundle to a temp directory
on every launch, turning a fast start into a very slow one.
