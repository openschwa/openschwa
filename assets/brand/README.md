# OpenSchwa brand assets

## The mark

**[ə]** — the project's namesake sound written the way a phonetician would:
the schwa between IPA square brackets. Amber brackets around a slate glyph,
on cream. The prompt that generated it lives next to this file in
[logo-prompts.md](logo-prompts.md); the toy and clay prompts there are for
marketing art only, never for the functional logo.

## Source of truth

`schwa-source.png` is the master (1254×1254, opaque cream). It is generated
art, and it is **never edited in place** — every asset below is derived from
it by `build-icons.py`, so the master is the only file anyone has to keep.

```bash
just brand
```

| Generated file | Use |
|---|---|
| `schwa.png` | Full mark, transparent, trimmed, capped at 420px. README and any large use |
| `ui/public/logo.png` | The mark at its natural aspect, transparent, for the app header |
| `ui/public/favicon.ico` | The mark on a cream tile, 16/32/48px |
| `ui/public/apple-touch-icon.png` | 180px, mark on a cream tile |

Do not hand-edit any of them. `Logo.svelte` is written by hand; everything else
is output.

Not gated in CI. The JSON Schema artifacts are a contract and drift in them
breaks clients; these are pictures, and gating them would make the build depend
on an image library.

## Three constraints the artwork imposes

Each of these was measured, and each is the reason a step in `build-icons.py`
exists.

**The background needs flood-filling, not thresholding.** The mark sits on
cream, but the glyph's counter *contains* the same cream. A global
cream-to-alpha pass punches a hole through the counter; filling inward from
the four corners stops at the outlines and leaves the enclosed cream intact.

**Small sizes need no crop.** The mark is three chunky shapes — two thick
brackets and one bold glyph — and all three survive the 16px favicon.

**The mark cannot sit on a dark surface unaided.** The glyph is `#3b3e45`.
Against the app's dark background `#0f172a` that is roughly **1.6:1** — far
below legible. Every standalone icon is therefore composited onto a cream tile
and brings its own contrast. In the app, `Logo.svelte` does the same job in
CSS, showing the chip only under `prefers-color-scheme: dark`, which is why
`logo.png` ships transparent.

## Palette

Taken from the artwork rather than imposed on it (the prompt asked for
`#3b3c40`/`#d97706`/`#faf7f0`; the render landed a hair off — the artwork is
the truth):

| Role | Value |
|---|---|
| Glyph | `#3b3e45` |
| Brackets | `#e36c02` |
| Background / tile | `#fcf8f0` |

The UI's own `--focus` and `--warn` tokens are amber and mean "the phone under
examination". The logo's amber is adjacent but never drawn from those tokens —
do not wire them together, or a palette change to the drill UI will silently
restyle the logo.

## Using it

- Keep clear space of at least a quarter of the mark's width on every side.
- Never place the untiled mark on a dark background. Use the tile, or
  `apple-touch-icon.png`, which already has one.
- 16px is the floor, and only for the tiled favicon.
- Do not recolour the mark, and do not stretch it — the aspect is landscape
  (roughly 3:2).
- There is no wordmark lockup. Live `<text>` in an SVG renders with whatever
  font the viewer has, so the README pairs the mark with an ordinary markdown
  heading instead.

## Known limitation: this is a raster

The mark is a PNG, so unlike the rest of the UI it cannot inherit
`currentColor`, and it does not scale indefinitely. Assets are generated at 2–8×
their display size, which covers hi-dpi screens, but a genuinely large use
(print, a big hero) would need the mark redrawn or traced to vector first.

Tracing was not attempted here: the reliable tools need a system libcairo or
potrace, which would turn a one-command setup into "install homebrew first".
