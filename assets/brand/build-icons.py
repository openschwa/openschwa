"""Derive every logo asset from one master image: the [ə] mark.

Run via `just brand`. The master is `schwa-source.png`, generated art
that is never edited in place — everything below is regenerated from it, so the
master stays the only thing anyone has to keep.

Three findings drove the shape of this script, each measured rather than
assumed:

1. **The background must be flood-filled, not thresholded.** The mark sits on
   cream, but the glyph's counter *contains* cream too. A global cream-to-alpha
   pass punches a hole through the counter; filling inward from the four
   corners stops at the outlines and leaves the enclosed cream alone.

2. **No crop is needed at small sizes.** The mark is three chunky shapes — two
   thick amber brackets around one bold slate glyph — so the full mark
   survives a 16px favicon; the bird's head-crop machinery is gone.

3. **The mark cannot sit directly on a dark page.** The slate glyph is about
   1.6:1 against the app's #0f172a. Every standalone icon is therefore
   composited onto a cream tile and brings its own contrast. In the app the
   same job is done in CSS by Logo.svelte, which is why the header asset ships
   transparent.

Not wired into CI: unlike the JSON Schema artifacts these are not a contract,
and gating them would make the build depend on an image library.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

CREAM = (252, 248, 240, 255)  # the master's background, reused as the tile
TILE_RADIUS = 0.22  # corner radius as a fraction of the tile

HERE = Path(__file__).parent
REPO = HERE.parent.parent
PUBLIC = REPO / "ui" / "public"
SOURCE = HERE / "schwa-source.png"


def cut_out() -> Image.Image:
    """The master with its background removed and trimmed to the artwork."""
    im = Image.open(SOURCE).convert("RGBA")
    flood = im.copy()
    corners = [(0, 0), (im.width - 1, 0), (0, im.height - 1), (im.width - 1, im.height - 1)]
    for corner in corners:
        ImageDraw.floodfill(flood, corner, (0, 0, 0, 0), thresh=40)
    im.putalpha(flood.split()[3])
    return im.crop(im.getbbox())


def _fit(im: Image.Image, box: int, pad: float) -> Image.Image:
    scale = (box * (1 - pad)) / max(im.size)
    return im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)


def on_tile(im: Image.Image, box: int, pad: float = 0.16) -> Image.Image:
    """Composite onto a rounded cream tile so the mark carries its own contrast."""
    tile = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    mask = Image.new("L", (box * 4, box * 4), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, box * 4 - 1, box * 4 - 1], radius=int(box * 4 * TILE_RADIUS), fill=255
    )
    tile.paste(Image.new("RGBA", (box, box), CREAM), (0, 0), mask.resize((box, box), Image.LANCZOS))
    art = _fit(im, box, pad)
    tile.paste(art, ((box - art.width) // 2, (box - art.height) // 2), art)
    return tile


def padded(im: Image.Image, height: int, pad: float = 0.12) -> Image.Image:
    """Scale to `height` keeping the artwork's own aspect, with a margin.

    Deliberately not squared off. The mark is landscape (roughly 3:2), and
    fitting it into a square canvas wastes ~40% of the height as empty space —
    at header sizes that shrinks the mark to the point of illegibility.
    """
    art = im.resize(
        (max(1, round(im.width * height / im.height)), height), Image.LANCZOS
    )
    margin = round(height * pad)
    canvas = Image.new("RGBA", (art.width + margin * 2, art.height + margin * 2), (0, 0, 0, 0))
    canvas.paste(art, (margin, margin), art)
    return canvas


def cap(im: Image.Image, height: int) -> Image.Image:
    """Scale down to `height`, preserving aspect.

    Right-sizing is the only compression applied here. The master's "flat"
    fills carry a little noise; palette-quantising would amplify it into
    visible mottling across the glyph, so the master is simply shipped in
    fewer pixels.
    """
    if im.height <= height:
        return im
    return im.resize((round(im.width * height / im.height), height), Image.LANCZOS)


def main() -> None:
    full = cut_out()
    written = []

    def save(im: Image.Image, path: Path, **kw: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        im.save(path, optimize=True, **kw)
        written.append(path)

    # The full mark, for the README and any large use. Capped at ~3x the
    # largest size anything in the repo actually displays it at.
    save(cap(full, 420), HERE / "schwa.png")

    # The header mark ships transparent; Logo.svelte draws the tile in CSS so
    # one asset serves both colour schemes.
    save(padded(cap(full, 320), 160), PUBLIC / "logo.png")

    # Favicons are the mark on a tile: the slate glyph does not survive dark
    # browser chrome unaided, and the tile carries the contrast.
    save(on_tile(full, 256), PUBLIC / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    save(on_tile(full, 180, pad=0.12).convert("RGB"), PUBLIC / "apple-touch-icon.png")

    for path in written:
        print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
