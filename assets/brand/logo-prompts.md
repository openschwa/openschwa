# OpenSchwa [ə] logo — gpt-image prompts

Prompts for the primary mark: the schwa as a phonetic transcription
`[ə]` — one mark for the organisation and the programme. Generated with
gpt-image (gpt-image-1). The chosen mark landed as `schwa-source.png`
(see `README.md`); these prompts stay as the record of how it was made.

## Palette (unchanged from the current brand)

| Role | Value |
|---|---|
| Glyph | `#3B3C40` |
| Brackets / accent | `#D97706` |
| Light surface | `#FAF7F0` |
| Dark surface | `#0F172A` |

Do not wire the mark's amber to the UI `--focus`/`--warn` tokens.

## Prompt 1 — primary mark

> Minimalist flat vector logo of the IPA phonetic transcription of the schwa
> sound: the characters "[ə]" — a bold, geometric lowercase schwa glyph "ə"
> (shaped exactly like a lowercase letter "e" rotated 180 degrees, with its
> open counter facing left) centered between two thin, clean square brackets.
> The glyph is dark slate gray (#3B3C40), the square brackets are warm amber
> (#D97706). Solid flat fills only: no gradients, no outlines, no 3D, no drop
> shadows, no texture, no photographic elements, and no other text or
> decoration anywhere. The glyph must be a true turned "e" — not a normal
> "e", not an "a", not a question mark, not a squiggle. Centered square
> composition on a solid cream (#FAF7F0) background, with generous empty
> margin around the mark so it stays crisp and legible when shrunk to a
> 16-pixel favicon. Style: modern minimal education-technology brand, crisp
> typographic geometry, perfectly balanced letterform.

## Prompt 2 — app icon / dark tile

> The same mark, as an app icon: the phonetic transcription "[ə]" — a bold
> geometric lowercase schwa glyph "ə" (a lowercase "e" rotated 180 degrees,
> open counter facing left) between two thin square brackets — in cream
> (#FAF7F0) with the brackets in warm amber (#D97706), centered on a solid
> dark slate (#0F172A) rounded-square tile that fills the frame. Flat 2D,
> solid fills, no gradients, no 3D, no shadows, no texture, no other text.
> Square 1:1 composition, generous margin inside the tile so the mark reads
> at small sizes. Modern minimal education-technology app icon.

## Prompt 3 — wordmark lockup

> A horizontal wordmark "OpenSchwa" in a bold geometric sans-serif, all
> lowercase, dark slate gray (#3B3C40), where the letter "e" in "Open" is
> replaced by the schwa glyph "ə" in warm amber (#D97706), so it reads
> "OpənSchwa". Flat vector text, crisp typographic geometry, no gradients, no
> 3D, no shadows, no other decoration, on a solid cream (#FAF7F0)
> background. The schwa glyph must be a true turned "e" (open counter facing
> left), not a normal "e".

## Practical gpt-image notes

- **API (gpt-image-1):** `size: 1024x1024`, `quality: high`,
  `output_format: png`, and `background: transparent` for a real alpha
  channel (transparency is supported for PNG/WebP). Solid cream is fine
  otherwise — `build-icons.py` already strips backgrounds by flood fill.
- **In ChatGPT:** no parameter controls; keep "solid cream background" in
  the prompt (the model sometimes fakes transparency with a checkerboard
  pattern). Generate several rolls and keep only the ones where `ə` is a
  genuine turned "e" — the glyph is the logo; reroll the rest.
- Prompt 3's text may come out misspelled; iterate, or set the wordmark in a
  vector tool instead of gpt-image.

## Assumptions

- The square brackets are part of the mark (IPA notation is the point); a
  bracketless `ə` is a one-line edit away.
- One mark serves both the organisation and the programme; the wordmark
  lockup is optional garnish.
- The palette stays on the current brand until the user says otherwise.

## Playful variants — toy & clay

Marketing/hero art, not favicon material: at 16px the plush/clay texture
turns to mush, so Prompt 1 remains the functional logo. Both variants have
soft shadows and therefore live on a solid background (no clean
transparency).

### Prompt T — toy / plush variant

> Toy version of the OpenSchwa logo: the phonetic transcription "[ə]" as a
> cuddly stuffed toy — a chunky, rounded, plush schwa glyph "ə" (shaped
> exactly like a lowercase letter "e" rotated 180 degrees, with its open
> counter facing left) sewn from soft dark slate-gray fabric, flanked by two
> matching amber plush square brackets, all three pieces softly stuffed with
> slightly squashy rounded forms and visible stitching seams. The glyph must
> be a true turned "e" — not a normal "e", not an "a", not a question mark.
> Soft matte studio lighting, one gentle contact shadow, centered square
> composition on a plain cream (#FAF7F0) background with generous margins.
> Style: cozy children's-education brand, like a plush toy photographed for a
> nursery catalogue — warm and tactile but clean and uncluttered. No other
> text, no other objects.

Mascot add-on (swap one sentence in): "Add two tiny, friendly embroidered
eyes and a small smile to the schwa glyph only."

### Prompt C — clay-like variant

> Claymation-style version of the OpenSchwa logo: the IPA transcription
> "[ə]" hand-sculpted from soft plasticine clay — a chunky, rounded lowercase
> schwa glyph "ə" (shaped exactly like a lowercase letter "e" rotated 180
> degrees, open counter facing left) in matte slate-gray clay, between two
> smooth amber clay square brackets, hand-modeled with soft edges, gentle
> fingerprints and slight imperfections. The glyph must be a true turned "e"
> — not a normal "e", not an "a", not a question mark. Sitting on a clean
> cream (#FAF7F0) studio surface, soft diffused daylight, shallow depth of
> field, subtle soft shadows, square composition with generous margins.
> Style: cozy stop-motion children's-education aesthetic, tactile and
> friendly. Single subject, no other text or props.
