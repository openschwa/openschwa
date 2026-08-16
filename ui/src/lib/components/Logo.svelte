<script lang="ts">
  // The [ə] mark. A raster, not inline SVG, so unlike the rest of the UI it
  // cannot inherit `currentColor` — the artwork carries fixed colours.
  //
  // That forces the cream chip below. The glyph is #3b3e45, which against the
  // dark theme's #0f172a is about 1.6:1 — invisible. On light surfaces it
  // reads fine unaided, so the chip appears only where it is needed rather
  // than boxing the mark in permanently.
  //
  // Regenerate /logo.png with `just brand`; see assets/brand/README.md.
  let { size = '2rem' }: { size?: string } = $props();
</script>

<span class="chip" style="--logo-size: {size}">
  <img src="/logo.png" alt="" width="278" height="198" />
</span>

<style>
  /* Sized by height, width auto: the mark is landscape (roughly 3:2), and
     forcing it into a square would waste most of the height as empty margin. */
  .chip {
    display: inline-flex;
    flex: none;
    height: var(--logo-size);
    border-radius: 18%/25%;
  }
  .chip img {
    height: 100%;
    width: auto;
  }
  @media (prefers-color-scheme: dark) {
    .chip {
      background: #fcf8f0;
    }
  }
</style>
