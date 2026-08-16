<script lang="ts">
  // Renders the STFT of the client-held Float32 buffer on a canvas (M0),
  // with engine annotation overlays — VOT boxes, voicing bars — in M3.
  //
  // The time axis is always 0..durationS, matching PhoneTimeline and
  // PitchContour, so the three stack into one aligned view.
  import type { Annotation } from '../api/types.gen';
  import { computeStft } from '../audio/stft';
  import { intensityToRgb } from './colormap';

  let {
    samples = null,
    sampleRate = 0,
    durationS = 0,
    speechInterval = null,
    annotations = [],
    height = 160,
  }: {
    samples?: Float32Array | null;
    sampleRate?: number;
    durationS?: number;
    speechInterval?: [number, number] | null;
    annotations?: Annotation[];
    height?: number;
  } = $props();

  let canvas = $state<HTMLCanvasElement | null>(null);
  let width = $state(0);

  const spectrogram = $derived(
    samples && samples.length > 0 && sampleRate > 0 ? computeStft(samples, sampleRate) : null,
  );

  const DYNAMIC_RANGE_DB = 70;

  $effect(() => {
    const target = canvas;
    const data = spectrogram;
    if (!target || !data || width <= 0 || durationS <= 0) return;

    const ratio = window.devicePixelRatio || 1;
    const pixelWidth = Math.max(1, Math.floor(width * ratio));
    const pixelHeight = Math.max(1, Math.floor(height * ratio));
    target.width = pixelWidth;
    target.height = pixelHeight;

    const context = target.getContext('2d');
    if (!context) return;

    const image = context.createImageData(pixelWidth, pixelHeight);
    const { magnitudes, hopS, startS, binCount } = data;

    for (let x = 0; x < pixelWidth; x += 1) {
      // Column -> time -> frame. Frames and pixels rarely align, so the nearest
      // frame is used rather than interpolating across a discontinuity.
      const time = (x / pixelWidth) * durationS;
      const frame = Math.round((time - startS) / hopS);
      const row = frame >= 0 && frame < magnitudes.length ? magnitudes[frame] : null;

      for (let y = 0; y < pixelHeight; y += 1) {
        const offset = (y * pixelWidth + x) * 4;
        // Canvas y grows downward; low frequencies belong at the bottom.
        const bin = Math.round((1 - y / (pixelHeight - 1)) * (binCount - 1));
        const db = row ? row[bin] : -DYNAMIC_RANGE_DB;
        const [r, g, b] = intensityToRgb(1 + db / DYNAMIC_RANGE_DB);
        image.data[offset] = r;
        image.data[offset + 1] = g;
        image.data[offset + 2] = b;
        image.data[offset + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);

    // The engine analysed only the region inside the VAD interval; shading the
    // rest keeps a learner from reading meaning into untouched audio.
    if (speechInterval) {
      context.fillStyle = 'rgba(10, 12, 20, 0.55)';
      const [start, end] = speechInterval;
      context.fillRect(0, 0, (start / durationS) * pixelWidth, pixelHeight);
      context.fillRect(
        (end / durationS) * pixelWidth,
        0,
        pixelWidth - (end / durationS) * pixelWidth,
        pixelHeight,
      );
    }
  });
</script>

<div class="spectrogram-view" style="height: {height}px" bind:clientWidth={width}>
  {#if samples && samples.length > 0}
    <canvas bind:this={canvas} style="height: {height}px" aria-label="Spectrogram of your recording"
    ></canvas>
    {#if annotations.length > 0}
      <!-- M3 draws VOT boxes and voicing bars over this canvas. -->
    {/if}
  {:else}
    <p class="empty">No recording yet.</p>
  {/if}
</div>

<style>
  .spectrogram-view {
    position: relative;
    width: 100%;
    background: #440154;
    border-radius: 4px;
    overflow: hidden;
  }
  canvas {
    display: block;
    width: 100%;
  }
  .empty {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    margin: 0;
    color: #cbd5e1;
    font-size: 0.875rem;
  }
</style>
