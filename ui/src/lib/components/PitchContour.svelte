<script lang="ts">
  // Student-vs-reference F0 overlay (semitones re speaker median), aligned to
  // the phone timeline. M0 draws the student track; M2 adds the teacher
  // reference and the nuclear-tone verdict.
  //
  // Semitones rather than hertz because that is what makes two speakers
  // comparable at all (docs/architecture.md §4).
  import type { Prosody } from '../api/types.gen';

  let {
    prosody = null,
    durationS = 0,
    height = 90,
  }: { prosody?: Prosody | null; durationS?: number; height?: number } = $props();

  let canvas = $state<HTMLCanvasElement | null>(null);
  let width = $state(0);

  /** Widest of the data range or ±6 semitones, so a flat delivery looks flat. */
  const MIN_RANGE_ST = 6;

  const voicedCount = $derived(
    prosody?.f0.semitones.reduce<number>((n, v) => n + (v == null ? 0 : 1), 0) ?? 0,
  );

  $effect(() => {
    const target = canvas;
    const track = prosody?.f0;
    if (!target || !track || width <= 0 || durationS <= 0) return;

    const ratio = window.devicePixelRatio || 1;
    target.width = Math.max(1, Math.floor(width * ratio));
    target.height = Math.max(1, Math.floor(height * ratio));
    const context = target.getContext('2d');
    if (!context) return;

    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);

    // Canvas gets no cascade, so theme tokens are read off the element.
    const styles = getComputedStyle(target);
    const gridColour = styles.getPropertyValue('--pitch-grid').trim() || '#cbd5e1';
    const lineColour = styles.getPropertyValue('--pitch-line').trim() || '#2563eb';

    const values = track.semitones.filter((v): v is number => v != null);
    if (values.length === 0) return;
    const span = Math.max(Math.max(...values) - Math.min(...values), MIN_RANGE_ST);
    const centre = (Math.max(...values) + Math.min(...values)) / 2;
    const top = centre + span / 2;

    const x = (index: number) => ((track.start_s + index * track.hop_s) / durationS) * width;
    const y = (semitones: number) => ((top - semitones) / span) * (height - 12) + 6;

    // Speaker's median: the zero line every value is expressed against.
    context.strokeStyle = gridColour;
    context.lineWidth = 1;
    context.setLineDash([3, 3]);
    context.beginPath();
    context.moveTo(0, y(0));
    context.lineTo(width, y(0));
    context.stroke();
    context.setLineDash([]);

    // Unvoiced frames break the line rather than being interpolated across:
    // a drawn-through gap would imply pitch where there is no voicing.
    context.strokeStyle = lineColour;
    context.lineWidth = 2;
    context.lineJoin = 'round';
    context.beginPath();
    let drawing = false;
    track.semitones.forEach((value, index) => {
      if (value == null) {
        drawing = false;
        return;
      }
      if (drawing) context.lineTo(x(index), y(value));
      else context.moveTo(x(index), y(value));
      drawing = true;
    });
    context.stroke();
  });
</script>

<div class="pitch-contour" bind:clientWidth={width}>
  {#if prosody && voicedCount > 0}
    <canvas
      bind:this={canvas}
      style="height: {height}px"
      aria-label="Pitch contour in semitones relative to your median"
    ></canvas>
    <p class="scale">
      semitones re median{prosody.f0.median_hz
        ? ` (${prosody.f0.median_hz.toFixed(0)} Hz)`
        : ''}
    </p>
  {:else}
    <p class="empty">No pitch data yet.</p>
  {/if}
</div>

<style>
  .pitch-contour {
    width: 100%;
  }
  canvas {
    display: block;
    width: 100%;
  }
  .empty,
  .scale {
    margin: 0.25rem 0 0;
    color: var(--faint);
    font-size: 0.8125rem;
  }
</style>
