<script lang="ts">
  // Live input level while recording.
  //
  // Exists because "the app says I'm too quiet" is otherwise undebuggable from
  // the outside: a muted input, the wrong device, and a working one all look
  // identical until an analysis comes back. A meter answers "is the microphone
  // producing anything at all" while you are still speaking.
  //
  // Scaled in dBFS, not linear amplitude — speech spans a range that a linear
  // bar renders as a barely-moving sliver.
  let {
    rms = 0,
    active = false,
    measured = false,
    floorDb = -70,
  }: { rms?: number; active?: boolean; measured?: boolean; floorDb?: number } = $props();

  const dbfs = $derived(20 * Math.log10(Math.max(rms, 1e-10)));
  const fraction = $derived(Math.max(0, Math.min(1, 1 - dbfs / floorDb)));

  // Thresholds match the engine: below -70 dBFS is a dead input, and that is
  // the only level the engine refuses. -40 is advisory, never a rejection.
  const state = $derived(dbfs < -70 ? 'dead' : dbfs < -40 ? 'low' : 'good');
</script>

<div class="meter" class:active>
  <div class="bar">
    <div class="fill {state}" style="width: {fraction * 100}%"></div>
  </div>
  <span class="readout">
    {#if !active}
      &nbsp;
    {:else if !measured}
      listening…
    {:else if state === 'dead'}
      no signal — check the input device
    {:else}
      {dbfs.toFixed(0)} dBFS{state === 'low' ? ' · quiet, but usable' : ''}
    {/if}
  </span>
</div>

<style>
  .meter {
    display: grid;
    gap: 0.25rem;
    margin-top: 0.75rem;
    opacity: 0.35;
    transition: opacity 0.2s;
  }
  .meter.active {
    opacity: 1;
  }
  .bar {
    height: 0.5rem;
    border-radius: 999px;
    background: var(--surface-strong);
    overflow: hidden;
  }
  .fill {
    height: 100%;
    transition: width 0.06s linear;
  }
  .fill.dead {
    background: var(--error);
  }
  .fill.low {
    background: var(--warn);
  }
  .fill.good {
    background: var(--ok);
  }
  .readout {
    font-size: 0.75rem;
    color: var(--faint);
    font-variant-numeric: tabular-nums;
  }
</style>
