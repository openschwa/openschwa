<script lang="ts">
  // Aligned phone intervals under the spectrogram; the anchor target for
  // feedback highlights ("this segment sounded like /z/").
  //
  // The mirror (M1) renders what the ear heard at the focus slot instead of
  // the drill's expected label: green when heard-as-intended, amber when a
  // different phone was heard, dimmed when the ear couldn't tell.
  //
  // Positioned by percentage over the same 0..durationS domain as the
  // spectrogram and pitch contour, so the three line up column for column.
  import type { Alignment } from '../api/types.gen';

  let {
    alignment = null,
    durationS = 0,
    highlightIndex = null,
    anchorIndex = null,
    mirrorIndex = null,
    mirrorHeard = null,
    mirrorOnTarget = false,
    mirrorUnsure = false,
  }: {
    alignment?: Alignment | null;
    durationS?: number;
    highlightIndex?: number | null;
    // A segmental_substitution verdict anchors here: rendered as an error,
    // because a flag is the one thing a learner must not miss.
    anchorIndex?: number | null;
    // The mirror's focus slot (phone_hearing item): rendered green (heard
    // as intended), amber with the heard phone (heard something else), or
    // dimmed (couldn't tell).
    mirrorIndex?: number | null;
    mirrorHeard?: string | null;
    mirrorOnTarget?: boolean;
    mirrorUnsure?: boolean;
  } = $props();

  const phones = $derived(alignment?.phones ?? []);

  function percent(seconds: number): number {
    return durationS > 0 ? (seconds / durationS) * 100 : 0;
  }
</script>

<div class="phone-timeline" class:uncertain={alignment?.status === 'low_confidence'}>
  {#if phones.length > 0 && durationS > 0}
    <div class="track">
      {#each phones as phone (phone.index)}
        <div
          class="phone"
          class:highlight={highlightIndex === phone.index && anchorIndex !== phone.index && mirrorIndex !== phone.index}
          class:error={anchorIndex === phone.index && mirrorIndex !== phone.index}
          class:mirror-ok={mirrorIndex === phone.index && mirrorOnTarget}
          class:mirror-other={mirrorIndex === phone.index && !mirrorOnTarget && !mirrorUnsure}
          class:mirror-unsure={mirrorIndex === phone.index && mirrorUnsure}
          style="left: {percent(phone.start_s)}%; width: {percent(phone.end_s - phone.start_s)}%"
          title={`/${phone.label}/  ${phone.start_s.toFixed(3)}–${phone.end_s.toFixed(3)}s${
            phone.gop != null ? `  ·  GOP ${phone.gop.toFixed(2)}` : ''
          }${anchorIndex === phone.index ? '  ·  flagged' : ''}${
            mirrorIndex === phone.index && mirrorHeard != null ? `  ·  heard /${mirrorHeard}/` : ''
          }`}
        >
          <span class="label">{mirrorIndex === phone.index && mirrorHeard != null ? mirrorHeard : phone.label}</span>
        </div>
      {/each}
    </div>
    {#if alignment?.status === 'low_confidence'}
      <p class="note">Alignment was uncertain — these boundaries are approximate.</p>
    {/if}
  {:else}
    <p class="empty">No alignment yet.</p>
  {/if}
</div>

<style>
  .phone-timeline {
    width: 100%;
  }
  .track {
    position: relative;
    height: 2.25rem;
    margin-top: 0.25rem;
  }
  .phone {
    position: absolute;
    top: 0;
    bottom: 0;
    display: grid;
    place-items: center;
    background: var(--border-soft);
    border-left: 1px solid var(--muted);
    border-right: 1px solid var(--muted);
    box-sizing: border-box;
    overflow: hidden;
  }
  .phone.highlight {
    background: var(--highlight);
    color: var(--highlight-fg);
  }
  .phone.error {
    background: var(--feedback-error-bg);
    border-color: var(--error);
    color: var(--error);
    font-weight: 700;
  }
  .phone.mirror-ok {
    background: var(--feedback-praise-bg);
    border-color: var(--ok);
    color: var(--ok);
    font-weight: 700;
  }
  .phone.mirror-other {
    background: var(--feedback-warn-bg);
    border-color: var(--warn);
    color: var(--warn);
    font-weight: 700;
  }
  .phone.mirror-unsure {
    opacity: 0.55;
  }
  .uncertain .phone {
    opacity: 0.65;
  }
  .label {
    font-family: 'Charis SIL', 'Doulos SIL', 'Gentium Plus', serif;
    font-size: 1rem;
    white-space: nowrap;
  }
  .empty,
  .note {
    margin: 0.25rem 0 0;
    color: var(--faint);
    font-size: 0.8125rem;
  }
</style>
