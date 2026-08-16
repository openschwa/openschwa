<script lang="ts">
  // The one component a correct UI must have: renders the confidence-gated
  // feedback[] list. Everything else on screen is optional evidence.
  import type { Alignment, FeedbackItem } from '../api/types.gen';

  let {
    feedback = [],
    alignment = null,
    analysed = false,
  }: {
    feedback?: FeedbackItem[];
    alignment?: Alignment | null;
    analysed?: boolean;
  } = $props();
</script>

<div class="feedback-panel">
  {#if !analysed}
    <p class="idle">Record an attempt to get feedback.</p>
  {:else if feedback.length > 0}
    <ul>
      {#each feedback as item (item.id)}
        <li class={item.severity}>{item.message}</li>
      {/each}
    </ul>
  {:else if alignment && alignment.status !== 'failed'}
    <!--
      An empty feedback list after a successful analysis is the honest M0 state,
      not a bug: no judgement type has cleared the shipping bar (`eval/README.md`) yet,
      so the engine shows evidence and withholds verdicts. Saying so beats an
      empty panel that reads as a failure.
    -->
    <p class="idle">
      Aligned successfully. Pronunciation scoring arrives in M1 — until the eval
      harness proves it, the engine shows you the measurements rather than
      guessing at a verdict.
    </p>
  {:else}
    <p class="idle">No feedback for this attempt.</p>
  {/if}
</div>

<style>
  .feedback-panel {
    margin: 0;
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 0.5rem;
  }
  li {
    padding: 0.625rem 0.75rem;
    border-radius: 4px;
    border-left: 3px solid;
  }
  li.error {
    background: var(--feedback-error-bg);
    border-color: var(--error);
  }
  li.warning {
    background: var(--feedback-warn-bg);
    border-color: var(--warn);
  }
  li.praise {
    background: var(--feedback-praise-bg);
    border-color: var(--ok);
  }
  .idle {
    margin: 0;
    color: var(--faint);
    font-size: 0.875rem;
    line-height: 1.5;
  }
</style>
