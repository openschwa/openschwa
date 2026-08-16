<script lang="ts">
  // First-run panel: the acoustic model is a gigabyte-plus download, so the
  // app stays usable and explains itself rather than silently failing to
  // analyse (docs/architecture.md, risks).
  import { pullModel, type ModelCatalog } from '../api/client';

  let { models = null, onready }: { models?: ModelCatalog | null; onready?: () => void } = $props();

  let downloading = $state(false);
  let bytesDone = $state(0);
  let bytesTotal = $state(0);
  let error = $state<string | null>(null);

  const missing = $derived((models?.models ?? []).filter((m) => m.state !== 'ready'));
  const runtimeMissing = $derived((models?.models ?? []).some((m) => !m.runtime_available));
  const percent = $derived(bytesTotal > 0 ? Math.round((bytesDone / bytesTotal) * 100) : 0);

  function gb(bytes: number): string {
    return `${(bytes / 1e9).toFixed(1)} GB`;
  }

  async function download(modelId: string) {
    downloading = true;
    error = null;
    bytesDone = 0;
    try {
      for await (const event of pullModel(modelId)) {
        if (event.error) throw new Error(event.error);
        if (event.bytes_total) {
          bytesDone = event.bytes_done ?? 0;
          bytesTotal = event.bytes_total;
        }
        if (event.done) onready?.();
      }
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      downloading = false;
    }
  }
</script>

{#if missing.length > 0}
  <section class="setup">
    <h2>One-time setup</h2>
    {#if runtimeMissing}
      <p>
        The engine is running without its ML extra, so it can measure audio but
        cannot align phones. Install it with
        <code>uv sync --extra ml</code> in <code>engine/</code>, then restart the engine.
      </p>
    {:else}
      {#each missing as model (model.id)}
        <p>
          The alignment model <code>{model.repo_id}</code> ({gb(model.download_bytes)},
          {model.license}) has not been downloaded yet. It runs entirely on this machine —
          nothing you record is ever uploaded anywhere.
        </p>
        {#if downloading}
          <progress value={bytesDone} max={bytesTotal || 1}></progress>
          <p class="progress-text">
            {percent}% · {gb(bytesDone)} of {gb(bytesTotal)}
          </p>
        {:else}
          <button onclick={() => download(model.id)}>Download model</button>
        {/if}
      {/each}
    {/if}
    {#if error}
      <p class="error">{error}</p>
    {/if}
  </section>
{/if}

<style>
  .setup {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
    background: var(--surface);
    margin-bottom: 1.5rem;
  }
  h2 {
    margin: 0 0 0.5rem;
    font-size: 1rem;
  }
  p {
    margin: 0 0 0.75rem;
    font-size: 0.875rem;
    line-height: 1.5;
  }
  code {
    background: var(--border-soft);
    padding: 0.05rem 0.3rem;
    border-radius: 3px;
    font-size: 0.85em;
  }
  progress {
    width: 100%;
  }
  .progress-text {
    margin: 0.25rem 0 0;
    color: var(--muted);
  }
  .error {
    color: var(--error);
  }
  button {
    font: inherit;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    border: 1px solid var(--border);
    background: var(--accent);
    color: var(--accent-fg);
    cursor: pointer;
  }
</style>
