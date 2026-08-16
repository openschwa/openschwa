<script lang="ts">
  import {
    analyze,
    getExercise,
    getExercises,
    getHealth,
    getModels,
    referenceAudioUrl,
    type AnalysisResult,
    type ExerciseDetail,
    type ExerciseSummary,
    type HealthResponse,
    type ModelCatalog,
  } from './lib/api/client';
  import { MicRecorder, type Recording } from './lib/audio/capture';
  import FeedbackPanel from './lib/components/FeedbackPanel.svelte';
  import LevelMeter from './lib/components/LevelMeter.svelte';
  import Logo from './lib/components/Logo.svelte';
  import ModelSetup from './lib/components/ModelSetup.svelte';
  import PhoneTimeline from './lib/components/PhoneTimeline.svelte';
  import PitchContour from './lib/components/PitchContour.svelte';
  import SpectrogramView from './lib/components/SpectrogramView.svelte';

  let health = $state<HealthResponse | null>(null);
  let models = $state<ModelCatalog | null>(null);
  let engineError = $state<string | null>(null);

  let exercises = $state<ExerciseSummary[]>([]);
  let exercise = $state<ExerciseDetail | null>(null);

  let recorder = $state<MicRecorder | null>(null);
  let recording = $state(false);
  let busy = $state(false);
  let captured = $state<Recording | null>(null);
  let analysis = $state<AnalysisResult | null>(null);
  let drillError = $state<string | null>(null);
  let inputLevel = $state(0);
  let levelMeasured = $state(false);

  /**
   * Peak-hold with decay, the way any level meter behaves.
   *
   * Raw per-block RMS arrives every few milliseconds and drops to nothing in
   * the gaps between words, which made the meter flash "no signal" mid-sentence
   * — alarming, and exactly the wrong message when the input is fine.
   */
  // Per audio block (~375/s at 48 kHz), giving roughly a 1.5 s fall from speech
  // level to silence. Faster than this and an ordinary pause between words
  // reads as a dead microphone; slower and the meter stops tracking speech.
  const LEVEL_DECAY = 0.992;
  function trackLevel(rms: number) {
    inputLevel = Math.max(rms, inputLevel * LEVEL_DECAY);
    levelMeasured = true;
  }

  const focusPhone = $derived(exercise?.phones.find((p) => p.focus) ?? null);
  const confusions = $derived(focusPhone?.confusions ?? []);
  const quality = $derived(analysis?.audio.quality ?? null);
  const lowLevel = $derived(
    quality?.speech_level_dbfs != null && quality.speech_level_dbfs < -40,
  );

  async function loadEngineState() {
    health = await getHealth();
    models = await getModels();
  }

  async function init() {
    // Load the capture worklet up front so pressing Record starts recording
    // immediately rather than after a module fetch.
    recorder = new MicRecorder();
    recorder.setLevelListener(trackLevel);
    void recorder.prepare().catch(() => {
      /* Recording will retry on demand and report the failure then. */
    });
    try {
      await loadEngineState();
      const catalog = await getExercises();
      exercises = catalog.exercises ?? [];
      if (exercises.length > 0) await select(exercises[0].id);
    } catch {
      engineError = 'Engine not reachable — start it with `uv run openschwa-engine` in engine/.';
    }
  }

  async function select(id: string) {
    // Each drill is a fresh attempt; carrying the previous result over would
    // leave a timeline on screen that belongs to a different word.
    analysis = null;
    captured = null;
    drillError = null;
    exercise = await getExercise(id);
  }

  async function toggleRecording() {
    drillError = null;
    if (recording) {
      await finishRecording();
      return;
    }
    try {
      recorder ??= new MicRecorder();
      recorder.setLevelListener(trackLevel);
      inputLevel = 0;
      levelMeasured = false;
      await recorder.start();
      recording = true;
      analysis = null;
      captured = null;
    } catch (error) {
      drillError = error instanceof Error ? error.message : String(error);
      recording = false;
    }
  }

  async function finishRecording() {
    if (!recorder || !exercise) return;
    recording = false;
    busy = true;
    inputLevel = 0;
    try {
      const result = await recorder.stop();
      // Keep the Float32 buffer: the client renders its own spectrogram and
      // never re-downloads audio it already has.
      captured = result;
      analysis = await analyze(result.wav, exercise.id);
    } catch (error) {
      drillError = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
    }
  }

  function playReference() {
    if (exercise?.has_reference_audio) void new Audio(referenceAudioUrl(exercise.id)).play();
  }

  init();
</script>

<main>
  <header>
    <h1><Logo /> OpenSchwa <span class="ipa">/ˈoʊpən ʃwɑː/</span></h1>
    {#if health}
      <p class="status">
        engine {health.engine_version} · contract v{health.schema_version} ·
        {#if health.analysis_available}
          <span class="ready">alignment ready</span>
        {:else}
          <span class="warn">alignment unavailable</span>
        {/if}
      </p>
    {:else if engineError}
      <p class="status error">{engineError}</p>
    {:else}
      <p class="status">connecting to engine…</p>
    {/if}
  </header>

  {#if health && !health.analysis_available}
    <ModelSetup {models} onready={loadEngineState} />
  {/if}

  {#if exercises.length > 0}
    <nav aria-label="Exercises">
      {#each exercises as summary (summary.id)}
        <button
          class="pill"
          class:selected={summary.id === exercise?.id}
          onclick={() => select(summary.id)}
          disabled={recording || busy}
        >
          {summary.text}
          {#if summary.focus_phone}<span class="ipa">/{summary.focus_phone}/</span>{/if}
        </button>
      {/each}
    </nav>
  {/if}

  {#if exercise}
    <section class="drill">
      <h2>{exercise.title}</h2>
      <p class="target">{exercise.text}</p>
      <p class="ipa transcription">
        {#each exercise.phones as phone (phone.index)}<span class:focus={phone.focus}
            >{phone.ph}</span
          >{/each}
      </p>
      {#if focusPhone}
        <p class="hint">
          Focus: <strong>/{focusPhone.ph}/</strong>
          {#if confusions.length > 0}
            — often heard as {confusions.map((c) => `/${c}/`).join(', ')}
          {/if}
        </p>
      {/if}
      {#if exercise.learner_notes}
        <p class="notes">{exercise.learner_notes}</p>
      {/if}

      <div class="controls">
        <button class="record" class:recording onclick={toggleRecording} disabled={busy}>
          {#if recording}Stop{:else if busy}Analysing…{:else}Record{/if}
        </button>
        {#if exercise.has_reference_audio}
          <button onclick={playReference} disabled={recording || busy}>Play reference</button>
        {/if}
      </div>

      <LevelMeter rms={inputLevel} active={recording} measured={levelMeasured} />

      {#if drillError}
        <p class="status error">{drillError}</p>
      {/if}
    </section>

    <section class="analysis">
      <SpectrogramView
        samples={captured?.samples ?? null}
        sampleRate={captured?.sampleRate ?? 0}
        durationS={captured?.durationS ?? 0}
        speechInterval={analysis?.audio.speech_interval_s ?? null}
        annotations={analysis?.annotations ?? []}
      />
      <PhoneTimeline
        alignment={analysis?.alignment ?? null}
        durationS={captured?.durationS ?? 0}
        highlightIndex={focusPhone?.index ?? null}
      />
      <PitchContour
        prosody={analysis?.prosody ?? null}
        durationS={captured?.durationS ?? 0}
      />

      {#if quality?.clipping}
        <p class="status warn">Recording clipped — move back from the microphone.</p>
      {:else if quality?.too_quiet}
        <p class="status warn">
          No signal from the microphone — check that the right input device is
          selected and not muted.
        </p>
      {:else if lowLevel}
        <!-- Advice, not a refusal: the analysis above ran normally. Level is
             rarely low now that takes are peak-normalised before upload — when
             it still is, the crest factor is high, which usually means distance
             from the microphone. That is the one thing still worth suggesting:
             "raise your gain" stopped being actionable once the app started
             applying the gain itself. -->
        <p class="status warn">
          Recording level is low ({quality?.speech_level_dbfs?.toFixed(0)} dBFS) even
          after boosting — it analysed fine, but getting closer to the microphone
          would give a stronger signal.
        </p>
      {/if}

      <FeedbackPanel
        feedback={analysis?.feedback ?? []}
        alignment={analysis?.alignment ?? null}
        analysed={analysis !== null}
      />
    </section>
  {/if}
</main>

<style>
  main {
    font-family: system-ui, sans-serif;
    max-width: 48rem;
    margin: 2rem auto;
    padding: 0 1rem;
    color: var(--fg);
  }
  h1 {
    font-size: 1.5rem;
    margin: 0;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  .ipa {
    font-family: 'Charis SIL', 'Doulos SIL', 'Gentium Plus', serif;
    font-weight: normal;
  }
  h1 .ipa {
    opacity: 0.6;
    font-size: 1rem;
  }
  .status {
    margin: 0.25rem 0 1.25rem;
    font-size: 0.8125rem;
    color: var(--muted);
  }
  .status.error {
    color: var(--error);
  }
  .ready {
    color: var(--ok);
  }
  .warn {
    color: var(--focus);
  }
  nav {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.25rem;
  }
  button {
    font: inherit;
    padding: 0.45rem 0.9rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg);
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.55;
    cursor: default;
  }
  .pill.selected {
    background: var(--accent);
    color: var(--accent-fg);
    border-color: var(--accent);
  }
  .drill {
    border: 1px solid var(--border-soft);
    border-radius: 6px;
    padding: 1.25rem;
    margin-bottom: 1.5rem;
  }
  .drill h2 {
    margin: 0 0 0.75rem;
    font-size: 0.9375rem;
    font-weight: 600;
    color: var(--muted);
  }
  .target {
    font-size: 2.5rem;
    margin: 0;
    line-height: 1.1;
  }
  .transcription {
    font-size: 1.5rem;
    margin: 0.25rem 0 0.75rem;
    color: var(--muted);
  }
  .transcription .focus {
    color: var(--focus);
    font-weight: 700;
  }
  .hint,
  .notes {
    margin: 0 0 0.5rem;
    font-size: 0.875rem;
    color: var(--muted);
  }
  .controls {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
  }
  .record {
    border-radius: 4px;
    background: var(--accent);
    color: var(--accent-fg);
    border-color: var(--accent);
    min-width: 8rem;
  }
  .record.recording {
    background: var(--error);
    border-color: var(--error);
    color: #ffffff;
  }
  .analysis {
    display: grid;
    gap: 0.75rem;
  }
</style>
