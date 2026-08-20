# OpenSchwa architecture

Living reference: the system as designed, and why. Rationale sits next to the
thing it constrains rather than in a separate decision log — §1 holds the
invariants, §4 the pipeline arguments, §8 the risks and their mitigations, §9
the gaps M0 left open on purpose. Status: **M0 complete** — the skeleton walks
(record → align → render).

Plain-language version for linguists and non-coders:
[architecture-for-linguists.md](architecture-for-linguists.md).

## 1. Design principles

1. **The engine returns judgments and annotations, never pixels.** The client
   owns raw audio and all rendering (waveform, spectrogram, pitch curve). The
   engine returns time intervals, scores, labels, and confidence. Payloads
   stay small, and web/desktop/server deployments stay identical.
2. **No feedback beats wrong feedback.** Every judgment carries a calibrated
   confidence; only above-threshold items reach `feedback[]`; a feedback
   *type* ships only after the eval harness proves its precision against the
   shipping bar (§7). Refusing to judge ("retry") is a first-class outcome.
3. **Scripted drills only, never open transcription.** The engine always
   receives the target phone sequence and the expected confusion set, so
   segmental scoring is closed-set discrimination ("more like /ð/ or /z/?").
   General ASR is the wrong instrument by construction, not merely a weaker
   one: it is trained to be robust to accents — it hears *zis* and transcribes
   *this* — which is exactly the property a mispronunciation detector must not
   have. Open phone-level transcription of L2 speech sits near 60–80% precision
   in the MDD literature, unusable as student-facing truth, and its false
   positives destroy learner trust faster than silence ever could. Scripted
   drills also match minimal-pair pedagogy, so the constraint costs the product
   nothing.
4. **Client-server split from day one.** The engine is a localhost HTTP
   service; the UI is a static SPA. Tauri wraps both later; hosting deploys
   the same engine. Nothing may assume co-location beyond a base URL.
   Inference runs wherever the engine runs, always on CPU; there is no separate
   cloud-inference tier, which would add cost from day one and end offline use.

## 2. Repo layout

```
schemas/analysis_result.v1.schema.json   # THE shared contract (generated, committed)
engine/          # Python: FastAPI + analysis pipeline (uv, ruff, pytest, mypy)
  src/openschwa_engine/
    server.py config.py app.py           # app factory, settings (OPENSCHWA_* env), desktop entry
    api/         # thin routers: health, exercises, analyze, models
    schemas/     # pydantic contract models — SOURCE OF TRUTH + export script
    audio/       # decode → resample 16k mono → VAD trim → quality checks   (M0)
    alignment/   # CTC phoneme posteriors + forced alignment                (M0/M1)
    scoring/     # GOP + closed-set contrasts + calibration.yaml            (M1)
    prosody/     # F0, semitones, DTW vs reference, nuclear tone rules      (M2)
    measurements/# durations, voicing, VOT, formants                        (M1–M3)
    feedback/    # composer: evidence + thresholds → gated feedback[]       (M1)
    models/      # registry, download/cache, phone-set mapping              (M0/M1)
    content/     # pack loading + validation, reference precompute          (M0)
  packaging/     # PyInstaller spec for the desktop bundle + its notes
ui/              # Vite + Svelte 5 SPA (no SvelteKit); Vitest; svelte-check
  src/lib/api/         # fetch client + types.gen.ts (generated from schema)
  src/lib/audio/       # AudioWorklet capture, WAV encode, STFT rendering
  src/lib/components/  # SpectrogramView, PitchContour, PhoneTimeline, FeedbackPanel
content/         # exercise packs: YAML validated by content/schema/exercise.schema.json
eval/            # offline eval harness; corpora never committed; reports/ committed
docs/            # architecture.md (coders) + architecture-for-linguists.md (no-code)
justfile         # setup / models / dev / test / schema / package / eval
```

Toolchain: **uv** (engine), **npm + Vite** (UI), **just** (tasks), GitHub
Actions CI (lint, tests, schema-drift gates on both sides). torch and the
acoustic-model runtime sit in the engine's optional `ml` extra; CI installs
without it on purpose, so the degraded path stays exercised (§4).

Why this shape: the speech/ML ecosystem — torch, forced aligners, Praat via
parselmouth — exists in Python and effectively nowhere else, so the engine's
language was never a free choice. The UI's was. Svelte 5 without SvelteKit,
because the core surfaces (spectrogram, pitch contour, phone timeline) are
imperative canvas rendering, where a compiled framework with a small runtime
fits and React's re-render model buys nothing — and because a static SPA that
the engine itself serves has no use for SvelteKit's server half.

## 3. The API contract

Pydantic models in `engine/src/openschwa_engine/schemas/analysis.py` are the
single source of truth. `just schema` exports the JSON Schema
(`schemas/analysis_result.v1.schema.json`) and the UI's generated types
(`ui/src/lib/api/types.gen.ts`); CI fails if either drifts. Versioning: URL
prefix `/v1/` + `schema_version` field; breaking changes mean a new v2 module
and schema file, never mutation of v1.

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/health` | status, versions, model cache states |
| GET | `/v1/exercises` | exercise summaries |
| GET | `/v1/exercises/{id}` | full spec (UI renders target text/IPA/focus before recording) |
| GET | `/v1/exercises/{id}/reference-audio` | teacher reference WAV |
| POST | `/v1/analyze` | multipart WAV + `exercise_id` → `AnalysisResult`; `?include_ungated=true` for dev/eval |
| GET | `/v1/models` | model manifest + cache state |
| POST | `/v1/models/pull` | download with NDJSON progress (first-run UX) |

**AnalysisResult v1** (see the pydantic models for exact shapes):
`audio` (duration, post-VAD interval, quality checks) · `alignment` (status
`ok|low_confidence|failed`, words, phones with intervals + GOP + calibrated
score + confidence) · `contrasts` (posteriors renormalized over {target} ∪
confusions → verdict + calibrated confidence) · `prosody` (F0 in semitones re
speaker median, reference contour included, DTW distance, nuclear tone) ·
`annotations` (flat tagged union: `vot|voicing|duration|formants`, extensible
in minor versions — clients ignore unknown types) · `feedback` (ONLY
gate-passing items; anchor interval for UI highlighting; evidence pointers).

Contract invariants: all times are seconds in the **original upload
timeline**; `feedback[]` is the only thing a naive UI must render;
`alignment.status != ok` short-circuits to a single "retry" item.

**Serving the UI.** When a built SPA is present the engine mounts it at `/`,
after the `/v1` routes. That makes the packaged app and the hosted deployment a
single origin, which is why the client addresses the API relatively rather than
at a fixed port — the engine auto-increments past a busy 8577, and a compiled-in
port would break the moment it did. In development Vite serves the UI instead
and the client uses the localhost engine URL.

**Audio capture:** AudioWorklet → Float32 → client-encoded 16-bit PCM WAV
upload. Not MediaRecorder/opus (lossy; server decode would need ffmpeg).
`getUserMedia` with echoCancellation/noiseSuppression/autoGainControl
**disabled** — browser DSP destroys the cues being measured. Engine resamples
(soxr → 16 kHz mono). The client keeps its Float32 buffer and renders
spectrogram/waveform locally; it never re-downloads its own audio.

## 4. Engine pipeline (per /v1/analyze)

decode → resample → VAD trim (silero-vad, energy fallback) → quality checks →
CTC phoneme posteriors + forced alignment of the exercise phone sequence →
per-phone GOP → focus-phone closed-set contrast scoring (Platt-calibrated) →
prosody (parselmouth F0 → octave-jump cleanup → semitones → DTW vs cached
reference → rule-based nuclear tone over the author-marked nucleus) →
measurements (durations, voicing fraction, VOT for utterance-initial stops,
formant medians with reliability score) → feedback composer (thresholds from
committed `scoring/calibration.yaml`, produced only by the eval harness).

Known issues owned up front: **CTC posterior peakiness** (see §8 for the
mitigation, `alignment/ctc.py` for the detail); **phone-set mapping is a real
module** — one canonical IPA inventory; per-model mapping tables with
round-trip tests. Models: pinned HF manifest, resumable downloads to a
platformdirs cache (`OPENSCHWA_MODEL_DIR` override), lazy singleton loading,
single-worker uvicorn. ONNX Runtime export is the designated M4 packaging
escape hatch. Nuclear tone is decided by piecewise-slope rules rather than a
classifier on purpose — rules over an author-marked nucleus are transparent and
debuggable, and a model earns its place only if the rules provably fall short.

**Confidence and GOP measure different things, and the split is load-bearing.**
`confidence` scores the *best* token in a phone's frames — "is the model sure
something phone-like is happening here", i.e. can this recording be analysed at
all. `gop` scores the *target* token against that best — "was it the phone we
asked for". Only confidence feeds the gate. Scoring confidence on the target
instead collapses the two, and a clear mispronunciation then presents as a
processing failure: the learner is told "try again" at exactly the moment they
should be told what they said. `alignment/ctc.py` carries the tests that pin
this apart.

**Absolute level is not a usability test, and must never gate analysis.** The
app requires `autoGainControl: false`, and browsers that honour it return a raw
capture commonly 15-25 dB below what a consumer recorder produces. That audio is
entirely analysable: its SNR is unchanged, and the acoustic model normalises its
input to zero mean and unit variance before it sees anything. An absolute
`too_quiet` threshold therefore rejected good recordings and told people to speak
louder, which could not help — the whole chain was scaled down, not the voice.
`audio.quality.too_quiet` now means only that the input is *dead*; the measured
`speech_level_dbfs` is reported so the UI can advise without refusing, and
whether a quiet recording can actually be scored is decided downstream by
alignment confidence, which measures it directly. The UI shows a live input
meter during recording so a muted or misrouted device is visible immediately
rather than inferred from a failed analysis.

**The engine degrades rather than failing.** The acoustic model and its runtime
live in the optional `ml` extra. Without them the engine still starts, serves
exercises, measures audio and F0, and returns a schema-valid result whose
alignment status is `failed` and whose `feedback[]` holds one retry. Recording
problems (no speech, clipping, too quiet) are diagnosed before the model is
consulted, so those messages stay specific even on an engine with no weights.

## 5. Content format

YAML packs under `content/packs/<pack>/`, validated against
`content/schema/exercise.schema.json` at load (fail loud). An exercise: id,
type (`minimal_pair|word|sentence|intonation`), text, IPA, phone list with
exactly one `focus: true` + `confusions` for segmental types, optional
`pair_with`, `reference_audio`, optional `prosody` block
(`nuclear_syllable_index`, `expected_tone`), `learner_notes`. Reference audio
is teacher-recorded; the engine precomputes reference F0/alignment at pack
load. First pack: `en-dh-z` (/ð/ → /z/).

## 6. Milestones

- **M0 — walking skeleton. ✅ Done.** Record in browser → aligned phones + F0 →
  UI renders timeline + pitch over client-rendered spectrogram. All accept
  criteria met: `just dev` round-trips end to end; responses validate against
  the committed schema; a 3.5 s clip analyses in ~270 ms warm on CPU (target
  was < 2 s for 3 s); CI green. Provisional aligner is
  `facebook/wav2vec2-lv-60-espeak-cv-ft`, pinned by commit; the shipping choice
  belongs to the M1 bake-off below. M0 ships **no pronunciation verdicts**:
  `retry` is the only feedback kind, because no judgement type has cleared the
  shipping bar (§7) yet.
- **M1 — one contrast put to the bar (/ð/→/z/). ✅ Done, negative result.**
  Everything the milestone promised *machinery*-wise shipped and is tested:
  the eval harness over L2-ARCTIC + speechocean762 (adapters, checkpointed
  runner, Platt fitting, precision-first threshold sweep), closed-set
  contrast scoring with four aggregations (label-frame
  mean, spike frame, frame vote, GOP), the calibration pipeline
  (scoring/calibration.yaml, written only by the harness), silero VAD behind
  detect_speech, the gated anchored segmental_substitution feedback path,
  and the second bake-off candidate (charsiu CTC, manifest + phone table +
  committed snapshots — swapping stayed configuration, never pipeline code).
  **The bake-off verdict is negative**: over 6,678 labeled token runs per
  model, neither candidate discriminates /ð/ vs {z,d,v} above chance —
  held-out AUC 0.59 (espeak) / 0.58 (charsiu), best operating point precision
  0.19 at recall 0.01, against the **precision ≥ 0.90 at recall ≥ 0.4**
  accept criterion. Per policy no calibration was committed and no verdicts
  ship; the engine stays retry-only. For the alignment job itself the
  bake-off did decide: charsiu won alignment sanity (0.90 vs 0.82 mean
  confidence on L2 speech), size (0.38 vs 1.26 GB), and latency (135 vs
  357 ms median), and became the default aligner. Evidence:
  eval/reports/m1-bakeoff-2026-08-18-*.md. **Accept criterion NOT met —
  recorded, not waived.**
- **Option 3 — fine-tuned 4-class {ð,z,d,v} judge. ✅ Done, negative result.**
  Trained Wav2Vec2ForCTC from the charsiu base on the user's RTX 4060 laptop:
  8,177 L2-ARCTIC train-split segments + speechocean762 train segments (capped
  at 700 to keep the Mandarin share balanced), frozen head then full
  fine-tune, v1→v4 iterations, integrated as `dh-contrast-v1` (role=contrast).
  Two real bugs were found and fixed along the way: an export that wrote
  silent segments (int16 scaling), and — the one that invalidated every early
  exam — **CTC blank dominance in the exam scoring**: the judge emits ~95%
  blank frames, and the old per-frame closed-set renormalization gave each
  blank frame's dust a full vote, collapsing the exam to chance (AUC 0.51)
  while training-time validation (mass aggregation) read 0.81. Fixed in
  `6c27a3b` (mass-weighted aggregation). The honest v4 verdict against the
  accent-agnostic bar (one threshold for every learner): held-out pooled AUC
  **0.679** (l2arctic errors 0.648; per-L1 0.53–0.79), best operating point
  precision 1.0 at recall **0.0017** — the bar (**precision ≥ 0.90 at recall
  ≥ 0.4**) is not met. Per policy no calibration was committed and no
  verdicts ship; the engine stays retry-only. Evidence: eval/reports-v4/.
  **Accept criterion NOT met — recorded, not waived.**

  **Round two — the classification head + 10-dim DSP fusion, 10 controlled
  iterations (v13–v23, one variable per run).** Switched to a 4-class
  cross-entropy head over charsiu with dual mean+max pooling and ten
  hand-designed DSP cues (sibilance prominence, burst contrast, voicing
  continuity…), plus the expert-bracketed so762 errors (52 clean rows), mined
  hard negatives (top-500 correct tokens the judge misreads, Mandarin-only
  rounds), error-segment augmentation, and a label-smoothing sweep. Every
  exam ran the accent-agnostic pooled bar on the same held-out pool. The
  series plateaued at **AUC 0.83–0.90** (best: v22, Mandarin mining +
  smoothing 0.03 + augmentation, AUC **0.902**), with **P ≈ 0.74 at recall
  0.40** and ≤ 2% recall at precision 0.90 — the bar needs AUC ≈ 0.94. Two
  walls, measured precisely: (1) expert-judged *correct* Mandarin /ð/ that
  the acoustics genuinely render sibilant/stopped — the top of the score
  range interleaves them with true errors, so no single accent-blind
  threshold separates them; (2) compressed error confidence. A window-pad
  experiment (0.10 → 0.05 s) ruled out coarticulation bleed as the cause.
  **No run met the bar; no calibration was committed.** Per the agreed
  10-iteration cap the line stops here, the engine stays retry-only, and the
  next jump (if ever attempted) needs a stronger representation, not another
  recipe: ~0.94 AUC, i.e. roughly double the remaining separation gap.
  Evidence: eval/reports-v13/ … eval/reports-v23-final/. **Accept criterion
  NOT met — recorded, not waived.**
  ("please." / "please?" / "please!"). Accept: ≥ 90% fall-vs-rise on a
  purpose-recorded ~100-utterance set; octave errors < 2% of voiced frames.
  ~~Plus the packaging spike~~ — **done early**: `just package` produces a
  working unsigned onedir bundle (~590 MB, torch ~400 MB of it), engine serving
  the built SPA at one origin. Notes and traps in
  [engine/packaging/README.md](../engine/packaging/README.md). Remaining M4 work
  is the Tauri shell, signing, and notarization — not the freeze itself.
- **M3 — annotated spectrograms.** Engine `annotations[]` over client STFT.
  Accept: VOT within ±10 ms of manual Praat on ≥ 80% of ~50 hand-measured
  utterance-initial voiceless stops; unreliable tokens gated out.
- **M4 — desktop packaging.** Tauri + PyInstaller-onedir sidecar,
  health-poll lifecycle, first-run download UI. Accept: signed macOS +
  Windows installers; fully offline after first run; cold start < 5 s.

## 7. Evaluation

See [eval/README.md](../eval/README.md) for datasets, procedure, and the
**shipping bar** — the policy that governs whether a feedback type may ship at
all. It is the authority; everywhere else in this repo cites it.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Model accuracy on strong accents, child voices, laptop mics | Confidence gating; retry over wrong verdicts; per-L1 fairness audit in the eval report (informational, never a gate); closed-set scoring |
| CTC peakiness undermines interval GOP | Measured twice: the M1 bake-off (all aggregations near chance on /ð/, AUC ≤ 0.59) and the Option 3 judge, where blank-dominant CTC outputs plus per-frame renormalization collapsed the exam to chance until mass-weighted aggregation shipped (6c27a3b); see eval/reports/ and eval/reports-v4/ |
| Phone-set mapping bugs silently corrupt verdicts | One canonical IPA inventory; per-model tables; round-trip tests |
| PyInstaller + torch bundle (~0.5–1 GB, notarization, SmartScreen) | **Spike done**: 590 MB unsigned bundle works. Signing/notarization still M4; ONNX Runtime fallback still pre-planned if size bites |
| Formant tracking unreliable (high F0 voices) | Supporting evidence only; own reliability gate; never the basis of a vowel verdict |
| Automatic VOT is hard | M3 scope: utterance-initial voiceless stops only; gate out unclear tokens |
| Browser audio DSP / sample-rate quirks | Constraints disabled; engine-side resampling; `audio.quality` reports levels for UI advice. Note that disabling `autoGainControl` is what *causes* the low raw-capture levels — so level must never gate analysis (see below) |
| First-run model download (0.3–1.2 GB) | Pinned manifest with sizes; resumable; NDJSON progress; app browsable while downloading |
| F0 octave jumps, unvoiced gaps, duration mismatch | Cleanup pass; DTW on voiced frames of time-normalized contours; semitone domain |
| Port conflicts / firewall prompts | Bind 127.0.0.1 only; auto-increment + discovery via stdout and /v1/health |

## 9. Carried out of M0

Deliberate gaps, each with the milestone that closes it. None of them are
silent: every one either fails loud or is visible in the contract.

| Gap | Why it was left | Closes in |
|---|---|---|
| **VAD is energy-based**, not silero | Scripted single-word drills recorded deliberately; costs no extra download. Uses hysteresis — a single threshold clipped the /s/ off "this", since voiceless fricatives sit 20–30 dB under the neighbouring vowel | ✅ M1 — silero behind detect_speech, energy fallback for engines without the ml extra |
| **GOP's denominator spans the whole multilingual vocabulary** | The espeak model is multilingual, so Mandarin tone-tagged vowels compete with English phones and inflate GOP magnitude for reasons unrelated to pronunciation. Closed-set contrast scoring renormalises over {target} ∪ confusions and is the designed fix | ✅ M1 — scoring/contrast.py renormalizes over {target} ∪ confusions (the bake-off showed the signal is still too weak — see eval/reports/) |
| **`words[]` holds one word spanning the utterance** | The exercise schema carries a flat phone list with no word boundaries. Correct for the word and minimal-pair drills that exist; degenerate for sentences, of which there are none | Content-schema change, before sentence drills |
| **`Phone.score` is always null** | Mapping GOP to a 0–1 "how good was it" needs committed calibration, which needs the eval harness. An uncalibrated number in that field would read as a verdict | ✅ M1 — calibrated when calibration.yaml exists; absent until a contrast passes the bar |
| **Alignment thresholds are placeholders** | `min_alignment_confidence` / `low_alignment_confidence` in `config.py` gate "could this be analysed", not pronunciation. Real numbers come from eval | ✅ M1 — the values live in calibration.yaml; the negative bake-off kept the M0 placeholders (see eval/reports/) |
| **No reference recordings exist** | The teacher audio is unrecorded, so packs declare paths that do not resolve. The loader warns rather than failing, `has_reference_audio` reports it, and the UI hides playback | When the pack author records them — and picks their licence, still an open question (proposal: **CC BY-SA 4.0**) |
| **`?include_ungated=true` is not implemented** | There is nothing ungated to reveal yet — no contrasts are computed | ✅ M1 — implemented; the harness consumes it on every token run |
