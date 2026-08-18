/* Generated from schemas/*.schema.json — DO NOT EDIT.
 * Regenerate with `just schema` (or `npm run gen:types`).
 * Sources: analysis_result.v1.schema.json, exercise_catalog.v1.schema.json, exercise_detail.v1.schema.json, health.v1.schema.json, model_catalog.v1.schema.json */

export interface AnalysisResult {
  schema_version: "1.0";
  engine_version: string;
  exercise_id: string;
  audio: AudioInfo;
  alignment: Alignment;
  contrasts?: ContrastResult[];
  prosody?: Prosody | null;
  annotations?: Annotation[];
  feedback?: FeedbackItem[];
}
export interface AudioInfo {
  duration_s: number;
  sample_rate: number;
  /**
   * Post-VAD speech region; null when no speech was detected.
   */
  speech_interval_s?: [number, number] | null;
  quality: AudioQuality;
}
export interface AudioQuality {
  clipping: boolean;
  snr_db_est?: number | null;
  /**
   * The input is effectively dead — no usable signal at all. NOT merely a low recording level: absolute level says nothing about whether speech can be analysed, so a quiet-but-clean recording is analysed normally and the UI advises from `speech_level_dbfs` instead of refusing.
   */
  too_quiet: boolean;
  /**
   * RMS of the detected speech region, dB relative to full scale.
   */
  speech_level_dbfs?: number | null;
  /**
   * Peak sample level of the whole recording, dBFS.
   */
  peak_dbfs?: number | null;
}
export interface Alignment {
  status: "ok" | "low_confidence" | "failed";
  confidence: number;
  words?: Word[];
  phones?: Phone[];
  /**
   * Why the analysis refused (no speech, clipping, missing model, ...). Set only when status=failed; the composer turns it into a retry message.
   */
  reason?: string | null;
}
export interface Word {
  text: string;
  start_s: number;
  end_s: number;
  phone_indices: number[];
}
export interface Phone {
  index: number;
  /**
   * Canonical IPA label from the engine's internal inventory.
   */
  label: string;
  start_s: number;
  end_s: number;
  /**
   * Raw goodness-of-pronunciation (log-posterior based).
   */
  gop?: number | null;
  /**
   * GOP mapped to [0,1] via committed calibration.
   */
  score?: number | null;
  confidence: number;
}
/**
 * Closed-set discrimination for one focus phone of the exercise.
 */
export interface ContrastResult {
  phone_index: number;
  target: string;
  confusion_set: string[];
  /**
   * Posterior mass renormalized over {target} ∪ confusion_set.
   */
  posteriors: {
    [k: string]: number;
  };
  verdict: "on_target" | "substituted" | "uncertain";
  /**
   * The confusion phone heard; set only when verdict=substituted.
   */
  detected?: string | null;
  /**
   * Calibrated (Platt-scaled), not raw margin.
   */
  confidence: number;
  /**
   * Log-ratio on the single frame most favouring a confusion.
   */
  spike_score?: number | null;
  /**
   * Share of label frames where a confusion outvoted the target.
   */
  vote_fraction?: number | null;
}
export interface Prosody {
  f0: F0Track;
  /**
   * Teacher-reference contour, precomputed at pack load.
   */
  reference?: F0Track | null;
  /**
   * Per-frame-normalized DTW over voiced regions.
   */
  dtw_distance?: number | null;
  nuclear_tone?: NuclearTone | null;
}
export interface F0Track {
  hop_s: number;
  start_s: number;
  /**
   * F0 in semitones relative to the speaker's median; null = unvoiced frame.
   */
  semitones: (number | null)[];
  median_hz?: number | null;
}
export interface NuclearTone {
  detected: "fall" | "rise" | "fall_rise" | "level";
  expected?: ("fall" | "rise" | "fall_rise" | "level") | null;
  match?: boolean | null;
  confidence: number;
}
/**
 * Acoustic event for spectrogram overlay. New `type`s may be added in
 * minor schema versions; clients must ignore types they don't know.
 */
export interface Annotation {
  type: "vot" | "voicing" | "duration" | "formants";
  phone_index?: number | null;
  /**
   * @minItems 2
   * @maxItems 2
   */
  interval_s: [number, number];
  value?: number | null;
  unit: string;
  expected_range?: [number, number] | null;
  verdict?: ("in_range" | "outside_range") | null;
  confidence: number;
  /**
   * formants only
   */
  f1?: number | null;
  /**
   * formants only
   */
  f2?: number | null;
}
export interface FeedbackItem {
  id: string;
  /**
   * Open enum, grows per milestone. Current: segmental_substitution, retry. Planned: nuclear_tone_mismatch (M2), vot_out_of_range (M3).
   */
  kind: string;
  severity: "error" | "warning" | "praise";
  confidence: number;
  /**
   * Stable key for future i18n.
   */
  message_key: string;
  message: string;
  /**
   * What the UI highlights.
   */
  anchor?: Anchor | null;
  /**
   * Pointers into contrasts/annotations/prosody, e.g. {'contrast_index': 0}.
   */
  evidence?: {
    [k: string]: number;
  };
}
export interface Anchor {
  phone_index?: number | null;
  interval_s?: [number, number] | null;
}
export interface ExerciseCatalog {
  schema_version: "1.0";
  exercises?: ExerciseSummary[];
}
/**
 * Enough to render a picker without fetching every exercise.
 */
export interface ExerciseSummary {
  id: string;
  pack_id: string;
  type: "minimal_pair" | "word" | "sentence" | "intonation";
  title: string;
  text: string;
  ipa: string;
  /**
   * The drilled phone; null for intonation exercises.
   */
  focus_phone?: string | null;
}
/**
 * Everything the UI needs to present a drill before recording.
 */
export interface ExerciseDetail {
  schema_version: "1.0";
  id: string;
  pack_id: string;
  type: "minimal_pair" | "word" | "sentence" | "intonation";
  title: string;
  lang: string;
  text: string;
  ipa: string;
  phones: ExercisePhone[];
  pair_with?: string | null;
  prosody?: ExerciseProsody | null;
  learner_notes?: string | null;
  /**
   * False when the pack declares reference audio that has not been recorded yet — the UI hides playback rather than offering a broken control.
   */
  has_reference_audio: boolean;
}
export interface ExercisePhone {
  /**
   * Position in the target phone sequence; matches Phone.index.
   */
  index: number;
  /**
   * Canonical IPA label from the engine's internal inventory.
   */
  ph: string;
  focus?: boolean;
  confusions?: string[];
}
export interface ExerciseProsody {
  nuclear_syllable_index: number;
  expected_tone: "fall" | "rise" | "fall_rise" | "level";
}
export interface HealthResponse {
  status: "ok";
  engine_version: string;
  schema_version: "1.0";
  alignment_model: string;
  analysis_available: boolean;
  models?: ModelInfo[];
}
export interface ModelInfo {
  id: string;
  state: "ready" | "missing" | "downloading";
  repo_id: string;
  /**
   * Pinned upstream commit — never a branch name.
   */
  revision: string;
  download_bytes: number;
  license: string;
  note: string;
  /**
   * False when the engine's `ml` extra is not installed; weights alone are not enough.
   */
  runtime_available: boolean;
}
export interface ModelCatalog {
  schema_version: "1.0";
  models?: ModelInfo[];
}
