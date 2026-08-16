// Thin fetch client for the local engine. The base URL is the ONLY place the
// UI knows where the engine lives — nothing else may assume co-location
// (docs/architecture.md, principle 4).
import type {
  AnalysisResult,
  ExerciseCatalog,
  ExerciseDetail,
  ExerciseSummary,
  HealthResponse,
  ModelCatalog,
} from './types.gen';

/**
 * Where the engine lives.
 *
 * In dev the UI is served by Vite on another port, so it needs the engine's
 * address. In a build the engine serves these files itself, so the correct
 * answer is "same origin" — an empty base. Hard-coding the port in the build
 * would break the moment the engine auto-increments past a busy 8577, and
 * would break the hosted deployment entirely.
 */
export const ENGINE_URL: string =
  (import.meta.env?.VITE_ENGINE_URL as string | undefined) ??
  (import.meta.env?.DEV ? 'http://127.0.0.1:8577' : '');

export type {
  AnalysisResult,
  ExerciseCatalog,
  ExerciseDetail,
  ExerciseSummary,
  HealthResponse,
  ModelCatalog,
};

export class EngineError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/** Surface the engine's own `detail` message — it is written for a human. */
async function failure(response: Response, path: string): Promise<EngineError> {
  let detail = `HTTP ${response.status}`;
  try {
    const body = (await response.json()) as { detail?: string };
    if (body?.detail) detail = body.detail;
  } catch {
    // Non-JSON error body; the status line is all we have.
  }
  return new EngineError(`${path}: ${detail}`, response.status);
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${ENGINE_URL}${path}`);
  if (!response.ok) throw await failure(response, path);
  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>('/v1/health');
}

export function getExercises(): Promise<ExerciseCatalog> {
  return getJson<ExerciseCatalog>('/v1/exercises');
}

export function getExercise(id: string): Promise<ExerciseDetail> {
  return getJson<ExerciseDetail>(`/v1/exercises/${encodeURIComponent(id)}`);
}

export function getModels(): Promise<ModelCatalog> {
  return getJson<ModelCatalog>('/v1/models');
}

/** URL of the teacher reference WAV; only valid when `has_reference_audio`. */
export function referenceAudioUrl(id: string): string {
  return `${ENGINE_URL}/v1/exercises/${encodeURIComponent(id)}/reference-audio`;
}

/** Upload a 16-bit PCM WAV (client-encoded — see lib/audio/capture.ts) for scoring. */
export async function analyze(wav: Blob, exerciseId: string): Promise<AnalysisResult> {
  const form = new FormData();
  form.append('audio', wav, 'recording.wav');
  form.append('exercise_id', exerciseId);
  const response = await fetch(`${ENGINE_URL}/v1/analyze`, { method: 'POST', body: form });
  if (!response.ok) throw await failure(response, '/v1/analyze');
  return (await response.json()) as AnalysisResult;
}

export interface PullProgress {
  model_id: string;
  bytes_done?: number;
  bytes_total?: number;
  state?: string;
  done?: boolean;
  error?: string;
}

/**
 * Download model weights, yielding progress as the engine streams NDJSON.
 *
 * The first run pulls over a gigabyte, so this reports bytes rather than
 * leaving the user staring at a spinner (docs/architecture.md, risks).
 */
export async function* pullModel(modelId?: string): AsyncGenerator<PullProgress> {
  const query = modelId ? `?model_id=${encodeURIComponent(modelId)}` : '';
  const response = await fetch(`${ENGINE_URL}/v1/models/pull${query}`, { method: 'POST' });
  if (!response.ok) throw await failure(response, '/v1/models/pull');
  if (!response.body) throw new EngineError('/v1/models/pull: no response stream', 500);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffered = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffered += decoder.decode(value, { stream: true });

    // A chunk can split a line, so the trailing partial is carried forward.
    const lines = buffered.split('\n');
    buffered = lines.pop() ?? '';
    for (const line of lines) {
      if (line.trim()) yield JSON.parse(line) as PullProgress;
    }
  }
  if (buffered.trim()) yield JSON.parse(buffered) as PullProgress;
}
