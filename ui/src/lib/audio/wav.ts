// 16-bit PCM WAV encoding.
//
// The client encodes its own WAV so the engine needs no ffmpeg and the audio
// never passes through a lossy codec (docs/architecture.md §3). This is the
// exact format engine/audio/decode.py expects.

const HEADER_BYTES = 44;
const BITS_PER_SAMPLE = 16;
const WAVE_FORMAT_PCM = 1;

/** Convert float samples in [-1, 1] to a mono 16-bit PCM WAV blob. */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(HEADER_BYTES + samples.length * 2);
  const view = new DataView(buffer);

  const writeAscii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };

  writeAscii(0, 'RIFF');
  view.setUint32(4, HEADER_BYTES - 8 + samples.length * 2, true);
  writeAscii(8, 'WAVE');
  writeAscii(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM fmt chunk length
  view.setUint16(20, WAVE_FORMAT_PCM, true);
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, BITS_PER_SAMPLE, true);
  writeAscii(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  for (let i = 0; i < samples.length; i += 1) {
    // Clamp before scaling: a sample above 1.0 would wrap to a large negative
    // 16-bit value and read as a click the engine would score as an artefact.
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(HEADER_BYTES + i * 2, Math.round(clamped * 32767), true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/** Peak levels below this mean "no real signal" — normalising would only
 * amplify a dead input's noise floor and defeat the engine's dead-mic check. */
const NORMALIZE_ELIGIBILITY_PEAK = 0.01; // -40 dBFS

const NORMALIZE_TARGET_PEAK = 0.95;

/**
 * Scale a whole take up to a healthy level before 16-bit encoding.
 *
 * macOS's raw capture path (which requesting `autoGainControl: false` selects)
 * tops out 20-30 dB below what consumer recorders deliver, even with the
 * system input slider at maximum — the user has no knob left to turn. A single
 * constant gain applied to the full take fixes that transparently: unlike the
 * browser AGC this project bans, it is not time-varying, so relative dynamics,
 * VOT, aspiration ratios, formants, and F0 are untouched. It also spends the
 * 16-bit quantisation budget on signal instead of headroom.
 *
 * Never scales down: a hardware-clipped take keeps its full-scale samples so
 * the engine's clipping detector still sees them. Inputs with no real signal
 * are left alone so a dead microphone still reads as dead.
 *
 * Returns the gain applied, in dB (0 when untouched).
 */
export function normalizePeak(samples: Float32Array): number {
  let peak = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const magnitude = Math.abs(samples[i]);
    if (magnitude > peak) peak = magnitude;
  }
  if (peak < NORMALIZE_ELIGIBILITY_PEAK || peak >= NORMALIZE_TARGET_PEAK) return 0;

  const gain = NORMALIZE_TARGET_PEAK / peak;
  for (let i = 0; i < samples.length; i += 1) samples[i] *= gain;
  return 20 * Math.log10(gain);
}

/** Concatenate captured blocks into one contiguous buffer. */
export function concatBlocks(blocks: Float32Array[]): Float32Array {
  const total = blocks.reduce((sum, block) => sum + block.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const block of blocks) {
    merged.set(block, offset);
    offset += block.length;
  }
  return merged;
}
