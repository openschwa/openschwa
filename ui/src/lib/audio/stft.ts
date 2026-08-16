// Client-side spectrogram computation (M0 basic, M3 annotated).
//
// Contract: the client renders the spectrogram itself over the Float32 buffer it
// already holds — the engine sends only annotations (time intervals, labels,
// verdicts) to overlay. The engine never renders pixels
// (docs/architecture.md, principle 1).
//
// The transform is an iterative radix-2 Cooley-Tukey FFT. A 3-second clip at
// 48 kHz is ~570 frames of 1024 points; the O(n log n) version is a few
// milliseconds, while a naive DFT would be several seconds of blocked main
// thread on every recording.

export interface Spectrogram {
  /** magnitudes[frame][bin] in dB, normalised so 0 dB is the loudest bin. */
  magnitudes: Float32Array[];
  /** Seconds per frame. */
  hopS: number;
  /** Time of the first frame's centre, in seconds. */
  startS: number;
  /** Hz of the highest bin retained. */
  maxFrequencyHz: number;
  binCount: number;
}

export interface StftOptions {
  fftSize?: number;
  hopSize?: number;
  /** Bins above this are dropped: speech detail lives well below Nyquist. */
  maxFrequencyHz?: number;
  /** Values this far below the peak clamp to the floor. */
  dynamicRangeDb?: number;
}

const DEFAULTS = {
  fftSize: 1024,
  hopSize: 256,
  maxFrequencyHz: 8000,
  dynamicRangeDb: 70,
} as const;

/** Periodic Hann window — the right choice for analysis (vs. the symmetric one). */
export function hannWindow(size: number): Float32Array {
  const window = new Float32Array(size);
  for (let i = 0; i < size; i += 1) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / size));
  }
  return window;
}

/**
 * In-place iterative radix-2 FFT. `real`/`imag` must have power-of-two length.
 */
export function fft(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  if (n <= 1) return;
  if ((n & (n - 1)) !== 0) throw new Error(`FFT size must be a power of two, got ${n}`);

  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }

  for (let length = 2; length <= n; length <<= 1) {
    const angle = (-2 * Math.PI) / length;
    const wReal = Math.cos(angle);
    const wImag = Math.sin(angle);
    for (let start = 0; start < n; start += length) {
      let curReal = 1;
      let curImag = 0;
      for (let k = 0; k < length / 2; k += 1) {
        const evenIndex = start + k;
        const oddIndex = evenIndex + length / 2;
        const oddReal = real[oddIndex] * curReal - imag[oddIndex] * curImag;
        const oddImag = real[oddIndex] * curImag + imag[oddIndex] * curReal;

        real[oddIndex] = real[evenIndex] - oddReal;
        imag[oddIndex] = imag[evenIndex] - oddImag;
        real[evenIndex] += oddReal;
        imag[evenIndex] += oddImag;

        const nextReal = curReal * wReal - curImag * wImag;
        curImag = curReal * wImag + curImag * wReal;
        curReal = nextReal;
      }
    }
  }
}

/** Short-time Fourier transform, returned as dB relative to the loudest bin. */
export function computeStft(
  samples: Float32Array,
  sampleRate: number,
  options: StftOptions = {},
): Spectrogram {
  const { fftSize, hopSize, maxFrequencyHz, dynamicRangeDb } = { ...DEFAULTS, ...options };
  const window = hannWindow(fftSize);
  const binCount = Math.min(
    fftSize / 2,
    Math.max(1, Math.ceil((maxFrequencyHz / sampleRate) * fftSize)),
  );

  const frameCount = samples.length >= fftSize ? 1 + Math.floor((samples.length - fftSize) / hopSize) : 0;
  const magnitudes: Float32Array[] = [];
  const real = new Float32Array(fftSize);
  const imag = new Float32Array(fftSize);
  let peakDb = -Infinity;

  for (let frame = 0; frame < frameCount; frame += 1) {
    const offset = frame * hopSize;
    for (let i = 0; i < fftSize; i += 1) real[i] = samples[offset + i] * window[i];
    imag.fill(0);
    fft(real, imag);

    const row = new Float32Array(binCount);
    for (let bin = 0; bin < binCount; bin += 1) {
      const power = real[bin] * real[bin] + imag[bin] * imag[bin];
      // 10*log10 of power, with a floor that keeps silence from becoming -Infinity.
      const db = 10 * Math.log10(Math.max(power, 1e-20));
      row[bin] = db;
      if (db > peakDb) peakDb = db;
    }
    magnitudes.push(row);
  }

  // Normalise to the clip's own peak so a quiet recording is still legible;
  // absolute levels are the engine's business, not the display's.
  for (const row of magnitudes) {
    for (let bin = 0; bin < row.length; bin += 1) {
      row[bin] = Math.max(row[bin] - peakDb, -dynamicRangeDb);
    }
  }

  return {
    magnitudes,
    hopS: hopSize / sampleRate,
    startS: fftSize / 2 / sampleRate,
    maxFrequencyHz: (binCount / fftSize) * sampleRate,
    binCount,
  };
}
