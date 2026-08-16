// FFT correctness matters beyond the picture: M3 draws engine annotations over
// these bins, so a wrong frequency axis would put a VOT box in the wrong place.
import { describe, expect, it } from 'vitest';
import { computeStft, fft, hannWindow } from './stft';

function sine(freq: number, sampleRate: number, length: number): Float32Array {
  const samples = new Float32Array(length);
  for (let i = 0; i < length; i += 1) samples[i] = Math.sin((2 * Math.PI * freq * i) / sampleRate);
  return samples;
}

describe('fft', () => {
  it('puts a pure tone in the bin matching its frequency', () => {
    const size = 64;
    const bin = 8;
    const real = new Float32Array(size);
    const imag = new Float32Array(size);
    for (let i = 0; i < size; i += 1) real[i] = Math.cos((2 * Math.PI * bin * i) / size);

    fft(real, imag);

    const magnitudes = Array.from({ length: size / 2 }, (_, k) =>
      Math.hypot(real[k], imag[k]),
    );
    const peak = magnitudes.indexOf(Math.max(...magnitudes));
    expect(peak).toBe(bin);
  });

  it('leaves a DC signal entirely in bin 0', () => {
    const real = new Float32Array(16).fill(1);
    const imag = new Float32Array(16);
    fft(real, imag);
    expect(real[0]).toBeCloseTo(16, 5);
    for (let k = 1; k < 8; k += 1) expect(Math.hypot(real[k], imag[k])).toBeCloseTo(0, 5);
  });

  it('rejects a non-power-of-two size rather than returning nonsense', () => {
    expect(() => fft(new Float32Array(6), new Float32Array(6))).toThrow(/power of two/);
  });
});

describe('hannWindow', () => {
  it('is periodic — zero at the start and never reaching zero again', () => {
    const window = hannWindow(8);
    expect(window[0]).toBeCloseTo(0, 10);
    expect(window[4]).toBeCloseTo(1, 10);
    expect(window[7]).toBeGreaterThan(0);
  });
});

describe('computeStft', () => {
  const sampleRate = 16000;

  it('places energy at the tone frequency', () => {
    const spectrogram = computeStft(sine(1000, sampleRate, 8192), sampleRate);
    const frame = spectrogram.magnitudes[4];
    const peakBin = frame.indexOf(Math.max(...frame));
    const peakHz = (peakBin / 1024) * sampleRate;
    expect(peakHz).toBeGreaterThan(900);
    expect(peakHz).toBeLessThan(1100);
  });

  it('reports a time axis the other views can align to', () => {
    const spectrogram = computeStft(sine(440, sampleRate, 16000), sampleRate, {
      fftSize: 1024,
      hopSize: 256,
    });
    expect(spectrogram.hopS).toBeCloseTo(256 / sampleRate, 10);
    expect(spectrogram.startS).toBeCloseTo(512 / sampleRate, 10);
    expect(spectrogram.magnitudes.length).toBe(1 + Math.floor((16000 - 1024) / 256));
  });

  it('normalises to the clip peak so quiet recordings stay legible', () => {
    const loud = computeStft(sine(1000, sampleRate, 4096), sampleRate);
    const quiet = computeStft(
      sine(1000, sampleRate, 4096).map((v) => v * 0.001) as Float32Array,
      sampleRate,
    );
    const peakOf = (s: typeof loud) => Math.max(...s.magnitudes.flatMap((r) => Array.from(r)));
    expect(peakOf(loud)).toBeCloseTo(0, 5);
    expect(peakOf(quiet)).toBeCloseTo(0, 5);
  });

  it('returns no frames for audio shorter than one window', () => {
    expect(computeStft(new Float32Array(100), sampleRate).magnitudes).toHaveLength(0);
  });

  it('keeps only bins below the requested ceiling', () => {
    const spectrogram = computeStft(sine(1000, sampleRate, 4096), sampleRate, {
      maxFrequencyHz: 4000,
    });
    expect(spectrogram.maxFrequencyHz).toBeLessThanOrEqual(4100);
    expect(spectrogram.binCount).toBeLessThan(512);
  });
});
