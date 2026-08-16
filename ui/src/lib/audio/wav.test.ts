// The engine parses these bytes directly (engine/audio/decode.py), so the
// header layout is a contract, not an implementation detail.
import { describe, expect, it } from 'vitest';
import { concatBlocks, encodeWav, normalizePeak } from './wav';

async function bytes(blob: Blob): Promise<DataView> {
  return new DataView(await blob.arrayBuffer());
}

function ascii(view: DataView, offset: number, length: number): string {
  return Array.from({ length }, (_, i) => String.fromCharCode(view.getUint8(offset + i))).join('');
}

describe('encodeWav', () => {
  it('writes a mono 16-bit PCM header the engine accepts', async () => {
    const view = await bytes(encodeWav(new Float32Array([0, 0.5, -0.5]), 48000));

    expect(ascii(view, 0, 4)).toBe('RIFF');
    expect(ascii(view, 8, 4)).toBe('WAVE');
    expect(ascii(view, 12, 4)).toBe('fmt ');
    expect(view.getUint16(20, true)).toBe(1); // PCM
    expect(view.getUint16(22, true)).toBe(1); // mono
    expect(view.getUint32(24, true)).toBe(48000);
    expect(view.getUint32(28, true)).toBe(48000 * 2); // byte rate
    expect(view.getUint16(32, true)).toBe(2); // block align
    expect(view.getUint16(34, true)).toBe(16);
    expect(ascii(view, 36, 4)).toBe('data');
    expect(view.getUint32(40, true)).toBe(6);
  });

  it('declares chunk sizes that match the payload', async () => {
    const blob = encodeWav(new Float32Array(1000), 16000);
    const view = await bytes(blob);
    expect(view.getUint32(4, true)).toBe(blob.size - 8);
    expect(view.getUint32(40, true)).toBe(2000);
  });

  it('scales samples to the full 16-bit range', async () => {
    const view = await bytes(encodeWav(new Float32Array([0, 1, -1]), 16000));
    expect(view.getInt16(44, true)).toBe(0);
    expect(view.getInt16(46, true)).toBe(32767);
    expect(view.getInt16(48, true)).toBe(-32767);
  });

  it('clamps out-of-range samples instead of letting them wrap', async () => {
    // Without clamping, 1.5 would wrap to a large negative value and be heard
    // as a click that the engine would measure as an artefact.
    const view = await bytes(encodeWav(new Float32Array([1.5, -1.5]), 16000));
    expect(view.getInt16(44, true)).toBe(32767);
    expect(view.getInt16(46, true)).toBe(-32767);
  });

  it('handles an empty recording without producing a malformed file', async () => {
    const blob = encodeWav(new Float32Array(0), 16000);
    expect(blob.size).toBe(44);
    expect((await bytes(blob)).getUint32(40, true)).toBe(0);
  });
});

describe('normalizePeak', () => {
  it('scales a quiet take up to the target peak', () => {
    // -53 dBFS-ish capture: what macOS's raw AGC-off path delivers at max gain.
    const samples = new Float32Array([0.002, -0.02, 0.01, 0.005]);
    const gainDb = normalizePeak(samples);
    expect(Math.max(...samples.map(Math.abs))).toBeCloseTo(0.95, 5);
    expect(gainDb).toBeCloseTo(20 * Math.log10(0.95 / 0.02), 3);
    // Constant gain: relative dynamics are exactly preserved.
    expect(samples[0] / samples[2]).toBeCloseTo(0.2, 5);
  });

  it('never scales down, so hardware clipping stays visible to the engine', () => {
    const clipped = new Float32Array([1.0, 1.0, -1.0, 0.4]);
    const before = [...clipped];
    expect(normalizePeak(clipped)).toBe(0);
    expect([...clipped]).toEqual(before);
  });

  it('leaves a dead input alone rather than amplifying its noise floor', () => {
    // Below -40 dBFS peak there is no real signal; boosting it would defeat
    // the engine's no-signal-from-the-microphone check.
    const dead = new Float32Array([0.0005, -0.0008, 0.0002]);
    const before = [...dead];
    expect(normalizePeak(dead)).toBe(0);
    expect([...dead]).toEqual(before);
  });

  it('handles an empty buffer', () => {
    expect(normalizePeak(new Float32Array(0))).toBe(0);
  });
});

describe('concatBlocks', () => {
  it('joins worklet blocks in order', () => {
    const merged = concatBlocks([new Float32Array([1, 2]), new Float32Array([3]), new Float32Array([4, 5])]);
    expect(Array.from(merged)).toEqual([1, 2, 3, 4, 5]);
  });

  it('returns an empty buffer for no blocks', () => {
    expect(concatBlocks([])).toHaveLength(0);
  });
});
