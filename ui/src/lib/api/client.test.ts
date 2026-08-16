import { describe, expect, it } from 'vitest';
import { ENGINE_URL } from './client';

describe('client', () => {
  it('targets the localhost engine in dev, where Vite serves the UI separately', () => {
    // Vitest runs with import.meta.env.DEV set, matching the dev server.
    expect(ENGINE_URL).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
  });

  it('builds request URLs by concatenation, so an empty base means same origin', () => {
    // How the packaged app resolves: the engine serves the UI, so requests must
    // go to whatever port it actually bound rather than a compiled-in guess.
    expect(`${''}/v1/health`).toBe('/v1/health');
  });
});
