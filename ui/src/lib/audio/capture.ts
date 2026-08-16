// Microphone capture.
//
// Contract: AudioWorklet -> Float32 PCM at the device's native rate ->
// client-side 16-bit PCM WAV encoding for upload. Explicitly NOT
// MediaRecorder/opus: lossy compression destroys formant/VOT detail and
// server-side decoding would drag in ffmpeg.
//
// CRITICAL: browser "enhancement" DSP must be disabled — it removes exactly
// the acoustic cues (aspiration noise, voicing energy, F0 stability) the
// engine measures:
export const CAPTURE_CONSTRAINTS: MediaStreamConstraints = {
  audio: {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  },
};

const WORKLET_URL = '/recorder-worklet.js';
const PROCESSOR_NAME = 'openschwa-recorder';

export interface Recording {
  /** Float32 samples kept client-side for spectrogram/waveform rendering.
   * Already peak-normalised — identical to what the WAV contains. */
  samples: Float32Array;
  sampleRate: number;
  /** 16-bit PCM WAV of the same samples, ready for POST /v1/analyze. */
  wav: Blob;
  durationS: number;
  /** Constant gain applied by peak normalisation, dB; 0 when none was needed. */
  normalizationGainDb: number;
}

export class CaptureError extends Error {}

/** Called with the RMS level (0-1) of each captured block while recording. */
export type LevelListener = (rms: number) => void;

import { concatBlocks, encodeWav, normalizePeak } from './wav';

/** Turn the browser's getUserMedia failures into something a learner can act on. */
function describe(error: unknown): string {
  const name = error instanceof Error ? error.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return 'Microphone access was blocked. Allow it in your browser and try again.';
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    return 'No microphone found. Connect one and reload.';
  }
  return `Could not start recording: ${error instanceof Error ? error.message : String(error)}`;
}

/**
 * One recording session: `start()`, then `stop()` for the captured audio.
 *
 * The instance holds an AudioContext and a live microphone stream, so callers
 * must `dispose()` when finished — a stream left open keeps the browser's
 * recording indicator lit.
 */
export class MicRecorder {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private node: AudioWorkletNode | null = null;
  private blocks: Float32Array[] = [];
  private recording = false;
  private onLevel: LevelListener | null = null;

  /**
   * Watch the input level while recording.
   *
   * Worth putting on screen rather than burying: a muted or misrouted device is
   * otherwise indistinguishable from a working one until an analysis comes
   * back, and "speak louder" is useless advice when the capture chain is what
   * is wrong.
   */
  setLevelListener(listener: LevelListener | null): void {
    this.onLevel = listener;
  }

  get isRecording(): boolean {
    return this.recording;
  }

  /** Native device rate; 0 before the first `start()`. */
  get sampleRate(): number {
    return this.context?.sampleRate ?? 0;
  }

  /**
   * Create the audio context and load the worklet module ahead of time.
   *
   * Loading the module is a network fetch and costs the better part of a second
   * on a cold start. Paying it inside `start()` means the first word is spoken
   * before capture is running, so the app calls this at init and `start()`
   * becomes near-instant. Safe without a user gesture: the context is created
   * suspended and only resumed once the learner presses Record.
   */
  async prepare(): Promise<void> {
    if (this.context) return;
    // No sampleRate override: resampling in the browser would apply an unknown
    // filter before the engine ever sees the signal. The engine resamples to
    // 16 kHz itself, with soxr, where it can be reasoned about.
    const context = new AudioContext();
    await context.audioWorklet.addModule(WORKLET_URL);
    this.context = context;
  }

  async start(): Promise<void> {
    if (this.recording) return;
    this.blocks = [];

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new CaptureError(
          'This browser cannot record audio. A recent Chrome, Edge, Firefox, or Safari is needed.',
        );
      }
      this.stream ??= await navigator.mediaDevices.getUserMedia(CAPTURE_CONSTRAINTS);
      await this.prepare();
      const context = this.context!;
      if (context.state === 'suspended') await context.resume();

      // Arm first. Blocks are only accepted while armed, and everything from
      // here to the end of the method is synchronous, so no stale audio can
      // slip in ahead of the take (the original room-tone bug) and none of the
      // current take is dropped.
      this.recording = true;

      // The capture graph is built ONCE per microphone stream and reused for
      // every take; a retake only re-arms the flag.
      //
      // It used to be rebuilt on every start(), and that was a serious bug:
      // stop() disconnected the worklet's *outputs*, but the old source still
      // fed its input, so the old node stayed alive — and its message handler
      // kept appending into `blocks` whenever recording was armed. Every
      // retake added one more live producer, so take N stored each 128-sample
      // render quantum N times over. Audio in which every quantum repeats has
      // a period of exactly one quantum — at 48 kHz, 48000/128 = 375 Hz —
      // which the pitch tracker then reported as the speaker's median on
      // every retake, while alignment saw N-times-stretched stuttering it
      // could not line up with anything.
      if (!this.node) {
        this.source = context.createMediaStreamSource(this.stream);
        this.node = new AudioWorkletNode(context, PROCESSOR_NAME);
        this.node.port.onmessage = (event: MessageEvent<Float32Array>) => {
          if (!this.recording) return;
          this.blocks.push(event.data);
          if (this.onLevel) {
            let sum = 0;
            for (let i = 0; i < event.data.length; i += 1) sum += event.data[i] * event.data[i];
            this.onLevel(Math.sqrt(sum / event.data.length));
          }
        };

        // Some browsers only pull a node that reaches the destination, so the
        // chain terminates through a silent gain rather than being left
        // dangling.
        const silence = context.createGain();
        silence.gain.value = 0;
        this.source.connect(this.node).connect(silence).connect(context.destination);
      }
    } catch (error) {
      await this.dispose();
      throw error instanceof CaptureError ? error : new CaptureError(describe(error));
    }
  }

  /** Stop and return what was captured. The mic stays open for a fast retake. */
  async stop(): Promise<Recording> {
    if (!this.recording || !this.context) {
      throw new CaptureError('stop() called before start()');
    }
    this.recording = false;

    const sampleRate = this.context.sampleRate;
    const samples = concatBlocks(this.blocks);
    this.blocks = [];

    // The graph stays connected for the next take — the `recording` flag alone
    // gates capture. Tearing the node down per take is what caused the
    // producer-accumulation bug described in start(); only dispose() may
    // dismantle the graph.

    if (samples.length === 0) {
      throw new CaptureError('No audio was captured — check that the right microphone is selected.');
    }

    // Constant-gain normalisation before 16-bit encoding: raw AGC-off captures
    // sit far below full scale even at maximum system input gain, and the user
    // has no knob left to turn — see normalizePeak for why this is transparent
    // where browser AGC is not.
    const normalizationGainDb = normalizePeak(samples);

    return {
      samples,
      sampleRate,
      wav: encodeWav(samples, sampleRate),
      durationS: samples.length / sampleRate,
      normalizationGainDb,
    };
  }

  /** Release the microphone and audio context. */
  async dispose(): Promise<void> {
    this.recording = false;
    this.blocks = [];
    if (this.node) {
      this.node.port.onmessage = null;
      this.node.disconnect();
      this.node = null;
    }
    // The source's input edge must be broken explicitly — disconnecting the
    // worklet node does not release it, which is how the accumulation bug
    // survived stop() in the first place.
    this.source?.disconnect();
    this.source = null;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    if (this.context && this.context.state !== 'closed') await this.context.close();
    this.context = null;
  }
}
