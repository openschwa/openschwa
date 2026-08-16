// Microphone capture worklet. Lives in public/ so it is served verbatim in dev
// and copied to dist on build — an AudioWorklet module must not be rewritten by
// the bundler, since it runs in its own realm with no module graph.
//
// It does no processing whatsoever: it hands raw Float32 blocks to the main
// thread, which owns buffering, WAV encoding, and rendering. Anything clever
// here would be signal processing applied before the engine measures the signal.
class RecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel) {
      // The render quantum's buffer is recycled after this call, so the copy is
      // mandatory. Transferring it avoids a second copy in structured clone.
      const block = new Float32Array(channel);
      this.port.postMessage(block, [block.buffer]);
    }
    return true;
  }
}

registerProcessor('openschwa-recorder', RecorderProcessor);
