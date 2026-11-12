// Signet — inference: ONNX session, resampling, smoothing, emission state machine.
import * as ort from '../../vendor/ort/ort.wasm.min.mjs';

const T_FIXED = 48;
const DIMS = 150;

ort.env.wasm.wasmPaths = new URL('../../vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = 1;

export class Recognizer {
  constructor() {
    this.session = null;
    this.labels = [];
    this.smooth = null;       // EMA of probabilities
    this.streak = 0;          // consecutive ticks over threshold for the same word
    this.streakWord = -1;
    this.lastEmitWord = -1;
    this.lastEmitAt = 0;
    this.threshold = 0.55;
    this.tick = 0;
  }

  async load(modelUrl, labelsUrl) {
    this.labels = await (await fetch(labelsUrl)).json();
    this.session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
    });
  }

  // frames: array of Float32Array(150), raw coords, -2 = missing. Returns {probs, top} or null.
  resample(frames) {
    const n = frames.length;
    if (n < 8) return null;
    const out = new Float32Array(T_FIXED * DIMS);
    for (let t = 0; t < T_FIXED; t++) {
      const pos = (n - 1) * (t / (T_FIXED - 1));
      const i0 = Math.floor(pos);
      const i1 = Math.min(i0 + 1, n - 1);
      const w = pos - i0;
      const a = frames[i0], b = frames[i1];
      for (let d = 0; d < DIMS; d++) {
        out[t * DIMS + d] = (a[d] * (1 - w) + b[d] * w) - 0.5;
      }
    }
    return out;
  }

  async infer(frames) {
    const input = this.resample(frames);
    if (!input) return null;
    const tensor = new ort.Tensor('float32', input, [1, T_FIXED, DIMS]);
    const result = await this.session.run({ landmarks: tensor });
    const logits = result.logits.data;
    const probs = new Float32Array(logits.length);
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) { probs[i] = Math.exp(logits[i] - max); sum += probs[i]; }
    for (let i = 0; i < logits.length; i++) probs[i] /= sum;

    // temporal smoothing
    if (!this.smooth) this.smooth = probs.slice();
    else for (let i = 0; i < probs.length; i++) this.smooth[i] = 0.35 * this.smooth[i] + 0.65 * probs[i];

    // top-5
    const idx = [...this.smooth.keys()];
    idx.sort((a, b) => this.smooth[b] - this.smooth[a]);
    const top = idx.slice(0, 5).map(i => ({ word: this.labels[i], prob: this.smooth[i], index: i }));

    // emission state machine
    let emitted = null;
    const best = top[0];
    const now = performance.now();
    if (best.prob >= this.threshold) {
      if (best.index === this.streakWord) this.streak++;
      else { this.streakWord = best.index; this.streak = 1; }
      const cooled = best.index !== this.lastEmitWord || now - this.lastEmitAt > 2600;
      if (this.streak >= 3 && cooled) {
        emitted = best;
        this.lastEmitWord = best.index;
        this.lastEmitAt = now;
        this.streak = 0;
      }
    } else {
      this.streak = 0;
      this.streakWord = -1;
    }
    this.tick++;
    return { top, emitted };
  }

  reset() {
    this.smooth = null;
    this.streak = 0;
    this.streakWord = -1;
  }
}
