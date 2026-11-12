// Signet — temporal resampling, shared by inference and tests. No dependencies.
export function resampleFrames(frames, T = 48, DIMS = 150) {
  const n = frames.length;
  if (n < 8) return null;
  const out = new Float32Array(T * DIMS);
  for (let t = 0; t < T; t++) {
    const pos = (n - 1) * (t / (T - 1));
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
