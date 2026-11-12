// Node: node tests/resample.test.mjs
import { resampleFrames } from '../assets/js/resample.js';
import assert from 'node:assert';

const DIMS = 150;
const mk = v => { const f = new Float32Array(DIMS); f.fill(v); return f; };

// too short -> null
assert.strictEqual(resampleFrames([mk(0), mk(0)], 48, DIMS), null);

// constant frames stay constant (minus the 0.5 shift)
{
  const out = resampleFrames(Array.from({ length: 30 }, () => mk(0.7)), 48, DIMS);
  assert.strictEqual(out.length, 48 * DIMS);
  for (const v of out) assert.ok(Math.abs(v - 0.2) < 1e-6, `got ${v}`);
}

// endpoints are preserved: first and last resampled frames equal source ends
{
  const frames = Array.from({ length: 10 }, (_, i) => mk(i / 10));
  const out = resampleFrames(frames, 48, DIMS);
  assert.ok(Math.abs(out[0] - (0.0 - 0.5)) < 1e-6);
  assert.ok(Math.abs(out[47 * DIMS] - (0.9 - 0.5)) < 1e-6);
}

// linear ramp interpolates monotonically
{
  const frames = Array.from({ length: 20 }, (_, i) => mk(i));
  const out = resampleFrames(frames, 48, DIMS);
  for (let t = 1; t < 48; t++) assert.ok(out[t * DIMS] >= out[(t - 1) * DIMS]);
}

console.log('resample tests: 4 passed');
