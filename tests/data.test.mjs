// Node: node tests/data.test.mjs  (run from repo root)
import assert from 'node:assert';
import { readFileSync } from 'node:fs';

const labels = JSON.parse(readFileSync('models/labels.json', 'utf8'));
assert.strictEqual(labels.length, 100);
assert.strictEqual(new Set(labels.map(w => w.toLowerCase())).size, 100, 'duplicate labels');

const meta = JSON.parse(readFileSync('models/meta.json', 'utf8'));
for (const k of ['params', 'val_top1', 'val_top5', 'test_top1', 'test_top5']) assert.ok(k in meta, k);
assert.ok(meta.test_top1 > 0 && meta.test_top1 < 1);
assert.ok(meta.test_top5 >= meta.test_top1);

const samples = JSON.parse(readFileSync('models/samples.json', 'utf8'));
assert.ok(samples.length >= 5);
for (const s of samples) {
  assert.ok(labels.includes(s.label), `unknown label ${s.label}`);
  assert.ok(s.video_id && s.split === 'test', 'sample missing provenance');
  assert.ok(Array.isArray(s.frames) && s.frames.length >= 8);
  for (const f of s.frames) assert.strictEqual(f.length, 150);
}
console.log(`data tests: passed (${labels.length} labels, ${samples.length} samples)`);
