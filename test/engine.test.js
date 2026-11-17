// Compares the native engine with mlx-lm on a tiny random Qwen3.5 model.
// Make the model first: python test/make_tiny.py <dir>, then
// TINY_MODEL=<dir> npm test
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const { NativeModel } = require('..');

const dir = process.env.TINY_MODEL;

test('logits match mlx-lm (prefill + steps)', { skip: !dir && 'TINY_MODEL not set' }, () => {
  const ref = JSON.parse(fs.readFileSync(path.join(dir, 'reference.json'), 'utf8'));
  const model = new NativeModel(dir);
  const e = model.engine;
  const got = [e.logits(Int32Array.from(ref.tokens.slice(0, ref.prefill)))];
  for (let t = ref.prefill; t < ref.tokens.length; t++) got.push(e.logits(Int32Array.of(ref.tokens[t])));
  got.forEach((row, i) => {
    const want = ref.logits[i];
    let maxDiff = 0;
    let scale = 0;
    for (let j = 0; j < want.length; j++) {
      maxDiff = Math.max(maxDiff, Math.abs(row[j] - want[j]));
      scale = Math.max(scale, Math.abs(want[j]));
    }
    const argmax = (a) => a.indexOf(Math.max(...a));
    console.log(`step ${i}: max diff ${maxDiff.toFixed(4)} (scale ${scale.toFixed(2)})`);
    assert.strictEqual(argmax(Array.from(row)), argmax(want), `argmax differs at step ${i}`);
    assert.ok(maxDiff <= 0.02 * scale + 0.05, `logits differ at step ${i}: ${maxDiff}`);
  });
});

test('greedy generation is deterministic', { skip: !dir && 'TINY_MODEL not set' }, () => {
  const model = new NativeModel(dir);
  const a = [...model.generate([5, 17, 300, 42], { maxTokens: 10 })];
  const b = [...model.generate([5, 17, 300, 42], { maxTokens: 10 })];
  assert.deepStrictEqual(a, b);
  assert.ok(a.length > 0);
});

test('greedy tokens match mlx-lm on a 700-token prompt', { skip: !dir && 'TINY_MODEL not set' }, () => {
  const ref = JSON.parse(fs.readFileSync(path.join(dir, 'reference_long.json'), 'utf8'));
  const model = new NativeModel(dir);
  model.eos = new Set();
  const got = [...model.generate(ref.prompt, { maxTokens: ref.greedy.length, prefillStep: 256 })];
  assert.deepStrictEqual(got, ref.greedy);
});
