// LoRA in the native engine vs mlx-lm on the tiny model.
// Make the references: python test/make_tiny.py <dir> && python test/make_lora_ref.py <dir>
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { NativeModel } = require('..');
const { readSafetensors } = require('./safetensors.js');

const dir = process.env.TINY_MODEL;
const skip = (!dir || !fs.existsSync(path.join(dir ?? '', 'reference_lora.json'))) && 'LoRA reference not made';

function maxDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

test('logits with loaded adapters match mlx-lm', { skip }, () => {
  const ref = JSON.parse(fs.readFileSync(path.join(dir, 'reference_lora.json'), 'utf8'));
  const e = new NativeModel(dir).engine;
  const plain = e.logits(Int32Array.from(ref.tokens));
  e.loraLoad(path.join(dir, 'adapters_init.safetensors'), 20);
  e.reset();
  const withLora = e.logits(Int32Array.from(ref.tokens));
  assert.ok(maxDiff(plain, ref.logits) > 0.01, 'adapters should change the output');
  assert.ok(maxDiff(withLora, ref.logits) < 1e-3, `diff ${maxDiff(withLora, ref.logits)}`);
});

for (const checkpoint of [false, true]) {
  test(`training steps match mlx-lm (gradCheckpoint ${checkpoint})`, { skip }, () => {
    const ref = JSON.parse(fs.readFileSync(path.join(dir, 'reference_lora.json'), 'utf8'));
    const e = new NativeModel(dir).engine;
    e.loraLoad(path.join(dir, 'adapters_init.safetensors'), 20);
    for (const s of ref.steps) {
      const r = e.trainStep(Int32Array.from(s.tokens), s.offset, s.length, { learningRate: ref.lr, gradCheckpoint: checkpoint });
      console.log(`loss ${r.loss.toFixed(5)} (mlx-lm ${s.loss.toFixed(5)}), tokens ${r.tokens}`);
      assert.strictEqual(r.tokens, s.ntoks);
      assert.ok(Math.abs(r.loss - s.loss) < 1e-3, `loss ${r.loss} vs ${s.loss}`);
    }
    const file = path.join(os.tmpdir(), `adapters-${process.pid}-${checkpoint}.safetensors`);
    e.loraSave(file);
    const got = readSafetensors(file);
    const want = readSafetensors(path.join(dir, 'adapters_final.safetensors'));
    assert.deepStrictEqual(Object.keys(got).sort(), Object.keys(want).sort());
    let worst = 0;
    for (const k of Object.keys(want)) {
      assert.deepStrictEqual(got[k].shape, want[k].shape, k);
      worst = Math.max(worst, maxDiff(got[k].values, want[k].values));
    }
    console.log(`max adapter diff after training: ${worst}`);
    assert.ok(worst < 1e-5, `adapters differ by ${worst}`);
    fs.rmSync(file);
  });
}

// The loss in chunks (checkpointed, so the vocabulary logits are never all in memory) gives the same gradients.
// Chunks of 16 positions: the 33-token reference sequences are split in two. Identical on an M2 Pro; another GPU
// may pick other matmul kernels for the chunks, and the tiny model is bfloat16, so the check is on the
// gradients' relative difference (as for the DeltaNet below), not on bit-equal adapters after Adam.
test('gradients with the loss in chunks match the unchunked ones', { skip }, () => {
  const ref = JSON.parse(fs.readFileSync(path.join(dir, 'reference_lora.json'), 'utf8'));
  assert.ok(ref.steps.every((s) => s.tokens.length - 1 > 16), 'the reference sequences are longer than a chunk');
  const s = ref.steps[0];
  const grads = (lossChunk) => {
    const e = new NativeModel(dir).engine;
    e.loraLoad(path.join(dir, 'adapters_init.safetensors'), 20);
    const file = path.join(os.tmpdir(), `grads-loss-${process.pid}-${lossChunk}.safetensors`);
    e.loraGradients(Int32Array.from(s.tokens), s.offset, s.length, file, true, lossChunk);
    const g = readSafetensors(file);
    fs.rmSync(file);
    return g;
  };
  const whole = grads(0);
  const chunked = grads(16);
  grads(256);  // the options are process-wide: back to the default
  let num = 0;
  let den = 0;
  for (const k of Object.keys(whole)) {
    const a = whole[k].values;
    const b = chunked[k].values;
    for (let i = 0; i < a.length; i++) { num += (a[i] - b[i]) ** 2; den += a[i] ** 2; }
  }
  const diff = Math.sqrt(num / den);
  console.log(`relative gradient difference, loss in chunks: ${diff.toExponential(2)}`);
  assert.ok(diff < 5e-2, `gradients differ by ${diff}`);
});

// The Gated DeltaNet recurrence in chunkwise form (deltaChunk, for long examples) computes the same
// gradients as mlx-lm's step-by-step loop. With bfloat16 activations (the tiny model as made) the rounding
// differs by ~1% (relative), as it does for the checkpointed segments; with the same model in float32
// (TINY_MODEL_F32: its weights cast to float32) the two agree to ~5e-6, so the difference is rounding.
const gradientDiff = (modelDir, segment, chunk) => {
  const ref = JSON.parse(fs.readFileSync(path.join(modelDir, 'reference_lora.json'), 'utf8'));
  const s = ref.steps[0];
  const grads = (seg, ch) => {
    const e = new NativeModel(modelDir).engine;
    e.loraLoad(path.join(modelDir, 'adapters_init.safetensors'), 20);
    const file = path.join(os.tmpdir(), `grads-${process.pid}-${seg}-${ch}.safetensors`);
    e.loraGradients(Int32Array.from(s.tokens), s.offset, s.length, file, true, 0, seg, ch);
    const g = readSafetensors(file);
    fs.rmSync(file);
    return g;
  };
  const base = grads(0, 0);
  const other = grads(segment, chunk);
  grads(0, 0);  // the options are process-wide: back to the defaults
  let num = 0;
  let den = 0;
  for (const k of Object.keys(base)) {
    const a = base[k].values;
    const b = other[k].values;
    for (let i = 0; i < a.length; i++) { num += (a[i] - b[i]) ** 2; den += a[i] ** 2; }
  }
  return Math.sqrt(num / den);
};

for (const deltaChunk of [8, 16, 64]) {
  test(`gradients with the DeltaNet in chunks of ${deltaChunk} match the step-by-step loop`, { skip }, () => {
    const diff = gradientDiff(dir, 0, deltaChunk);
    console.log(`relative gradient difference (bfloat16): ${diff.toExponential(2)}`);
    assert.ok(diff < 5e-2, `gradients differ by ${diff}`);  // bfloat16 rounding: ~1% on an M2 Pro
    const f32 = process.env.TINY_MODEL_F32;
    if (f32 && fs.existsSync(path.join(f32, 'reference_lora.json'))) {
      const exact = gradientDiff(f32, 0, deltaChunk);
      console.log(`relative gradient difference (float32): ${exact.toExponential(2)}`);
      assert.ok(exact < 1e-4, `float32 gradients differ by ${exact}`);
    }
  });
}
