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

// The loss in chunks (checkpointed, so the vocabulary logits are never all in memory) gives the same steps.
test('training steps with the loss in chunks match the unchunked ones', { skip }, () => {
  const ref = JSON.parse(fs.readFileSync(path.join(dir, 'reference_lora.json'), 'utf8'));
  const run = (lossChunk) => {
    const e = new NativeModel(dir).engine;
    e.loraLoad(path.join(dir, 'adapters_init.safetensors'), 20);
    const losses = ref.steps.map((s) => e.trainStep(Int32Array.from(s.tokens), s.offset, s.length, { learningRate: ref.lr, gradCheckpoint: true, lossChunk }).loss);
    const file = path.join(os.tmpdir(), `adapters-chunk-${process.pid}-${lossChunk}.safetensors`);
    e.loraSave(file);
    const adapters = readSafetensors(file);
    fs.rmSync(file);
    return { losses, adapters };
  };
  // Chunks of 16 positions: the 33-token reference sequences are split in two. (With chunks of 8 the
  // GPU picks another matmul kernel for so few rows, and bfloat16 rounding differs by ~1%; real chunks are 256.)
  const whole = run(0);
  const chunked = run(16);
  assert.ok(ref.steps.every((s) => s.tokens.length - 1 > 16), 'the reference sequences are longer than a chunk');
  assert.deepStrictEqual(chunked.losses, whole.losses);
  let worst = 0;
  for (const k of Object.keys(whole.adapters)) worst = Math.max(worst, maxDiff(whole.adapters[k].values, chunked.adapters[k].values));
  console.log(`losses ${chunked.losses.map((l) => l.toFixed(4)).join(', ')}; max adapter diff chunked vs whole: ${worst}`);
  assert.ok(worst < 1e-6, `adapters differ by ${worst}`);
});
