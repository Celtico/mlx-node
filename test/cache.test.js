// Prompt cache: reusing saved states must give exactly the tokens a full
// prefill gives (tiny model from make_tiny.py).
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const { NativeModel } = require('..');

const dir = process.env.TINY_MODEL;
const skip = !dir && 'TINY_MODEL not set';

// Deterministic "random" token ids below the tiny vocabulary.
function ids(n, seed) {
  const out = [];
  let x = seed;
  for (let i = 0; i < n; i++) {
    x = (x * 1103515245 + 12345) % 2147483648;
    out.push(2 + (x % 500));
  }
  return out;
}

function run(model, prompt, options = {}) {
  return [...model.generate(prompt, { maxTokens: 20, ...options })];
}

// The same prompt with nothing saved: the reference.
function fresh(model, prompt) {
  model.clearCache();
  return run(model, prompt, { cache: false });
}

test('a conversation reuses the cache and gives the same tokens', { skip }, () => {
  const model = new NativeModel(dir);
  model.eos = new Set();
  const turn1 = ids(600, 1);
  const answer1 = run(model, turn1);
  assert.strictEqual(model.lastPrompt.reused, 0);

  // Second message: the first prompt, its answer and something new.
  const turn2 = [...turn1, ...answer1, ...ids(50, 2)];
  const answer2 = run(model, turn2);
  assert.strictEqual(model.lastPrompt.reused, turn1.length + answer1.length);
  assert.deepStrictEqual(answer2, fresh(model, turn2));
  assert.deepStrictEqual(answer1, fresh(model, turn1));
});

test('a prompt that changes in the middle starts from a shorter state', { skip }, () => {
  const model = new NativeModel(dir);
  model.eos = new Set();
  const start = ids(300, 3);
  const a = [...start, ...ids(400, 4)];
  const b = [...start, ...ids(400, 5)];
  const c = [...start, ...ids(400, 6)];
  run(model, a);
  // b shares 300 tokens with a: nothing saved covers them yet, so b reads it
  // all and keeps a checkpoint at 300 (as the worker does).
  const gotB = run(model, b);
  assert.strictEqual(model.lastPrompt.reused, 0);
  const gotC = run(model, c);
  assert.strictEqual(model.lastPrompt.reused, 300);
  assert.deepStrictEqual(gotB, fresh(model, b));
  assert.deepStrictEqual(gotC, fresh(model, c));
});

test('changing the adapters drops the saved states', { skip: (!dir || !fs.existsSync(path.join(dir ?? '', 'adapters_init.safetensors'))) && 'LoRA reference not made' }, () => {
  const model = new NativeModel(dir);
  model.eos = new Set();
  const prompt = ids(400, 7);
  run(model, prompt);
  model.engine.loraLoad(path.join(dir, 'adapters_init.safetensors'), 20);
  const got = run(model, [...prompt, ...ids(10, 8)]);
  assert.strictEqual(model.lastPrompt.reused, 0);
  assert.strictEqual(model.memory().promptEntries, 1);  // only the checkpoint just made
  assert.deepStrictEqual(got, fresh(model, [...prompt, ...ids(10, 8)]));
});

test('the checkpoint of the shared start outlives the answers, as in mlx_lm', { skip }, () => {
  const model = new NativeModel(dir);
  model.eos = new Set();
  const start = ids(300, 9);
  run(model, [...start, ...ids(100, 10)], { maxTokens: 3 });
  // A prompt that changes after the shared start keeps a checkpoint there («system»).
  let prompt = [...start, ...ids(100, 11)];
  let answer = run(model, prompt, { maxTokens: 3 });
  // Many turns of a conversation: each leaves one more state after its answer («assistant»).
  for (let turn = 0; turn < 10; turn++) {
    prompt = [...prompt, ...answer, ...ids(20, 20 + turn)];
    answer = run(model, prompt, { maxTokens: 3 });
    assert.ok(model.lastPrompt.reused > 300);
  }
  assert.ok(model.memory().promptEntries <= 8);
  // A new message whose instructions change after the shared start reuses that checkpoint.
  const next = [...start, ...ids(100, 40)];
  const got = run(model, next);
  assert.strictEqual(model.lastPrompt.reused, 300);
  // The same tokens as reading it without the cache in the same pieces (300, then the rest): the saved
  // state is intact after all those turns. (Other pieces can round differently on this random model.)
  model.clearCache();
  assert.deepStrictEqual(got, run(model, next, { cache: false, prefillStep: 300 }));
});
