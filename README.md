# @celtico/mlx-node

Run and fine-tune MLX language models from Node.js — no Python.

`@celtico/mlx-node` runs large language models with Apple's [MLX](https://github.com/ml-explore/mlx) directly from Node.js, through a native Node-API addon written in C++. There is no Python process in between.

- **Generation** with streaming, greedy or sampled (temperature, top-p, top-k). Greedy output is token-for-token identical to `mlx_lm`.
- **Prompt cache** like `mlx_lm`'s `LRUPromptCache`: the next message of a conversation only reads what is new.
- **LoRA training** with the same parameters and adapter files as `mlx_lm lora`, which load in both.
- **Tokenizer and chat templates** in JavaScript (`@huggingface/tokenizers`, `@huggingface/jinja`).

Supports **Qwen 3.5** models in mlx-community format (for example MiMo V2.6 9B, 4-bit) on **Apple Silicon Macs**. Linux (CUDA) and Windows are planned.

## Install

```sh
npm install @celtico/mlx-node
```

On install, it downloads the prebuilt addon for its version from this repository's GitHub release, and `libmlx.dylib` and `mlx.metallib` from the official `mlx-metal` package on PyPI, at the exact MLX version it was built with (no Python needed). Nothing is compiled.

## Use

```js
const { ChatModel } = require('@celtico/mlx-node/chat');

const chat = new ChatModel('/path/to/mlx-community/model');
for (const piece of chat.stream([{ role: 'user', content: 'Write a JavaScript function that adds two numbers.' }], { maxTokens: 200 })) {
  process.stdout.write(piece);
}
chat.model.close(); // frees the model's memory
```

Lower level, with token ids:

```js
const { NativeModel } = require('@celtico/mlx-node');

const model = new NativeModel('/path/to/model');
for (const id of model.generate(promptIds, { maxTokens: 100, temperature: 0 })) { /* … */ }
console.log(model.lastPrompt); // { reused, total }: prompt tokens that came from the cache
```

LoRA adapters (`adapters.safetensors` + `adapter_config.json`, from `mlx_lm lora` or from here):

```js
model.loadAdapters('/path/to/adapters');

const { trainLora } = require('@celtico/mlx-node/lora');
await trainLora(chat, examples /* [{ messages: [...] }] */, {
  adapterPath: '/tmp/adapters', iters: 100, learningRate: 1e-5, numLayers: 8,
  maskPrompt: true, gradCheckpoint: true, onReport: (report) => console.log(report),
});
```

Calls are synchronous and use the GPU: run the engine in its own process (a `child_process` or an Electron `utilityProcess`) so your UI never waits on it.

## Build from source

Needs Xcode command line tools, `cmake` and MLX's Python package at the version in `package.json` (`"mlx"`), only to find its headers and library:

```sh
pip install mlx==0.32.4
npm install
npm run build        # MLX_PYTHON=/path/to/python if it is not python3
```

Tests compare the engine with `mlx_lm` on a tiny random model:

```sh
python test/make_tiny.py /tmp/tiny
python test/make_lora_ref.py /tmp/tiny
TINY_MODEL=/tmp/tiny npm test
```

## Releases

Pushing a tag `vX.Y.Z` runs `.github/workflows/release.yml`: it builds and tests the addon on an Apple Silicon runner and attaches it to the release. Then `npm publish --access public`.

## Licence

MIT. Parts are ported from [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Apple Inc.); see `LICENSE`.
