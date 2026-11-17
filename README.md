# @celtico/mlx-node

The native MLX engine for Node.js — no Python.

`@celtico/mlx-node` runs large language models with Apple's [MLX](https://github.com/ml-explore/mlx) directly from Node.js, through a native Node-API addon written in C++. It works with token ids: generation with streaming, greedy or sampled (temperature, top-p, top-k), a prompt cache like `mlx_lm`'s `LRUPromptCache`, LoRA adapters and LoRA training steps. Greedy output is token-for-token identical to `mlx_lm`.

**To chat with a model or train LoRA adapters, use [@celtico/node-mlx-lm](https://github.com/Celtico/node-mlx-lm)**, which adds the tokenizer, the chat templates and the training loop on top of this engine.

Supports **Qwen 3.5** models in mlx-community format (for example MiMo V2.6 9B, 4-bit) on **Apple Silicon Macs**. Linux (CUDA) and Windows are planned.

## Install

```sh
npm install @celtico/mlx-node
```

On install, it downloads the prebuilt addon for its version from this repository's GitHub release, and `libmlx.dylib` and `mlx.metallib` from the official `mlx-metal` package on PyPI, at the exact MLX version it was built with (no Python needed). Nothing is compiled.

## Use

```js
const { NativeModel } = require('@celtico/mlx-node');

const model = new NativeModel('/path/to/mlx-community/model');
for (const id of model.generate(promptIds, { maxTokens: 100, temperature: 0 })) { /* … */ }
console.log(model.lastPrompt); // { reused, total }: prompt tokens that came from the cache

model.loadAdapters('/path/to/adapters'); // adapters.safetensors + adapter_config.json, as `mlx_lm lora` writes them
model.close();                           // frees the model's memory
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

MIT. Parts are ported from [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Apple Inc.); see `NOTICE`.
