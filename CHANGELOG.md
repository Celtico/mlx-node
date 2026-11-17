# Changelog

## 0.1.2

- LoRA training of Qwen 3.5 models in much less memory:
  - `trainStep(..., { deltaChunk: 64 })` computes the Gated DeltaNet recurrence in chunkwise form (as flash-linear-attention), keeping one state per chunk instead of one per token. Gradients match the step-by-step loop to ~5e-6 in float32.
  - With `gradCheckpoint`, the cross-entropy goes in checkpointed chunks of 256 positions (`lossChunk`), so the vocabulary logits are never all in memory at once.
  - MiMo 9B on a 16 GB Mac: 1,536-token examples train in 11.1 GB (before, 512 did not fit).
- `deltaSegment`: the step-by-step recurrence in checkpointed segments (kept for comparison).

## 0.1.1

- Tagged but not released: the new tests were too strict for the CI runner's GPU.

## 0.1.0

- First release: Qwen 3.5 text models (MiMo) on MLX from Node.js, prompt cache, LoRA training.
