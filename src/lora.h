// LoRA adapters for Qwen35, compatible with mlx_lm (same parameter names,
// adapters.safetensors layout, loss and Adam optimizer).
#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qwen35.h"

namespace mlxnode {

struct LoraOptions {
  int num_layers = 16;
  int rank = 8;
  float scale = 20.0f;
  std::vector<std::string> keys;  // empty: every linear layer (mlx_lm default)
};

struct AdamOptions {
  double beta1 = 0.9;
  double beta2 = 0.999;
  float eps = 1e-8f;
};

class Lora {
 public:
  explicit Lora(Qwen35& model) : model_(model) {}

  // New adapters with mlx_lm's initialization (a uniform, b zeros).
  void init(const LoraOptions& opts);
  // Loads adapters.safetensors (parameter names from mlx_lm).
  void load(const std::string& file, float scale);
  void save(const std::string& file) const;
  void remove();
  bool active() const { return !targets_.empty(); }

  // Masked cross entropy as mlx_lm default_loss for one sequence.
  // tokens: [1, T] (padded like mlx_lm); targets counted from offset to length.
  std::pair<float, int> loss(const mx::array& tokens, int offset, int length);
  // Gradients of the loss, by parameter name (for checks against mlx_lm).
  std::unordered_map<std::string, mx::array> gradients(const mx::array& tokens, int offset, int length, bool checkpoint);
  // One training step: gradients + Adam update. Returns (loss, tokens).
  std::pair<float, int> step(const mx::array& tokens, int offset, int length,
                             float learning_rate, bool checkpoint);
  // With checkpointing, the cross entropy is computed over this many positions at a time, each chunk
  // recomputed in the backward pass: the logits over the whole vocabulary (248,320 for Qwen 3.5) are never
  // all in memory. 0 turns it off.
  void set_loss_chunk(int positions) { loss_chunk_ = positions; }

 private:
  std::vector<mx::array> params() const;
  void assign(const std::vector<mx::array>& p);
  mx::array masked_ce(const mx::array& tokens, int offset, int length,
                      bool checkpoint, mx::array* ntoks);

  Qwen35& model_;
  std::vector<std::pair<std::string, Linear*>> targets_;
  std::vector<mx::array> m_, v_;
  AdamOptions adam_;
  int loss_chunk_ = 256;
};

}  // namespace mlxnode
