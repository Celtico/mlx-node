// Qwen3.5 text model (hybrid Gated DeltaNet + full attention) on MLX.
// Port of mlx_lm/models/qwen3_5.py (text only, batch size 1).
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "mlx/mlx.h"

namespace mlxnode {

namespace mx = mlx::core;

struct Quant {
  int group_size = 64;
  int bits = 4;
};

struct Config {
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_hidden_layers = 0;
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int head_dim = 0;
  int vocab_size = 0;
  float rms_norm_eps = 1e-6f;
  int linear_num_value_heads = 0;
  int linear_num_key_heads = 0;
  int linear_key_head_dim = 0;
  int linear_value_head_dim = 0;
  int linear_conv_kernel_dim = 4;
  int full_attention_interval = 4;
  bool tie_word_embeddings = false;
  float rope_theta = 1e7f;
  float partial_rotary_factor = 0.25f;
  // Quantization: default plus per-module overrides (module path -> quant).
  Quant quant;
  std::map<std::string, Quant> quant_overrides;
};

// Linear layer that is either quantized (weight/scales/biases) or dense.
struct Linear {
  mx::array weight{0.0f};
  std::optional<mx::array> scales;
  std::optional<mx::array> biases;
  Quant q;
  // Optional LoRA adapter (mlx_lm LoRALinear): y + scale * (x @ a) @ b.
  std::optional<mx::array> lora_a, lora_b;  // [in, r], [r, out]
  float lora_scale = 0.0f;
  mx::array operator()(const mx::array& x) const;
  int input_dims() const;
};

struct KVCache {
  std::optional<mx::array> keys, values;
  int offset = 0;
  // Appends new keys/values [1, H, L, D] and returns the filled views.
  std::pair<mx::array, mx::array> update(const mx::array& k, const mx::array& v);
};

struct LinearState {
  std::optional<mx::array> conv;  // [1, K-1, conv_dim]
  std::optional<mx::array> state; // [1, Hv, Dv, Dk] float32
};

// The whole cache at one point. A copy shares the arrays; MLX never changes an
// array that something else still holds (slice_update then copies), so a saved
// state stays as it was while the model goes on.
struct CacheState {
  std::vector<KVCache> kv;
  std::vector<LinearState> lin;
  int offset = 0;
  size_t nbytes() const;
};

struct Layer {
  bool is_linear = false;
  mx::array input_norm{0.0f}, post_norm{0.0f};
  Linear gate_proj, up_proj, down_proj;
  // Full attention
  Linear q_proj, k_proj, v_proj, o_proj;
  mx::array q_norm{0.0f}, k_norm{0.0f};
  // Gated DeltaNet
  Linear in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj;
  mx::array conv_weight{0.0f}, dt_bias{0.0f}, A_log{0.0f}, gnorm{0.0f};
};

class Qwen35 {
 public:
  Qwen35(const std::string& dir, const Config& cfg);

  // tokens: int32 [1, S]. Returns logits [1, S, vocab] (lazy) and updates the cache.
  mx::array forward(const mx::array& tokens);
  // Forward without cache, as in training (ops path for Gated DeltaNet so it
  // can be differentiated). checkpoint: recompute each layer in backward.
  mx::array forward_train(const mx::array& tokens, bool checkpoint);

  // Linear layers that can take a LoRA adapter, by weight-file path
  // (e.g. "language_model.model.layers.31.mlp.gate_proj"), in layer order.
  std::vector<std::pair<std::string, Linear*>> lora_targets(int num_layers, const std::vector<std::string>& keys);
  Linear* linear_by_path(const std::string& path);
  void clear_lora();
  void reset();
  int offset() const { return offset_; }
  CacheState save_cache() const { return {kv_, lin_, offset_}; }
  void load_cache(const CacheState& s) {
    kv_ = s.kv;
    lin_ = s.lin;
    offset_ = s.offset;
  }
  // All arrays that hold cache state, to evaluate them together.
  std::vector<mx::array> cache_arrays() const;
  const Config& config() const { return cfg_; }

 private:
  mx::array attention(Layer& l, KVCache* c, const mx::array& x);
  mx::array gated_delta_net(Layer& l, LinearState* c, const mx::array& x, bool kernel);
  mx::array layer_forward(Layer& l, size_t i, const mx::array& h, bool cached);
  mx::array mlp(const Layer& l, const mx::array& x) const;
  mx::array embed(const mx::array& tokens) const;

  Config cfg_;
  Linear embed_tokens_, lm_head_;
  mx::array final_norm_{0.0f};
  std::vector<Layer> layers_;
  std::vector<KVCache> kv_;
  std::vector<LinearState> lin_;
  int offset_ = 0;
};

// Gated delta rule recurrence (Metal kernel on GPU, ops loop elsewhere).
std::pair<mx::array, mx::array> gated_delta_update(
    const mx::array& q, const mx::array& k, const mx::array& v,
    const mx::array& a, const mx::array& b, const mx::array& A_log,
    const mx::array& dt_bias, const std::optional<mx::array>& state,
    bool allow_kernel = true);

}  // namespace mlxnode
