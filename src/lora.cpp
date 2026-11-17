// Port of mlx_lm tuner (lora.py LoRALinear, trainer.py default_loss and the
// train step) and mlx.optimizers.Adam to the MLX C++ API.
#include "lora.h"
#include <algorithm>

#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace mlxnode {

using mx::array;

void Lora::init(const LoraOptions& opts) {
  remove();
  targets_ = model_.lora_targets(opts.num_layers, opts.keys);
  for (auto& [path, lin] : targets_) {
    int in = lin->input_dims();
    int out = lin->weight.shape(0);
    float s = 1.0f / std::sqrt(static_cast<float>(in));
    lin->lora_a = mx::random::uniform(array(-s), array(s), {in, opts.rank}, mx::float32);
    lin->lora_b = mx::zeros({opts.rank, out}, mx::float32);
    lin->lora_scale = opts.scale;
  }
  auto p = params();
  mx::eval(p);
  m_.clear();
  v_.clear();
}

void Lora::load(const std::string& file, float scale) {
  remove();
  auto [weights, meta] = mx::load_safetensors(file);
  for (auto& [path, lin] : model_.lora_targets(-1, {})) {
    auto a = weights.find(path + ".lora_a");
    auto b = weights.find(path + ".lora_b");
    if (a == weights.end() || b == weights.end()) continue;
    lin->lora_a = a->second;
    lin->lora_b = b->second;
    lin->lora_scale = scale;
    targets_.emplace_back(path, lin);
  }
  if (targets_.empty()) throw std::runtime_error("no LoRA weights in " + file);
  auto p = params();
  mx::eval(p);
  m_.clear();
  v_.clear();
}

void Lora::save(const std::string& file) const {
  std::unordered_map<std::string, array> out;
  for (auto& [path, lin] : targets_) {
    out.insert_or_assign(path + ".lora_a", *lin->lora_a);
    out.insert_or_assign(path + ".lora_b", *lin->lora_b);
  }
  mx::save_safetensors(file, out, {{"format", "mlx"}});
}

void Lora::remove() {
  model_.clear_lora();
  targets_.clear();
  m_.clear();
  v_.clear();
}

std::vector<array> Lora::params() const {
  std::vector<array> p;
  for (auto& [path, lin] : targets_) p.insert(p.end(), {*lin->lora_a, *lin->lora_b});
  return p;
}

void Lora::assign(const std::vector<array>& p) {
  for (size_t i = 0; i < targets_.size(); ++i) {
    targets_[i].second->lora_a = p[2 * i];
    targets_[i].second->lora_b = p[2 * i + 1];
  }
}

array Lora::masked_ce(const array& batch, int offset, int length, bool checkpoint,
                      array* ntoks) {
  int T = batch.shape(1) - 1;
  auto inputs = mx::slice(batch, {0, 0}, {1, T});
  auto targets = mx::slice(batch, {0, 1}, {1, T + 1});
  // nn.losses.cross_entropy (mlx 0.32.4): on the GPU the fused kernel, which
  // accumulates in float32; elsewhere the plain path, in the logits dtype.
  auto cross_entropy = [](array logits, const array& targets) {
    if (mx::default_device() == mx::Device::gpu)
      return mx::astype(mx::fast::cross_entropy(logits, targets), logits.dtype());
    logits = mx::subtract(logits, mx::stop_gradient(mx::max(logits, -1, true)));
    auto score = mx::squeeze(mx::take_along_axis(logits, mx::expand_dims(targets, -1), -1), -1);
    return mx::subtract(mx::logsumexp(logits, -1), score);
  };
  array ce(0.0f);
  if (checkpoint && loss_chunk_ > 0 && T > loss_chunk_) {
    // Chunked: each chunk's logits are made and dropped inside a checkpoint, so the backward pass
    // recomputes them chunk by chunk (a 2,048-token example: ~6 GB of logits and gradients → <1 GB).
    auto hidden = model_.hidden_train(inputs, true);
    std::vector<array> parts;
    for (int start = 0; start < T; start += loss_chunk_) {
      int stop = std::min(T, start + loss_chunk_);
      auto h = mx::slice(hidden, {0, start, 0}, {1, stop, hidden.shape(2)});
      // The targets are captured, not inputs: nothing is differentiated with respect to them.
      auto t = mx::stop_gradient(mx::slice(targets, {0, start}, {1, stop}));
      auto chunk_ce = mx::checkpoint([this, cross_entropy, t](const std::vector<array>& in) {
        return std::vector<array>{cross_entropy(model_.head(in[0]), t)};
      });
      parts.push_back(chunk_ce({h})[0]);
    }
    ce = mx::concatenate(parts, 1);
  } else {
    ce = cross_entropy(model_.forward_train(inputs, checkpoint), targets);
  }

  auto steps = mx::arange(1, T + 1, mx::int32);
  auto mask = mx::logical_and(mx::greater_equal(steps, array(offset)),
                              mx::less_equal(steps, array(length)));
  ce = mx::multiply(ce, mask);
  *ntoks = mx::sum(mask);
  return mx::divide(mx::sum(mx::astype(ce, mx::float32)), *ntoks);
}

std::pair<float, int> Lora::loss(const array& tokens, int offset, int length) {
  array ntoks(0);
  auto l = masked_ce(tokens, offset, length, false, &ntoks);
  mx::eval({l, ntoks});
  return {l.item<float>(), ntoks.item<int>()};
}

std::unordered_map<std::string, array> Lora::gradients(const array& tokens, int offset,
                                                      int length, bool checkpoint) {
  auto current = params();
  std::vector<int> argnums(current.size());
  for (size_t i = 0; i < argnums.size(); ++i) argnums[i] = static_cast<int>(i);
  auto fn = [&](const std::vector<array>& p) {
    assign(p);
    array ntoks(0);
    return std::vector<array>{masked_ce(tokens, offset, length, checkpoint, &ntoks), ntoks};
  };
  auto [values, grads] = mx::value_and_grad(fn, argnums)(current);
  assign(current);
  std::unordered_map<std::string, array> out;
  for (size_t i = 0; i < targets_.size(); ++i) {
    out.insert_or_assign(targets_[i].first + ".lora_a", grads[2 * i]);
    out.insert_or_assign(targets_[i].first + ".lora_b", grads[2 * i + 1]);
  }
  return out;
}

std::pair<float, int> Lora::step(const array& tokens, int offset, int length,
                                 float learning_rate, bool checkpoint) {
  if (targets_.empty()) throw std::runtime_error("no LoRA adapters: call init or load");
  auto current = params();
  std::vector<int> argnums(current.size());
  for (size_t i = 0; i < argnums.size(); ++i) argnums[i] = static_cast<int>(i);

  auto fn = [&](const std::vector<array>& p) {
    assign(p);
    array ntoks(0);
    auto l = masked_ce(tokens, offset, length, checkpoint, &ntoks);
    return std::vector<array>{l, ntoks};
  };
  auto [values, grads] = mx::value_and_grad(fn, argnums)(current);
  assign(current);
  // The gradients first, before the optimizer state and the weights change: if the GPU aborts this
  // (macOS stops long GPU work while the screen needs it, «Impacting Interactivity»), nothing has
  // changed and the same step can simply run again.
  {
    std::vector<array> pending = grads;
    pending.insert(pending.end(), values.begin(), values.end());
    mx::eval(pending);
  }

  // mlx.optimizers.Adam (bias_correction=False). 1 - beta is computed in
  // double, as Python does, so the update is bit-identical.
  if (m_.empty()) {
    for (auto& p : current) {
      m_.push_back(mx::zeros_like(p));
      v_.push_back(mx::zeros_like(p));
    }
  }
  std::vector<array> updated;
  for (size_t i = 0; i < current.size(); ++i) {
    auto& g = grads[i];
    auto lr = array(learning_rate, g.dtype());
    m_[i] = mx::add(mx::multiply(array(static_cast<float>(adam_.beta1)), m_[i]),
                    mx::multiply(array(static_cast<float>(1.0 - adam_.beta1)), g));
    v_[i] = mx::add(mx::multiply(array(static_cast<float>(adam_.beta2)), v_[i]),
                    mx::multiply(array(static_cast<float>(1.0 - adam_.beta2)), mx::square(g)));
    updated.push_back(mx::subtract(
        current[i], mx::divide(mx::multiply(lr, m_[i]),
                               mx::add(mx::sqrt(v_[i]), array(adam_.eps)))));
  }
  assign(updated);
  std::vector<array> to_eval = updated;
  to_eval.insert(to_eval.end(), m_.begin(), m_.end());
  to_eval.insert(to_eval.end(), v_.begin(), v_.end());
  to_eval.insert(to_eval.end(), values.begin(), values.end());
  mx::eval(to_eval);
  return {values[0].item<float>(), values[1].item<int>()};
}

}  // namespace mlxnode
