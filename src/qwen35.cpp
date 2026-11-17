// Port of mlx_lm/models/qwen3_5.py, qwen3_next.py and gated_delta.py (MIT,
// Apple Inc.) to the MLX C++ API. Text only, batch size 1.
#include "qwen35.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>

namespace mlxnode {

namespace fs = std::filesystem;
using mx::array;

namespace {

// Scalar with the same dtype as x, so bf16 activations are not promoted.
array scalar(float v, const array& like) { return array(v, like.dtype()); }

// The element-wise functions mlx_lm compiles (shapeless), so they run as one
// fused kernel each, as there: nn.silu, activations.swiglu and precise_swiglu.
// Their inputs all have the same shape, which shapeless compiling needs.
array silu(const array& x) {
  static auto f = mx::compile(
      [](const std::vector<array>& in) {
        return std::vector<array>{mx::multiply(in[0], mx::sigmoid(in[0]))};
      },
      true);
  return f({x})[0];
}

array swiglu(const array& gate, const array& x) {
  static auto f = mx::compile(
      [](const std::vector<array>& in) {
        return std::vector<array>{mx::multiply(mx::multiply(in[0], mx::sigmoid(in[0])), in[1])};
      },
      true);
  return f({gate, x})[0];
}

// silu(gate) * x in float32, back in h's dtype (RMSNormGated).
array precise_swiglu(const array& h, const array& gate, const array& x) {
  static auto f = mx::compile(
      [](const std::vector<array>& in) {
        auto g = mx::astype(in[1], mx::float32);
        auto y = mx::multiply(mx::multiply(g, mx::sigmoid(g)), mx::astype(in[2], mx::float32));
        return std::vector<array>{mx::astype(y, in[0].dtype())};
      },
      true);
  return f({h, gate, x})[0];
}

// g = exp(-exp(A_log) * softplus(a + dt_bias)), in float32. Not compiled:
// mlx_lm compiles it shapeless, but here A_log and dt_bias [Hv] broadcast
// against a [B, S, Hv], and a shapeless compile traced at one S gave wrong
// values at another (the prompt cache test caught it). It is tiny anyway.
array compute_g(const array& A_log, const array& a, const array& dt_bias) {
  auto x = mx::add(a, dt_bias);
  auto softplus = mx::logaddexp(x, scalar(0.0f, x));
  return mx::exp(mx::negative(mx::multiply(mx::exp(mx::astype(A_log, mx::float32)), softplus)));
}

array take_weight(std::unordered_map<std::string, array>& w,
                  const std::string& name) {
  auto it = w.find(name);
  if (it == w.end()) throw std::runtime_error("missing weight: " + name);
  array a = it->second;
  w.erase(it);
  return a;
}

Linear take_linear(std::unordered_map<std::string, array>& w,
                   const Config& cfg, const std::string& path) {
  Linear l;
  l.weight = take_weight(w, path + ".weight");
  auto s = w.find(path + ".scales");
  if (s != w.end()) {
    l.scales = s->second;
    w.erase(s);
    auto b = w.find(path + ".biases");
    if (b != w.end()) {
      l.biases = b->second;
      w.erase(b);
    }
    auto o = cfg.quant_overrides.find(path);
    l.q = o != cfg.quant_overrides.end() ? o->second : cfg.quant;
  }
  return l;
}

// ---------------------------------------------------------------------------
// Gated delta kernel: copied verbatim from mlx_lm gated_delta.py
// (_make_gated_delta_packed_kernel). Needs Dk == 128 and Dv % 8 == 0.
const char* kPackedSource = R"(
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);

          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * gt;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);

          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }
)";

bool use_metal_kernel() {
  static bool ok = mx::metal::is_available() &&
                   mx::default_device() == mx::Device(mx::Device::gpu);
  return ok;
}

std::pair<array, array> gated_delta_kernel(const array& q, const array& k,
                                           const array& v, const array& g,
                                           const array& beta,
                                           const array& state) {
  static auto kernel = mx::fast::metal_kernel(
      "gated_delta_step_packed_btree",
      {"q", "k", "v", "g", "beta", "state_in", "T"}, {"y", "state_out"},
      kPackedSource);
  int B = k.shape(0), T = k.shape(1), Hk = k.shape(2), Dk = k.shape(3);
  int Hv = v.shape(2), Dv = v.shape(3);
  auto out = kernel(
      {q, k, v, g, beta, state, array(T)},
      {{B, T, Hv, Dv}, state.shape()}, {q.dtype(), state.dtype()},
      {32, Dv / 8, B * Hv}, {32, 2, 1},
      {{"InT", q.dtype()},
       {"StT", state.dtype()},
       {"Dk", Dk},
       {"Dv", Dv},
       {"Hk", Hk},
       {"Hv", Hv}},
      std::nullopt, false, {});
  return {out[0], out[1]};
}

// One recurrent step, compiled like mlx_lm's @mx.compile
// _gated_delta_step_ops so the fused Metal kernels (and the results) are the
// same. Inputs: q, k [B, H, Dk], v [B, H, Dv], g, beta [B, H],
// state [B, H, Dv, Dk]. Outputs: y [B, H, Dv], new state.
std::vector<array> gated_delta_step(const std::vector<array>& in) {
  const auto& q = in[0];
  const auto& k = in[1];
  const auto& v = in[2];
  const auto& g = in[3];
  const auto& beta = in[4];
  auto state = mx::multiply(in[5], mx::expand_dims(g, {-1, -2}));
  auto k_row = mx::expand_dims(k, -2);  // [B, H, 1, Dk]
  auto kv_mem = mx::sum(mx::multiply(state, k_row), -1);  // [B, H, Dv]
  auto delta = mx::multiply(mx::subtract(v, kv_mem), mx::expand_dims(beta, -1));
  state = mx::add(state, mx::multiply(k_row, mx::expand_dims(delta, -1)));
  auto y = mx::sum(mx::multiply(state, mx::expand_dims(q, -2)), -1);
  return {mx::astype(y, q.dtype()), state};
}

// Steps per checkpointed segment of the recurrence when it is longer (see gated_delta_ops); 0: none.
// Off by default: with segments the gradients differ from mlx_lm by ~1% (relative), the same for any
// segment size, and the cause is not understood yet. trainStep's { deltaSegment } turns it on.
int kDeltaSegment = 0;

// Ops reference (mlx_lm gated_delta_ops): sequential loop over time.
std::pair<array, array> gated_delta_ops(array q, array k, const array& v,
                                        const array& g, const array& beta,
                                        array state) {
  static auto step = mx::compile(gated_delta_step);
  int T = q.shape(1), Hk = q.shape(2), Hv = v.shape(2);
  if (Hv / Hk > 1) {
    q = mx::repeat(q, Hv / Hk, -2);
    k = mx::repeat(k, Hv / Hk, -2);
  }
  // Steps [from, to) of the recurrence from `state`: (outputs [B, to-from, H, Dv], last state).
  auto run = [](const array& q, const array& k, const array& v, const array& g, const array& beta,
                array state, int from, int to) {
    std::vector<array> ys;
    for (int t = from; t < to; ++t) {
      auto qt = mx::squeeze(mx::slice(q, {0, t, 0, 0}, {q.shape(0), t + 1, q.shape(2), q.shape(3)}), 1);
      auto kt = mx::squeeze(mx::slice(k, {0, t, 0, 0}, {k.shape(0), t + 1, k.shape(2), k.shape(3)}), 1);
      auto vt = mx::squeeze(mx::slice(v, {0, t, 0, 0}, {v.shape(0), t + 1, v.shape(2), v.shape(3)}), 1);
      auto gt = mx::squeeze(mx::slice(g, {0, t, 0}, {g.shape(0), t + 1, g.shape(2)}), 1);
      auto bt = mx::squeeze(mx::slice(beta, {0, t, 0}, {beta.shape(0), t + 1, beta.shape(2)}), 1);
      auto out = step({qt, kt, vt, gt, bt, state});
      ys.push_back(out[0]);
      state = out[1];
    }
    return std::make_pair(mx::stack(ys, 1), state);
  };
  if (kDeltaSegment <= 0 || T <= kDeltaSegment) return run(q, k, v, g, beta, state, 0, T);
  // Longer sequences (training): the steps go in checkpointed segments, so the backward pass keeps only
  // the state between segments and recomputes one segment at a time. Every step keeps several copies of
  // the [H, Dv, Dk] state (2 MB for Qwen 3.5 9B) for its gradients: a 512-token example needed ~5 GB per
  // layer, 4,096 tokens ~40 GB. The arithmetic is the same, so the results are too.
  std::vector<array> parts;
  for (int from = 0; from < T; from += kDeltaSegment) {
    int to = std::min(T, from + kDeltaSegment);
    auto slice_t = [&](const array& x) {
      mx::Shape start(x.ndim(), 0);
      mx::Shape stop = x.shape();
      start[1] = from;
      stop[1] = to;
      return mx::slice(x, start, stop);
    };
    auto segment = mx::checkpoint([run](const std::vector<array>& in) {
      int n = in[0].shape(1);
      auto [ys, last] = run(in[0], in[1], in[2], in[3], in[4], in[5], 0, n);
      return std::vector<array>{ys, last};
    });
    auto out = segment({slice_t(q), slice_t(k), slice_t(v), slice_t(g), slice_t(beta), state});
    parts.push_back(out[0]);
    state = out[1];
  }
  return {mx::concatenate(parts, 1), state};
}

}  // namespace

void set_delta_segment(int steps) { kDeltaSegment = steps; }

std::pair<array, array> gated_delta_update(
    const array& q, const array& k, const array& v, const array& a,
    const array& b, const array& A_log, const array& dt_bias,
    const std::optional<array>& state_in, bool allow_kernel) {
  auto beta = mx::sigmoid(b);
  auto g = compute_g(A_log, a, dt_bias);
  int B = q.shape(0), Dk = q.shape(3), Hv = v.shape(2), Dv = v.shape(3);
  array state = state_in ? *state_in : mx::zeros({B, Hv, Dv, Dk}, mx::float32);
  if (allow_kernel && use_metal_kernel() && Dk == 128 && Dv % 8 == 0) {
    return gated_delta_kernel(q, k, v, g, beta, state);
  }
  return gated_delta_ops(q, k, v, g, beta, state);
}

array Linear::operator()(const array& x) const {
  array y = scales ? mx::quantized_matmul(x, weight, *scales, biases, true,
                                          q.group_size, q.bits, "affine")
                   : mx::matmul(x, mx::transpose(weight));
  if (lora_a) {
    auto z = mx::matmul(mx::matmul(x, *lora_a), *lora_b);
    y = mx::add(y, mx::astype(mx::multiply(array(lora_scale, z.dtype()), z), x.dtype()));
  }
  return y;
}

int Linear::input_dims() const {
  return scales ? weight.shape(1) * 32 / q.bits : weight.shape(1);
}

std::pair<array, array> KVCache::update(const array& k, const array& v) {
  constexpr int step = 256;
  int L = k.shape(2);
  if (!keys || offset + L > keys->shape(2)) {
    int n = ((L + step - 1) / step) * step;
    auto kshape = k.shape();
    auto vshape = v.shape();
    kshape[2] = n;
    vshape[2] = n;
    auto nk = mx::zeros(kshape, k.dtype());
    auto nv = mx::zeros(vshape, v.dtype());
    if (keys) {
      auto ks = keys->shape(), vs = values->shape();
      ks[2] = offset;
      vs[2] = offset;
      keys = mx::concatenate({mx::slice(*keys, {0, 0, 0, 0}, ks), nk}, 2);
      values = mx::concatenate({mx::slice(*values, {0, 0, 0, 0}, vs), nv}, 2);
    } else {
      keys = nk;
      values = nv;
    }
  }
  auto ks = keys->shape(), vs = values->shape();
  keys = mx::slice_update(*keys, k, mx::Shape{0, 0, offset, 0}, mx::Shape{ks[0], ks[1], offset + L, ks[3]});
  values = mx::slice_update(*values, v, mx::Shape{0, 0, offset, 0}, mx::Shape{vs[0], vs[1], offset + L, vs[3]});
  offset += L;
  return {mx::slice(*keys, {0, 0, 0, 0}, {ks[0], ks[1], offset, ks[3]}),
          mx::slice(*values, {0, 0, 0, 0}, {vs[0], vs[1], offset, vs[3]})};
}

Qwen35::Qwen35(const std::string& dir, const Config& cfg) : cfg_(cfg) {
  std::unordered_map<std::string, array> w;
  for (auto& entry : fs::directory_iterator(dir)) {
    auto name = entry.path().filename().string();
    if (entry.path().extension() != ".safetensors" || name.rfind("model", 0) != 0)
      continue;
    auto [arrays, meta] = mx::load_safetensors(entry.path().string());
    for (auto& [k, a] : arrays) {
      // Keep only the text model; skip vision tower and MTP weights.
      if (k.rfind("language_model.", 0) != 0 || k.find("mtp.") != std::string::npos)
        continue;
      w.insert_or_assign(k, a);
    }
  }
  if (w.empty()) throw std::runtime_error("no language_model weights in " + dir);

  const std::string p = "language_model.model.";
  embed_tokens_ = take_linear(w, cfg_, p + "embed_tokens");
  final_norm_ = take_weight(w, p + "norm.weight");
  lm_head_ = cfg_.tie_word_embeddings ? embed_tokens_
                                      : take_linear(w, cfg_, "language_model.lm_head");
  for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
    Layer l;
    auto lp = p + "layers." + std::to_string(i) + ".";
    l.is_linear = (i + 1) % cfg_.full_attention_interval != 0;
    l.input_norm = take_weight(w, lp + "input_layernorm.weight");
    l.post_norm = take_weight(w, lp + "post_attention_layernorm.weight");
    l.gate_proj = take_linear(w, cfg_, lp + "mlp.gate_proj");
    l.up_proj = take_linear(w, cfg_, lp + "mlp.up_proj");
    l.down_proj = take_linear(w, cfg_, lp + "mlp.down_proj");
    if (l.is_linear) {
      auto ap = lp + "linear_attn.";
      l.in_proj_qkv = take_linear(w, cfg_, ap + "in_proj_qkv");
      l.in_proj_z = take_linear(w, cfg_, ap + "in_proj_z");
      l.in_proj_b = take_linear(w, cfg_, ap + "in_proj_b");
      l.in_proj_a = take_linear(w, cfg_, ap + "in_proj_a");
      l.out_proj = take_linear(w, cfg_, ap + "out_proj");
      l.conv_weight = take_weight(w, ap + "conv1d.weight");
      if (l.conv_weight.shape(-1) != 1)
        throw std::runtime_error("conv1d weight not in MLX layout (use an mlx-community model)");
      l.dt_bias = take_weight(w, ap + "dt_bias");
      l.A_log = take_weight(w, ap + "A_log");
      l.gnorm = take_weight(w, ap + "norm.weight");
    } else {
      auto ap = lp + "self_attn.";
      l.q_proj = take_linear(w, cfg_, ap + "q_proj");
      l.k_proj = take_linear(w, cfg_, ap + "k_proj");
      l.v_proj = take_linear(w, cfg_, ap + "v_proj");
      l.o_proj = take_linear(w, cfg_, ap + "o_proj");
      l.q_norm = take_weight(w, ap + "q_norm.weight");
      l.k_norm = take_weight(w, ap + "k_norm.weight");
    }
    layers_.push_back(std::move(l));
  }
  // Materialize the weights now so loading cost is paid at load time.
  std::vector<array> all;
  auto add = [&](const Linear& l) {
    all.push_back(l.weight);
    if (l.scales) all.push_back(*l.scales);
    if (l.biases) all.push_back(*l.biases);
  };
  add(embed_tokens_);
  add(lm_head_);
  all.push_back(final_norm_);
  for (auto& l : layers_) {
    all.insert(all.end(), {l.input_norm, l.post_norm});
    for (auto* x : {&l.gate_proj, &l.up_proj, &l.down_proj}) add(*x);
    if (l.is_linear) {
      for (auto* x : {&l.in_proj_qkv, &l.in_proj_z, &l.in_proj_b, &l.in_proj_a, &l.out_proj}) add(*x);
      all.insert(all.end(), {l.conv_weight, l.dt_bias, l.A_log, l.gnorm});
    } else {
      for (auto* x : {&l.q_proj, &l.k_proj, &l.v_proj, &l.o_proj}) add(*x);
      all.insert(all.end(), {l.q_norm, l.k_norm});
    }
  }
  mx::eval(all);
  reset();
}

void Qwen35::reset() {
  kv_.assign(layers_.size(), KVCache{});
  lin_.assign(layers_.size(), LinearState{});
  offset_ = 0;
}

size_t CacheState::nbytes() const {
  size_t n = 0;
  for (auto& c : kv)
    if (c.keys) n += c.keys->nbytes() + c.values->nbytes();
  for (auto& c : lin) {
    if (c.conv) n += c.conv->nbytes();
    if (c.state) n += c.state->nbytes();
  }
  return n;
}

std::vector<array> Qwen35::cache_arrays() const {
  std::vector<array> out;
  for (auto& c : kv_) {
    if (c.keys) out.insert(out.end(), {*c.keys, *c.values});
  }
  for (auto& c : lin_) {
    if (c.conv) out.push_back(*c.conv);
    if (c.state) out.push_back(*c.state);
  }
  return out;
}

array Qwen35::embed(const array& tokens) const {
  const auto& e = embed_tokens_;
  if (!e.scales) return mx::take(e.weight, tokens, 0);
  auto w = mx::take(e.weight, tokens, 0);
  auto s = mx::take(*e.scales, tokens, 0);
  std::optional<array> b;
  if (e.biases) b = mx::take(*e.biases, tokens, 0);
  return mx::dequantize(w, s, b, e.q.group_size, e.q.bits, "affine");
}

array Qwen35::mlp(const Layer& l, const array& x) const {
  return l.down_proj(swiglu(l.gate_proj(x), l.up_proj(x)));
}

array Qwen35::attention(Layer& l, KVCache* c, const array& x) {
  int B = x.shape(0), L = x.shape(1);
  int H = cfg_.num_attention_heads, Hkv = cfg_.num_key_value_heads, D = cfg_.head_dim;
  auto qg = mx::reshape(l.q_proj(x), {B, L, H, 2 * D});
  auto parts = mx::split(qg, 2, -1);
  auto queries = parts[0];
  auto gate = mx::reshape(parts[1], {B, L, H * D});
  auto keys = mx::reshape(l.k_proj(x), {B, L, Hkv, D});
  auto values = mx::reshape(l.v_proj(x), {B, L, Hkv, D});
  queries = mx::transpose(mx::fast::rms_norm(queries, l.q_norm, cfg_.rms_norm_eps), {0, 2, 1, 3});
  keys = mx::transpose(mx::fast::rms_norm(keys, l.k_norm, cfg_.rms_norm_eps), {0, 2, 1, 3});
  values = mx::transpose(values, {0, 2, 1, 3});
  int rope_dims = static_cast<int>(D * cfg_.partial_rotary_factor);
  int offset = c ? c->offset : 0;
  queries = mx::fast::rope(queries, rope_dims, false, cfg_.rope_theta, 1.0f, offset);
  keys = mx::fast::rope(keys, rope_dims, false, cfg_.rope_theta, 1.0f, offset);
  auto [k_all, v_all] = c ? c->update(keys, values) : std::make_pair(keys, values);
  auto out = mx::fast::scaled_dot_product_attention(
      queries, k_all, v_all, 1.0f / std::sqrt(static_cast<float>(D)),
      L > 1 ? "causal" : "");
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, L, H * D});
  return l.o_proj(mx::multiply(out, mx::sigmoid(gate)));
}

array Qwen35::gated_delta_net(Layer& l, LinearState* c, const array& x, bool kernel) {
  int B = x.shape(0), S = x.shape(1);
  int Hk = cfg_.linear_num_key_heads, Hv = cfg_.linear_num_value_heads;
  int Dk = cfg_.linear_key_head_dim, Dv = cfg_.linear_value_head_dim;
  int key_dim = Hk * Dk, value_dim = Hv * Dv;
  int conv_dim = 2 * key_dim + value_dim;
  int K = cfg_.linear_conv_kernel_dim;

  auto qkv = l.in_proj_qkv(x);
  auto z = mx::reshape(l.in_proj_z(x), {B, S, Hv, Dv});
  auto b = l.in_proj_b(x);
  auto a = l.in_proj_a(x);

  array conv_state = c && c->conv ? *c->conv : mx::zeros({B, K - 1, conv_dim}, x.dtype());
  auto conv_input = mx::concatenate({conv_state, qkv}, 1);
  int total = conv_input.shape(1);
  if (c) c->conv = mx::contiguous(mx::slice(conv_input, {0, total - (K - 1), 0}, {B, total, conv_dim}));
  auto conv_out = silu(mx::conv1d(conv_input, l.conv_weight, 1, 0, 1, conv_dim));

  auto qs = mx::split(conv_out, mx::Shape{key_dim, 2 * key_dim}, -1);
  auto q = mx::reshape(qs[0], {B, S, Hk, Dk});
  auto k = mx::reshape(qs[1], {B, S, Hk, Dk});
  auto v = mx::reshape(qs[2], {B, S, Hv, Dv});

  // normalize_qk: L2-normalize q/k and fold Dk^-0.5 into q.
  float inv_scale = 1.0f / std::sqrt(static_cast<float>(Dk));
  float rms_eps = 1e-6f * inv_scale * inv_scale;
  q = mx::multiply(scalar(inv_scale * inv_scale, q), mx::fast::rms_norm(q, std::nullopt, rms_eps));
  k = mx::multiply(scalar(inv_scale, k), mx::fast::rms_norm(k, std::nullopt, rms_eps));

  auto [out, state] = gated_delta_update(q, k, v, a, b, l.A_log, l.dt_bias,
                                         c ? c->state : std::nullopt, kernel);
  if (c) c->state = state;

  // RMSNormGated: rms_norm(out) * silu(z), computed in float32.
  auto normed = mx::fast::rms_norm(out, l.gnorm, cfg_.rms_norm_eps);
  auto gated = precise_swiglu(out, z, normed);
  return l.out_proj(mx::reshape(gated, {B, S, value_dim}));
}

array Qwen35::layer_forward(Layer& l, size_t i, const array& h, bool cached) {
  auto n = mx::fast::rms_norm(h, l.input_norm, cfg_.rms_norm_eps);
  auto r = l.is_linear ? gated_delta_net(l, cached ? &lin_[i] : nullptr, n, cached)
                       : attention(l, cached ? &kv_[i] : nullptr, n);
  auto h2 = mx::add(h, r);
  return mx::add(h2, mlp(l, mx::fast::rms_norm(h2, l.post_norm, cfg_.rms_norm_eps)));
}

array Qwen35::forward(const array& tokens) {
  auto h = embed(tokens);
  for (size_t i = 0; i < layers_.size(); ++i) h = layer_forward(layers_[i], i, h, true);
  offset_ += tokens.shape(1);
  h = mx::fast::rms_norm(h, final_norm_, cfg_.rms_norm_eps);
  return lm_head_(h);
}

array Qwen35::forward_train(const array& tokens, bool checkpoint) {
  return lm_head_(hidden_train(tokens, checkpoint));
}

array Qwen35::hidden_train(const array& tokens, bool checkpoint) {
  auto h = embed(tokens);
  for (size_t i = 0; i < layers_.size(); ++i) {
    auto& l = layers_[i];
    if (!checkpoint) {
      h = layer_forward(l, i, h, false);
      continue;
    }
    // Like mlx_lm grad_checkpoint: the layer's trainable arrays are inputs of
    // the checkpointed function, so their gradients flow through it.
    std::vector<Linear*> lins;
    for (auto* x : {&l.gate_proj, &l.up_proj, &l.down_proj, &l.q_proj, &l.k_proj, &l.v_proj,
                    &l.o_proj, &l.in_proj_qkv, &l.in_proj_z, &l.in_proj_b, &l.in_proj_a, &l.out_proj})
      if (x->lora_a) lins.push_back(x);
    std::vector<array> inputs{h};
    for (auto* x : lins) inputs.insert(inputs.end(), {*x->lora_a, *x->lora_b});
    auto fn = mx::checkpoint([this, &l, i, lins](const std::vector<array>& in) {
      std::vector<std::pair<array, array>> saved;
      for (size_t j = 0; j < lins.size(); ++j) {
        saved.emplace_back(*lins[j]->lora_a, *lins[j]->lora_b);
        lins[j]->lora_a = in[1 + 2 * j];
        lins[j]->lora_b = in[2 + 2 * j];
      }
      auto out = layer_forward(l, i, in[0], false);
      for (size_t j = 0; j < lins.size(); ++j) {
        lins[j]->lora_a = saved[j].first;
        lins[j]->lora_b = saved[j].second;
      }
      return std::vector<array>{out};
    });
    h = fn(inputs)[0];
  }
  return mx::fast::rms_norm(h, final_norm_, cfg_.rms_norm_eps);
}

std::vector<std::pair<std::string, Linear*>> Qwen35::lora_targets(
    int num_layers, const std::vector<std::string>& keys) {
  std::vector<std::pair<std::string, Linear*>> out;
  int n = static_cast<int>(layers_.size());
  int first = num_layers < 0 ? 0 : std::max(0, n - num_layers);
  for (int i = first; i < n; ++i) {
    auto& l = layers_[i];
    auto lp = "language_model.model.layers." + std::to_string(i) + ".";
    std::vector<std::pair<std::string, Linear*>> all;
    if (l.is_linear) {
      all = {{"linear_attn.in_proj_qkv", &l.in_proj_qkv}, {"linear_attn.in_proj_z", &l.in_proj_z},
             {"linear_attn.in_proj_b", &l.in_proj_b}, {"linear_attn.in_proj_a", &l.in_proj_a},
             {"linear_attn.out_proj", &l.out_proj}};
    } else {
      all = {{"self_attn.q_proj", &l.q_proj}, {"self_attn.k_proj", &l.k_proj},
             {"self_attn.v_proj", &l.v_proj}, {"self_attn.o_proj", &l.o_proj}};
    }
    all.insert(all.end(), {{"mlp.gate_proj", &l.gate_proj}, {"mlp.up_proj", &l.up_proj},
                           {"mlp.down_proj", &l.down_proj}});
    for (auto& [k, lin] : all) {
      if (keys.empty() || std::find(keys.begin(), keys.end(), k) != keys.end())
        out.emplace_back(lp + k, lin);
    }
  }
  return out;
}

Linear* Qwen35::linear_by_path(const std::string& path) {
  for (auto& [p, lin] : lora_targets(-1, {}))
    if (p == path) return lin;
  return nullptr;
}

void Qwen35::clear_lora() {
  for (auto& [p, lin] : lora_targets(-1, {})) {
    lin->lora_a.reset();
    lin->lora_b.reset();
  }
}

}  // namespace mlxnode
