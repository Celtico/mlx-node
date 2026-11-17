// Prompt cache like mlx_lm's LRUPromptCache as the Python worker uses it
// (src/core/assistant/mlx-worker.py): copies of the model cache, keyed by the
// tokens they have read. A new prompt starts from the longest copy it begins
// with. The Gated DeltaNet state cannot be rewound, so only exact prefixes count.
#pragma once

#include <algorithm>
#include <list>
#include <optional>
#include <utility>
#include <vector>

#include "qwen35.h"

namespace mlxnode {

class PromptCache {
 public:
  size_t max_entries = 8;
  size_t max_bytes = 1000000000;

  // Kinds of entry, in mlx_lm's eviction order (LRUPromptCache.CacheOrder):
  // the state after an answer ("assistant") goes before the checkpoints of a
  // prompt ("system"), which the next prompts that share their start need.
  enum Kind { assistant = 0, user = 1, system = 2 };

  // The longest entry the prompt begins with that leaves at least one token
  // to read (the last one gives the logits): its length and a copy of it.
  // As in mlx_lm, using an entry does not change its place in the order.
  std::optional<std::pair<size_t, CacheState>> find(const std::vector<int32_t>& tokens) const {
    const Entry* best = nullptr;
    for (auto& e : entries_) {
      size_t n = e.tokens.size();
      if (n < tokens.size() && (!best || n > best->tokens.size()) &&
          std::equal(e.tokens.begin(), e.tokens.end(), tokens.begin()))
        best = &e;
    }
    if (!best) return std::nullopt;
    return std::make_pair(best->tokens.size(), best->state);
  }

  bool has(const std::vector<int32_t>& tokens, size_t length) const {
    return std::any_of(entries_.begin(), entries_.end(), [&](const Entry& e) {
      return e.tokens.size() == length && std::equal(e.tokens.begin(), e.tokens.end(), tokens.begin());
    });
  }

  // Adds (or replaces) an entry, then drops entries as mlx_lm's insert_cache
  // does: one if there are more than max_entries, then while over max_bytes.
  void insert(std::vector<int32_t> tokens, CacheState state, Kind kind) {
    entries_.remove_if([&](const Entry& e) {
      if (e.tokens != tokens) return false;
      bytes_ -= e.bytes;
      return true;
    });
    size_t n = state.nbytes();
    entries_.push_back({std::move(tokens), std::move(state), n, kind});
    bytes_ += n;
    if (entries_.size() > max_entries) pop();
    while (!entries_.empty() && bytes_ > max_bytes) pop();
  }

  void clear() {
    entries_.clear();
    bytes_ = 0;
  }
  size_t size() const { return entries_.size(); }
  size_t bytes() const { return bytes_; }

 private:
  struct Entry {
    std::vector<int32_t> tokens;
    CacheState state;
    size_t bytes;
    Kind kind;
  };

  size_t count(Kind kind) const {
    return std::count_if(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.kind == kind; });
  }

  // CacheOrder.pop: the oldest entry of the first kind that has at least as
  // many entries as the next kind; else of the last kind.
  void pop() {
    Kind kind = system;
    for (int i = 0; i + 1 < 3; ++i) {
      size_t a = count(static_cast<Kind>(i)), b = count(static_cast<Kind>(i + 1));
      if (a > 0 && a >= b) {
        kind = static_cast<Kind>(i);
        break;
      }
    }
    auto it = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.kind == kind; });
    if (it == entries_.end()) it = entries_.begin();
    bytes_ -= it->bytes;
    entries_.erase(it);
  }

  std::list<Entry> entries_;  // oldest first
  size_t bytes_ = 0;
};

}  // namespace mlxnode
