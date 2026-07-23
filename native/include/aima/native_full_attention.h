// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aima {

struct NativeFullAttentionQkPlan;

struct NativeFullAttentionStateMetrics {
  std::size_t cache_capacity = 0;
  std::size_t full_attention_layers = 0;
  std::size_t maximum_pv_splits = 0;
  std::uint64_t cache_payload_bytes = 0;
  std::uint64_t scratch_payload_bytes = 0;
  std::uint64_t qk_workspace_bytes = 0;
  std::uint64_t allocation_bytes = 0;
  double allocation_and_zero_ms = 0.0;
};

// Resident KV and O(1)-in-layer-count scratch for the ten Qwen3.6 full-
// attention layers.  Cache payload necessarily scales with context and layer
// count; code and transient scratch remain one shared parameterized surface.
class NativeFullAttentionState {
 public:
  NativeFullAttentionState();
  ~NativeFullAttentionState();
  NativeFullAttentionState(const NativeFullAttentionState&) = delete;
  NativeFullAttentionState& operator=(const NativeFullAttentionState&) = delete;

  NativeFullAttentionStateMetrics build(std::size_t cache_capacity,
                                        int device = 0);
  void reset() noexcept;
  bool built() const { return allocation_ != nullptr; }
  std::size_t cache_capacity() const { return cache_capacity_; }
  std::size_t maximum_pv_splits() const { return maximum_pv_splits_; }

  void* k_cache(std::size_t layer_index) const;
  void* v_cache(std::size_t layer_index) const;
  void* scores() const { return scores_; }
  void* probabilities() const { return probabilities_; }
  void* softmax_exponentials() const { return softmax_exponentials_; }
  void* pv_partials() const { return pv_partials_; }
  void* attention_output() const { return attention_output_; }
  void* gated_attention() const { return gated_attention_; }
  void* projected_attention() const { return projected_attention_; }
  void launch_grouped_qk(const void* q, const void* k_cache, void* scores,
                         void* stream) const;
  void launch_grouped_pv(const void* probabilities, const void* v_cache,
                         void* attention, void* stream) const;

 private:
  static std::size_t layer_slot(std::size_t layer_index);

  int device_ = 0;
  void* allocation_ = nullptr;
  std::uint64_t allocation_bytes_ = 0;
  std::size_t cache_capacity_ = 0;
  std::size_t maximum_pv_splits_ = 0;
  std::array<void*, 10> k_caches_{};
  std::array<void*, 10> v_caches_{};
  void* scores_ = nullptr;
  void* probabilities_ = nullptr;
  void* softmax_exponentials_ = nullptr;
  void* pv_partials_ = nullptr;
  void* attention_output_ = nullptr;
  void* gated_attention_ = nullptr;
  void* projected_attention_ = nullptr;
  std::unique_ptr<NativeFullAttentionQkPlan> qk_plan_;
};

struct NativeFullAttentionCoreMetrics {
  std::size_t layer_index = 0;
  std::size_t cache_end = 0;
  std::size_t pv_splits = 0;
  std::size_t native_kernel_launches = 0;
};

// q is [16,256], normalized_k/raw_v are [2,256], all BF16.  The cache uses
// sequence-major [capacity,2,256], matching the qualified q8192 engine.
NativeFullAttentionCoreMetrics launch_native_grouped_full_attention(
    std::size_t layer_index, std::size_t position, std::size_t cache_end,
    const void* q, const void* normalized_k, const void* raw_v,
    NativeFullAttentionState& state, void* stream = nullptr);

}  // namespace aima
