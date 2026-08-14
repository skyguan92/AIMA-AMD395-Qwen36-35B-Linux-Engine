#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <cstddef>
#include <cstdint>
#include <memory>

namespace aima {

class NativeDecodeExecutor;

// Process-resident launcher for the exact vLLM unified-attention prefill
// specialization used by Qwen3.5-VL on gfx1151. The HSACO is embedded in the
// native binary; construction loads it once and uploads immutable lookup
// tables for every admitted query/KV length. Launch performs no allocation,
// filesystem access, host-to-device copy, or Python/Torch/Triton runtime work.
struct NativeVlUnifiedAttentionMetrics {
  bool loaded = false;
  std::size_t image_bytes = 0;
  std::size_t metadata_bytes = 0;
  std::size_t max_query_tokens = 0;
  std::size_t max_kv_tokens = 0;
  std::size_t cache_blocks = 0;
  std::size_t launches = 0;
};

class NativeVlUnifiedAttentionPlan {
 public:
  NativeVlUnifiedAttentionPlan(NativeDecodeExecutor& executor,
                               std::size_t max_query_tokens,
                               std::size_t max_kv_tokens,
                               int device = 0);
  ~NativeVlUnifiedAttentionPlan();
  NativeVlUnifiedAttentionPlan(const NativeVlUnifiedAttentionPlan&) = delete;
  NativeVlUnifiedAttentionPlan& operator=(
      const NativeVlUnifiedAttentionPlan&) = delete;
  NativeVlUnifiedAttentionPlan(NativeVlUnifiedAttentionPlan&&) noexcept;
  NativeVlUnifiedAttentionPlan& operator=(
      NativeVlUnifiedAttentionPlan&&) noexcept;

  // Q/output are contiguous BF16 [query_tokens,16,256]. K/V are views of a
  // contiguous BF16 paged cache with physical block shape [1056,2,256]. A
  // smaller query than KV length represents a causal suffix prefill.
  void launch(const void* query_bf16, const void* key_cache_bf16,
              const void* value_cache_bf16, void* output_bf16,
              std::size_t query_tokens, std::size_t kv_tokens,
              void* stream = nullptr);

  const NativeVlUnifiedAttentionMetrics& metrics() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
