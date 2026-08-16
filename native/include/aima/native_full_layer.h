// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors
#pragma once

#include "aima/native_decode_executor.h"
#include "aima/native_decode_invocation.h"
#include "aima/native_decode_workspace.h"
#include "aima/native_full_attention.h"
#include "aima/native_weight_store.h"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace aima {

struct NativeFullLayerMetrics {
  std::size_t layer_index = 0;
  std::size_t cache_end = 0;
  std::size_t pv_splits = 0;
  std::size_t aot_launches = 0;
  std::size_t native_attention_launches = 0;
  std::size_t native_projection_launches = 0;
  std::size_t native_pointwise_launches = 0;
  double wall_ms = 0.0;
};

// Qualification-only synchronous observer for one singleton decode full-
// attention core. Product execution passes nullptr and incurs no stream
// synchronization or device-to-host transfer.
using NativeDecodeFullAttentionObserver = std::function<void(
    std::size_t layer_index, std::size_t cache_end,
    const void* qkv_projection, const void* query,
    const void* current_key, const void* current_value,
    const void* key_cache, const void* value_cache,
    const void* attention_output)>;

NativeFullLayerMetrics run_native_full_layer(
    std::size_t layer_index, std::size_t position, std::size_t cache_end,
    const NativeWeightStore& weights, const NativeDecodeWorkspace& workspace,
    const NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, NativeFullAttentionState& attention_state,
    int cu_count, void* stream = nullptr, bool synchronize = true,
    bool use_mrope = false,
    const NativeDecodeFullAttentionObserver* attention_observer = nullptr);

}  // namespace aima
