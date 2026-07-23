// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors
#pragma once

#include "aima/native_decode_executor.h"
#include "aima/native_decode_invocation.h"
#include "aima/native_decode_workspace.h"
#include "aima/native_full_attention.h"
#include "aima/native_moe_overlap.h"
#include "aima/native_weight_store.h"

#include <cstddef>

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

NativeFullLayerMetrics run_native_full_layer(
    std::size_t layer_index, std::size_t position, std::size_t cache_end,
    const NativeWeightStore& weights, const NativeDecodeWorkspace& workspace,
    const NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, NativeFullAttentionState& attention_state,
    int cu_count, void* stream = nullptr, bool synchronize = true,
    const NativeMoeOverlapResources* moe_overlap = nullptr);

}  // namespace aima
