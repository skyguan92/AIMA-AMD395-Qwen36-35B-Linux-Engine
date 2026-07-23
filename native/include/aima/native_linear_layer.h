#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_executor.h"
#include "aima/native_decode_invocation.h"
#include "aima/native_decode_workspace.h"
#include "aima/native_moe_overlap.h"
#include "aima/native_weight_store.h"

#include <cstddef>

namespace aima {

struct NativeLinearLayerMetrics {
  std::size_t layer_index = 0;
  std::size_t aot_launches = 0;
  std::size_t native_projection_launches = 0;
  std::size_t native_pointwise_launches = 0;
  double wall_ms = 0.0;
};

// Executes one decode-specialized linear-attention + MoE layer.  The same
// parameterized implementation is reused for every linear-attention layer;
// all state and O(1) scratch pointers are resident owners.
NativeLinearLayerMetrics run_native_linear_layer(
    std::size_t layer_index,
    const NativeWeightStore& weights,
    const NativeDecodeWorkspace& workspace,
    const NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor,
    int cu_count,
    void* stream = nullptr,
    bool synchronize = true,
    const NativeMoeOverlapResources* moe_overlap = nullptr);

}  // namespace aima
