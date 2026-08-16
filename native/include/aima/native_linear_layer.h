#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_executor.h"
#include "aima/native_decode_invocation.h"
#include "aima/native_decode_workspace.h"
#include "aima/native_weight_store.h"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace aima {

class Bf16GemmPlan;

struct NativeLinearLayerMetrics {
  std::size_t layer_index = 0;
  std::size_t aot_launches = 0;
  std::size_t native_projection_launches = 0;
  std::size_t native_pointwise_launches = 0;
  double wall_ms = 0.0;
};

// Qualification-only synchronous observer for current-vLLM layer-0 decode
// stages. The attention and post-attention/MoE boundary sets use separate
// callbacks so the frozen v3 attention oracle remains stable. Product
// execution passes nullptr and pays no device-to-host transfer or
// synchronization.
using NativeDecodeLinearLayer0Observer = std::function<void(
    const char* boundary_name, const void* device_tensor,
    std::uint64_t tensor_bytes, DecodeTensorDtype dtype)>;

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
    bool use_current_vllm_projections = false,
    const Bf16GemmPlan* shared_gate_plan = nullptr,
    const NativeDecodeLinearLayer0Observer* observer = nullptr,
    const NativeDecodeLinearLayer0Observer* tail_observer = nullptr);

}  // namespace aima
