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

class Bf16GemmPlan;
struct NativeDecodeNextInputNorm;

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
struct NativeDecodeFullAttentionObservation {
  std::size_t layer_index = 0;
  std::size_t cache_end = 0;
  const void* qkv_projection = nullptr;
  const void* query = nullptr;
  const void* current_key = nullptr;
  const void* current_value = nullptr;
  const void* key_cache = nullptr;
  const void* value_cache = nullptr;
  const void* attention_output = nullptr;
  const void* gated_attention = nullptr;
  const void* projected_attention = nullptr;
  const void* attention_residual = nullptr;
  const void* post_attention_norm = nullptr;
  const void* shared_gate_logits = nullptr;
  const void* shared_gate_up_projection = nullptr;
  const void* shared_activation = nullptr;
  const void* shared_down_projection = nullptr;
  const void* shared_moe_output = nullptr;
  const void* router_logits = nullptr;
  const void* router_weights = nullptr;
  const void* router_indices = nullptr;
  const void* routed_gate_up_projection = nullptr;
  const void* routed_activation = nullptr;
  const void* routed_weighted_expert_outputs = nullptr;
  const void* routed_moe_output = nullptr;
  const void* combined_moe_output = nullptr;
};

using NativeDecodeFullAttentionObserver = std::function<void(
    const NativeDecodeFullAttentionObservation& observation)>;

NativeFullLayerMetrics run_native_full_layer(
    std::size_t layer_index, std::size_t position, std::size_t cache_end,
    const NativeWeightStore& weights, const NativeDecodeWorkspace& workspace,
    const NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, NativeFullAttentionState& attention_state,
    int cu_count, void* stream = nullptr, bool synchronize = true,
    bool use_mrope = false,
    const Bf16GemmPlan* shared_gate_plan = nullptr,
    const NativeDecodeFullAttentionObserver* attention_observer = nullptr,
    bool input_norm_precomputed = false,
    const NativeDecodeNextInputNorm* next_input_norm = nullptr,
    const void* mrope_cosine_fp32 = nullptr,
    const void* mrope_sine_fp32 = nullptr);

}  // namespace aima
