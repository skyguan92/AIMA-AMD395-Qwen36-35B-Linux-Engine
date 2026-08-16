// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors
#pragma once

#include "aima/native_decode_executor.h"
#include "aima/native_decode_invocation.h"
#include "aima/native_decode_workspace.h"
#include "aima/native_full_attention.h"
#include "aima/native_full_layer.h"
#include "aima/native_lm_head.h"
#include "aima/native_linear_layer.h"
#include "aima/native_weight_store.h"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace aima {

// Qualification-only synchronous observer for the accumulated BF16 hidden row
// after each decode layer (0..39) and the final normalized row (40).  Product
// execution passes nullptr and pays no device-to-host transfer or synchronization.
using NativeDecodeLayerObserver =
    std::function<void(std::size_t boundary_index, const void* device_row)>;

struct NativeDecodeRunMetrics {
  std::size_t layer_count = 0;
  std::size_t linear_layer_count = 0;
  std::size_t full_layer_count = 0;
  std::size_t aot_launches = 0;
  std::size_t native_attention_launches = 0;
  std::size_t native_projection_launches = 0;
  std::size_t native_pointwise_launches = 0;
  std::size_t resident_state_pointer_swaps = 0;
  std::size_t native_lm_head_certificate_launches = 0;
  std::size_t lm_head_candidate_count = 0;
  bool lm_head_certified = false;
  std::uint32_t top1_token_id = 0;
  float top1_logit = 0.0f;
  double layer_submission_ms = 0.0;
  double synchronized_wall_ms = 0.0;
};

struct NativeDecodePrepareMetrics {
  std::size_t native_kernel_launches = 0;
  std::size_t position = 0;
  std::size_t rotary_position = 0;
  std::uint32_t input_token_id = 0;
};

struct NativeLmHeadTop1Metrics {
  std::size_t aot_launches = 0;
  std::size_t native_lm_head_certificate_launches = 0;
  std::size_t candidate_count = 0;
  bool certified = false;
  std::uint32_t top1_token_id = 0;
  float top1_logit = 0.0f;
  double synchronized_wall_ms = 0.0;
};

// Prepares the next resident decode step without host-side tensor libraries:
// device embedding lookup plus the model's 64-dimensional rotary slice.
NativeDecodePrepareMetrics prepare_native_decode_step(
    std::size_t position, std::uint32_t input_token_id,
    const NativeWeightStore& weights,
    const NativeDecodeInvocations& invocations, void* stream = nullptr);

// VL decode stores the token at the ordinary cache position while applying
// the shared post-prompt M-RoPE position derived from the request delta.
// Keeping the two coordinates explicit prevents cache geometry from being
// conflated with the rotary coordinate.
NativeDecodePrepareMetrics prepare_native_decode_step(
    std::size_t position, std::size_t rotary_position,
    std::uint32_t input_token_id, const NativeWeightStore& weights,
    const NativeDecodeInvocations& invocations, void* stream = nullptr);

// Runs the shared final RMSNorm, int8 global scan, and exact shortlist
// certificate from one resident BF16 hidden row.  Prefill and decode use this
// same product primitive, so first-token selection does not depend on Python.
NativeLmHeadTop1Metrics run_native_lm_head_top1(
    const void* final_hidden_row, const NativeWeightStore& weights,
    const NativeLmHeadStore& lm_head,
    const NativeDecodeWorkspace& workspace,
    const NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, int cu_count,
    const std::uint8_t* allowed_token_mask = nullptr,
    void* stream = nullptr);

// Executes one q8192 decode token from already-resident state.  All forty
// parameterized layers are submitted to one stream without per-layer host
// synchronization, followed by the captured final RMSNorm/int8 LM-head and a
// native globally certified top-1 selection.
NativeDecodeRunMetrics run_native_decode_token(
    std::size_t position, std::size_t cache_end,
    const NativeWeightStore& weights, const NativeLmHeadStore& lm_head,
    const NativeDecodeWorkspace& workspace,
    NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, NativeFullAttentionState& attention_state,
    int cu_count, const std::uint8_t* allowed_token_mask = nullptr,
    void* stream = nullptr,
    const NativeDecodeLayerObserver* layer_observer = nullptr,
    const NativeDecodeLinearLayer0Observer* linear_layer0_observer = nullptr,
    const NativeDecodeLinearLayer0Observer* layer0_tail_observer = nullptr,
    const NativeDecodeFullAttentionObserver* full_attention_observer = nullptr);

}  // namespace aima
