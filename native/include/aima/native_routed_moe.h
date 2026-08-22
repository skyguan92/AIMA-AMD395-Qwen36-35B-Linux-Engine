#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_executor.h"

#include <cstddef>

namespace aima {

// O(1) singleton-decode buffers for the current vLLM routed-MoE contract.
// Router weights remain FP32 through the second expert GEMM; every other
// activation/output buffer is BF16 and expert ids are int32.
struct NativeDecodeRoutedMoeBuffers {
  void* router_logits_bf16 = nullptr;
  void* router_weights_fp32 = nullptr;
  void* router_indices_i32 = nullptr;
  void* num_tokens_post_padded_i32 = nullptr;
  void* gate_up_bf16 = nullptr;
  void* activation_bf16 = nullptr;
  void* weighted_expert_outputs_bf16 = nullptr;
  void* output_bf16 = nullptr;
};

struct NativeDecodeRoutedMoeMetrics {
  std::size_t native_projection_launches = 0;
  std::size_t native_pointwise_launches = 0;
  std::size_t aot_launches = 0;
};

// The resident decoder is serial. These helpers expose one non-blocking side
// stream and event pair so the shared expert can execute beside the routed
// experts without changing either arithmetic chain. Call begin after the
// common input is ready on main_stream and complete before consuming the
// shared output on main_stream.
void* begin_native_decode_shared_expert_overlap(void* main_stream = nullptr);
void complete_native_decode_shared_expert_overlap(
    void* main_stream = nullptr);

// Materializes the current-vLLM singleton MoE tail after both the routed and
// shared expert projections are ready.  The fused launch preserves each BF16
// boundary exposed by the unfused chain: routed reduction, shared sigmoid
// scale, routed-plus-shared combine, and residual add.
void launch_native_decode_moe_tail(
    const void* weighted_expert_outputs_bf16,
    const void* fused_shared_input_bf16, const void* shared_down_bf16,
    const void* residual_bf16, void* routed_output_bf16,
    void* shared_output_bf16, void* combined_output_bf16,
    void* output_bf16, void* stream = nullptr);

// Executes the pinned current-vLLM singleton routed-MoE chain:
// BF16 router projection, FP32 normalized top-8 routing, two embedded Triton
// expert GEMMs, BF16-boundary SiLU-and-multiply, and the eight-row reduction.
NativeDecodeRoutedMoeMetrics run_native_decode_routed_moe(
    const void* hidden_bf16, const void* router_weight_bf16,
    const void* gate_up_weight_bf16, const void* down_weight_bf16,
    const NativeDecodeRoutedMoeBuffers& buffers,
    NativeDecodeExecutor& executor, int cu_count,
    void* stream = nullptr);

// Continue the same routed-MoE chain when an exact singleton projection has
// already populated buffers.router_logits_bf16 on the same stream.
NativeDecodeRoutedMoeMetrics run_native_decode_routed_moe_from_logits(
    const void* hidden_bf16, const void* gate_up_weight_bf16,
    const void* down_weight_bf16,
    const NativeDecodeRoutedMoeBuffers& buffers,
    NativeDecodeExecutor& executor, void* stream = nullptr,
    bool defer_sum = false);

}  // namespace aima
