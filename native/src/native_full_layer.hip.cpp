// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_full_layer.h"

#include "aima/bf16_wvsplitk.h"
#include "aima/native_pointwise.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>

namespace aima {
namespace {

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kQueryDimension = 4096;
constexpr std::size_t kSharedIntermediate = 512;
constexpr std::size_t kRawValueOffsetElements = 8704;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

const NativeDecodeWorkspaceView& require_workspace(
    const NativeDecodeWorkspace& workspace, const std::string& name,
    std::uint64_t minimum_bytes) {
  const NativeDecodeWorkspaceView* view = workspace.find(name);
  if (view == nullptr || view->device_pointer == nullptr ||
      view->payload_bytes < minimum_bytes ||
      view->dtype != DecodeTensorDtype::kBfloat16) {
    throw std::runtime_error("native full layer workspace mismatch: " + name);
  }
  return *view;
}

const NativeTensorView& require_weight(const NativeWeightStore& weights,
                                       const std::string& name,
                                       std::uint64_t expected_bytes) {
  const NativeTensorView* view = weights.find(name);
  if (view == nullptr || view->device_pointer == nullptr ||
      view->payload_bytes != expected_bytes) {
    throw std::runtime_error("native full layer weight mismatch: " + name);
  }
  return *view;
}

const char* require_argument_binding(const PreparedDecodeInvocation& invocation,
                                     const char* argument_name) {
  if (invocation.launch == nullptr) {
    throw std::runtime_error("native full layer invocation is missing");
  }
  for (std::size_t index = 0; index < invocation.launch->argument_count; ++index) {
    const DecodeArgument& argument = invocation.launch->arguments[index];
    if (argument.kind == DecodeArgumentKind::kTensor &&
        std::string(argument.name) == argument_name) {
      return argument.binding;
    }
  }
  throw std::runtime_error("native full layer schedule argument is missing: " +
                           std::string(argument_name));
}

}  // namespace

NativeFullLayerMetrics run_native_full_layer(
    std::size_t layer_index, std::size_t position, std::size_t cache_end,
    const NativeWeightStore& weights, const NativeDecodeWorkspace& workspace,
    const NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, NativeFullAttentionState& attention_state,
    int cu_count, void* stream_value, bool synchronize,
    const NativeMoeOverlapResources* moe_overlap) {
  const auto& launches = invocations.launches();
  const std::size_t base = layer_index * 10;
  if (!executor.loaded() || base + 10 >= launches.size() || cu_count <= 0) {
    throw std::runtime_error("native full layer owners are incomplete");
  }
  static constexpr const char* kExpectedSymbols[] = {
      "triton_rmsnorm_kernel",
      "triton_matvec_kernel",
      "triton_head_norm_rope_kernel",
      "triton_head_norm_rope_kernel",
      "triton_rmsnorm_kernel",
      "triton_matvec_kernel",
      "triton_router_topk_stage1_kernel",
      "triton_router_topk_stage2_softmax_kernel",
      "raw_row_gate_up_activation_kernel",
      "raw_row_down_sum_kernel",
  };
  for (std::size_t offset = 0; offset < 10; ++offset) {
    if (launches[base + offset].launch == nullptr ||
        launches[base + offset].launch->layer_index !=
            static_cast<std::int16_t>(layer_index) ||
        std::string(launches[base + offset].launch->symbol) !=
            kExpectedSymbols[offset]) {
      throw std::runtime_error(
          "requested layer is not a qualified full-attention template");
    }
  }

  const auto& input = require_workspace(
      workspace, require_argument_binding(launches[base], "x"),
      kHidden * sizeof(__hip_bfloat16));
  const auto& qkv = require_workspace(
      workspace, require_argument_binding(launches[base + 1], "out"),
      9216 * sizeof(__hip_bfloat16));
  const auto& q = require_workspace(
      workspace, require_argument_binding(launches[base + 2], "out"),
      kQueryDimension * sizeof(__hip_bfloat16));
  const auto& k = require_workspace(
      workspace, require_argument_binding(launches[base + 3], "out"),
      512 * sizeof(__hip_bfloat16));
  const auto& after_attn = require_workspace(
      workspace, require_argument_binding(launches[base + 4], "x"),
      kHidden * sizeof(__hip_bfloat16));
  const auto& shared_input = require_workspace(
      workspace, require_argument_binding(launches[base + 5], "out"),
      (1 + 2 * kSharedIntermediate) * sizeof(__hip_bfloat16));
  const auto& routed_moe = require_workspace(
      workspace, require_argument_binding(launches[base + 9], "out"),
      kHidden * sizeof(__hip_bfloat16));
  const auto& output = require_workspace(
      workspace, require_argument_binding(launches[base + 10], "x"),
      kHidden * sizeof(__hip_bfloat16));
  const auto& activated = require_workspace(
      workspace, "native.linear.shared_activation",
      kSharedIntermediate * sizeof(__hip_bfloat16));
  const auto& shared_down = require_workspace(
      workspace, "native.linear.shared_down", kHidden * sizeof(__hip_bfloat16));
  const auto& shared_scaled = require_workspace(
      workspace, "native.linear.shared_scaled", kHidden * sizeof(__hip_bfloat16));
  const auto& combined_moe = require_workspace(
      workspace, "native.linear.combined_moe", kHidden * sizeof(__hip_bfloat16));

  const std::string prefix =
      "model.language_model.layers." + std::to_string(layer_index);
  const auto& output_weight = require_weight(
      weights, prefix + ".self_attn.o_proj.weight",
      kHidden * kQueryDimension * sizeof(__hip_bfloat16));
  const auto& shared_down_weight = require_weight(
      weights, prefix + ".mlp.shared_expert.down_proj.weight",
      kHidden * kSharedIntermediate * sizeof(__hip_bfloat16));

  const auto started = std::chrono::steady_clock::now();
  hipStream_t stream = static_cast<hipStream_t>(stream_value);
  if (moe_overlap != nullptr && !moe_overlap->valid()) {
    throw std::runtime_error(
        "native full layer MoE overlap resources are incomplete");
  }
  const bool overlap_enabled =
      moe_overlap != nullptr && moe_overlap->valid();
  hipStream_t shared_stream =
      overlap_enabled
          ? static_cast<hipStream_t>(moe_overlap->auxiliary_stream)
          : stream;
  hipEvent_t branch_ready =
      overlap_enabled
          ? static_cast<hipEvent_t>(moe_overlap->branch_ready_event)
          : nullptr;
  hipEvent_t shared_done =
      overlap_enabled
          ? static_cast<hipEvent_t>(moe_overlap->shared_done_event)
          : nullptr;
  NativeFullLayerMetrics metrics;
  metrics.layer_index = layer_index;
  metrics.cache_end = cache_end;
  for (std::size_t offset = 0; offset < 4; ++offset) {
    executor.launch(launches[base + offset], stream);
    ++metrics.aot_launches;
  }
  const auto* raw_v = static_cast<const __hip_bfloat16*>(qkv.device_pointer) +
                      kRawValueOffsetElements;
  const NativeFullAttentionCoreMetrics attention =
      launch_native_grouped_full_attention(
          layer_index, position, cache_end, q.device_pointer,
          k.device_pointer, raw_v, attention_state, stream);
  metrics.pv_splits = attention.pv_splits;
  metrics.native_attention_launches = attention.native_kernel_launches;
  launch_full_attention_sigmoid_gate(
      attention_state.attention_output(), qkv.device_pointer,
      attention_state.gated_attention(), stream);
  ++metrics.native_pointwise_launches;
  launch_bf16_wvsplitk(
      output_weight.device_pointer, attention_state.gated_attention(), nullptr,
      attention_state.projected_attention(), kHidden, kQueryDimension,
      cu_count, stream);
  ++metrics.native_projection_launches;
  launch_bf16_add(input.device_pointer,
                  attention_state.projected_attention(),
                  after_attn.device_pointer, kHidden, stream);
  ++metrics.native_pointwise_launches;

  executor.launch(launches[base + 4], stream);
  ++metrics.aot_launches;
  if (overlap_enabled) {
    check_hip(hipEventRecord(branch_ready, stream),
              "hipEventRecord native full MoE branch ready");
    check_hip(hipStreamWaitEvent(shared_stream, branch_ready, 0),
              "hipStreamWaitEvent native full shared branch ready");
  }
  executor.launch(launches[base + 5], shared_stream);
  ++metrics.aot_launches;
  launch_shared_silu_multiply(shared_input.device_pointer,
                              activated.device_pointer, shared_stream);
  ++metrics.native_pointwise_launches;
  launch_bf16_wvsplitk(
      shared_down_weight.device_pointer, activated.device_pointer, nullptr,
      shared_down.device_pointer, kHidden, kSharedIntermediate, cu_count,
      shared_stream);
  ++metrics.native_projection_launches;
  launch_shared_sigmoid_scale(shared_input.device_pointer,
                              shared_down.device_pointer,
                              shared_scaled.device_pointer, shared_stream);
  ++metrics.native_pointwise_launches;
  if (overlap_enabled) {
    check_hip(hipEventRecord(shared_done, shared_stream),
              "hipEventRecord native full shared branch done");
  }
  for (std::size_t offset = 6; offset < 10; ++offset) {
    executor.launch(launches[base + offset], stream);
    ++metrics.aot_launches;
  }
  if (overlap_enabled) {
    check_hip(hipStreamWaitEvent(stream, shared_done, 0),
              "hipStreamWaitEvent native full shared branch done");
  }
  launch_bf16_add_pair(
      routed_moe.device_pointer, shared_scaled.device_pointer,
      after_attn.device_pointer, combined_moe.device_pointer,
      output.device_pointer, kHidden, stream);
  ++metrics.native_pointwise_launches;
  if (synchronize) {
    check_hip(hipStreamSynchronize(stream),
              "hipStreamSynchronize native full layer");
  }
  metrics.wall_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started)
                        .count();
  return metrics;
}

}  // namespace aima
