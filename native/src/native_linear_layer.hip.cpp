// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_linear_layer.h"

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
constexpr std::size_t kLinearValue = 4096;
constexpr std::size_t kSharedIntermediate = 512;

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
    throw std::runtime_error("native linear layer workspace mismatch: " + name);
  }
  return *view;
}

const NativeTensorView& require_weight(const NativeWeightStore& weights,
                                       const std::string& name,
                                       std::uint64_t expected_bytes) {
  const NativeTensorView* view = weights.find(name);
  if (view == nullptr || view->device_pointer == nullptr ||
      view->payload_bytes != expected_bytes) {
    throw std::runtime_error("native linear layer weight mismatch: " + name);
  }
  return *view;
}

const char* require_argument_binding(const PreparedDecodeInvocation& invocation,
                                     const char* argument_name) {
  if (invocation.launch == nullptr) {
    throw std::runtime_error("native linear layer invocation is missing");
  }
  for (std::size_t index = 0; index < invocation.launch->argument_count; ++index) {
    const DecodeArgument& argument = invocation.launch->arguments[index];
    if (argument.kind == DecodeArgumentKind::kTensor &&
        std::string(argument.name) == argument_name) {
      return argument.binding;
    }
  }
  throw std::runtime_error(
      "native linear layer schedule argument is missing: " +
      std::string(argument_name));
}

}  // namespace

NativeLinearLayerMetrics run_native_linear_layer(
    std::size_t layer_index, const NativeWeightStore& weights,
    const NativeDecodeWorkspace& workspace,
    const NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, int cu_count, void* stream_value,
    bool synchronize) {
  const auto& launches = invocations.launches();
  const std::size_t base = layer_index * 10;
  if (!executor.loaded() || base + 10 > launches.size() ||
      base + 10 >= launches.size() || cu_count <= 0) {
    throw std::runtime_error("native linear layer owners are incomplete");
  }
  static constexpr const char* kExpectedSymbols[] = {
      "triton_rmsnorm_kernel",
      "triton_fused_input_proj_conv_kernel",
      "fused_sigmoid_gating_delta_rule_update_kernel",
      "triton_linear_gated_norm_kernel",
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
      throw std::runtime_error("requested layer is not a qualified linear-attention template");
    }
  }

  const std::string suffix = std::to_string(layer_index);
  const auto& gated = require_workspace(
      workspace, require_argument_binding(launches[base + 3], "out"),
      kLinearValue * sizeof(__hip_bfloat16));
  const auto& after_attn = require_workspace(
      workspace, require_argument_binding(launches[base + 4], "x"),
      kHidden * sizeof(__hip_bfloat16));
  const auto& activated = require_workspace(
      workspace, "native.linear.shared_activation",
      kSharedIntermediate * sizeof(__hip_bfloat16));
  const auto& shared_down = require_workspace(
      workspace, "native.linear.shared_down", kHidden * sizeof(__hip_bfloat16));
  const auto& attention_output = require_workspace(
      workspace, "native.linear.attention_output",
      kHidden * sizeof(__hip_bfloat16));
  const auto& shared_scaled = require_workspace(
      workspace, "native.linear.shared_scaled", kHidden * sizeof(__hip_bfloat16));
  const auto& combined_moe = require_workspace(
      workspace, "native.linear.combined_moe", kHidden * sizeof(__hip_bfloat16));
  const auto& shared_input = require_workspace(
      workspace, require_argument_binding(launches[base + 5], "out"),
      (1 + 2 * kSharedIntermediate) * sizeof(__hip_bfloat16));
  const auto& routed_moe = require_workspace(
      workspace, require_argument_binding(launches[base + 9], "out"),
      kHidden * sizeof(__hip_bfloat16));

  const auto& input = require_workspace(
      workspace, require_argument_binding(launches[base], "x"),
      kHidden * sizeof(__hip_bfloat16));
  const auto& output = require_workspace(
      workspace, require_argument_binding(launches[base + 10], "x"),
      kHidden * sizeof(__hip_bfloat16));
  const std::string prefix =
      "model.language_model.layers." + suffix;
  const auto& output_weight = require_weight(
      weights, prefix + ".linear_attn.out_proj.weight",
      kHidden * kLinearValue * sizeof(__hip_bfloat16));
  const auto& shared_down_weight = require_weight(
      weights, prefix + ".mlp.shared_expert.down_proj.weight",
      kHidden * kSharedIntermediate * sizeof(__hip_bfloat16));

  const auto started = std::chrono::steady_clock::now();
  hipStream_t stream = static_cast<hipStream_t>(stream_value);
  NativeLinearLayerMetrics metrics;
  metrics.layer_index = layer_index;
  for (std::size_t offset = 0; offset < 4; ++offset) {
    executor.launch(launches[base + offset], stream);
    ++metrics.aot_launches;
  }
  launch_bf16_wvsplitk(
      output_weight.device_pointer, gated.device_pointer, nullptr,
      attention_output.device_pointer, kHidden, kLinearValue, cu_count, stream);
  ++metrics.native_projection_launches;
  launch_bf16_add(input.device_pointer, attention_output.device_pointer,
                  after_attn.device_pointer, kHidden, stream);
  ++metrics.native_pointwise_launches;

  executor.launch(launches[base + 4], stream);
  executor.launch(launches[base + 5], stream);
  metrics.aot_launches += 2;
  launch_shared_silu_multiply(shared_input.device_pointer,
                              activated.device_pointer, stream);
  ++metrics.native_pointwise_launches;
  launch_bf16_wvsplitk(
      shared_down_weight.device_pointer, activated.device_pointer, nullptr,
      shared_down.device_pointer, kHidden, kSharedIntermediate, cu_count,
      stream);
  ++metrics.native_projection_launches;
  launch_shared_sigmoid_scale(shared_input.device_pointer,
                              shared_down.device_pointer,
                              shared_scaled.device_pointer, stream);
  ++metrics.native_pointwise_launches;

  for (std::size_t offset = 6; offset < 10; ++offset) {
    executor.launch(launches[base + offset], stream);
    ++metrics.aot_launches;
  }
  launch_bf16_add_pair(
      routed_moe.device_pointer, shared_scaled.device_pointer,
      after_attn.device_pointer, combined_moe.device_pointer,
      output.device_pointer, kHidden, stream);
  ++metrics.native_pointwise_launches;
  if (synchronize) {
    check_hip(hipStreamSynchronize(stream),
              "hipStreamSynchronize native linear layer");
  }
  metrics.wall_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started)
                        .count();
  return metrics;
}

}  // namespace aima
