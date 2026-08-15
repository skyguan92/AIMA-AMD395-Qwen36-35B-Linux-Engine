// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_linear_layer.h"

#include "aima/bf16_wvsplitk.h"
#include "aima/native_pointwise.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima {
namespace {

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kLinearQkv = 8192;
constexpr std::size_t kLinearValue = 4096;
constexpr std::size_t kLinearGate = 32;
constexpr std::size_t kSharedIntermediate = 512;
constexpr std::size_t kRecurrentStateBytes =
    32ULL * 128ULL * 128ULL * sizeof(float);
constexpr char kPackedRecurrentKernelHash[] =
    "361b24af7b3fc502598ffb5fd1e191c9b82afc437361f92f4056bf8772a960dc";
constexpr char kCausalConvKernelHash[] =
    "ab71972380fed224052336c248656eb49e8d2ccd89acc4bebbee193e2c6a699c";
constexpr char kLinearGatedNormKernelHash[] =
    "2c40422c776225912e71c6cd74fb90ea37001e24b57e6cc135af84c048a791db";
constexpr AotLaunchConfig kPackedRecurrentLaunchConfig{
    4, 32, 1, 1, 32, 64};
constexpr AotLaunchConfig kCausalConvLaunchConfig{
    1, 32, 1, 4, 32, 2048};
constexpr AotLaunchConfig kLinearGatedNormLaunchConfig{
    32, 1, 1, 1, 32, 0};
constexpr float kLinearAttentionScale = 0.08838834764831845f;

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

const NativeDecodeWorkspaceView& require_typed_workspace(
    const NativeDecodeWorkspace& workspace, const std::string& name,
    std::uint64_t expected_bytes, DecodeTensorDtype dtype) {
  const NativeDecodeWorkspaceView* view = workspace.find(name);
  if (view == nullptr || view->device_pointer == nullptr ||
      view->payload_bytes != expected_bytes || view->dtype != dtype) {
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

void launch_packed_recurrent(
    std::size_t layer_index, const NativeDecodeWorkspace& workspace,
    const NativeDecodeInvocations& invocations, NativeDecodeExecutor& executor,
    hipStream_t stream) {
  if (layer_index >= 40 || layer_index % 4 == 3) {
    throw std::invalid_argument("packed recurrent layer index is invalid");
  }
  const std::size_t schedule_index = layer_index * 10 + 2;
  const std::size_t state_index = layer_index - layer_index / 4 + 1;
  const auto& state_base = require_typed_workspace(
      workspace, "native.linear.packed_ssm_state_base", kRecurrentStateBytes,
      DecodeTensorDtype::kFloat32);
  const auto& state_indices = require_typed_workspace(
      workspace, "native.linear.packed_ssm_state_indices",
      40 * sizeof(std::int32_t), DecodeTensorDtype::kInt32);
  void* canonical_state = invocations.tensor_pointer(schedule_index, "h0");
  void* expected_state =
      static_cast<unsigned char*>(state_base.device_pointer) +
      state_index * kRecurrentStateBytes;
  if (canonical_state != expected_state) {
    throw std::runtime_error(
        "native packed recurrent state layout is not contiguous");
  }

  // The frozen schedule records the prior out-of-place FLA kernel. The current
  // vLLM implementation consumes the same projection slices but normalizes Q/K
  // and rounds sigmoid(beta) in this packed, in-place kernel. Keep the schedule
  // as historical evidence and replace only this launch at execution time.
  void* mixed_qkv = invocations.tensor_pointer(schedule_index, "q");
  void* a = invocations.tensor_pointer(schedule_index, "a");
  void* b = invocations.tensor_pointer(schedule_index, "b");
  void* a_log = invocations.tensor_pointer(schedule_index, "A_log");
  void* dt_bias = invocations.tensor_pointer(schedule_index, "dt_bias");
  void* output = invocations.tensor_pointer(schedule_index, "o");
  void* initial_state = state_base.device_pointer;
  void* final_state = state_base.device_pointer;
  void* selected_state_index =
      static_cast<unsigned char*>(state_indices.device_pointer) +
      layer_index * sizeof(std::int32_t);
  float scale = kLinearAttentionScale;
  const std::vector<void*> parameters = {
      &mixed_qkv, &a,          &b,           &a_log, &dt_bias,
      &output,    &initial_state, &final_state, &selected_state_index,
      &scale,
  };
  executor.launch_embedded(kPackedRecurrentKernelHash,
                           kPackedRecurrentLaunchConfig, parameters, stream);
}

void launch_current_causal_conv(
    void* mixed_qkv, void* conv_weight, void* conv_state,
    void* state_index, NativeDecodeExecutor& executor, hipStream_t stream) {
  void* output = mixed_qkv;
  std::int32_t batch = 1;
  const std::vector<void*> parameters = {
      &mixed_qkv,
      &conv_weight,
      &conv_state,
      &state_index,
      &output,
      &batch,
  };
  executor.launch_embedded(kCausalConvKernelHash, kCausalConvLaunchConfig,
                           parameters, stream);
}

void launch_current_linear_gated_norm(
    void* core, void* gate, void* weight, void* output, void* rstd,
    NativeDecodeExecutor& executor, hipStream_t stream) {
  std::int32_t stride_x_row = 128;
  std::int32_t stride_y_row = 128;
  std::int32_t stride_z_row = 128;
  std::int32_t rows = 32;
  float epsilon = 1.0e-6f;
  const std::vector<void*> parameters = {
      &core, &output, &weight, &gate, &rstd,
      &stride_x_row, &stride_y_row, &stride_z_row, &rows, &epsilon,
  };
  executor.launch_embedded(kLinearGatedNormKernelHash,
                           kLinearGatedNormLaunchConfig, parameters, stream);
}

void observe_boundary(const NativeDecodeLinearLayer0Observer* observer,
                      const char* name, const void* device_tensor,
                      std::uint64_t tensor_bytes, DecodeTensorDtype dtype) {
  if (observer != nullptr) {
    (*observer)(name, device_tensor, tensor_bytes, dtype);
  }
}

}  // namespace

NativeLinearLayerMetrics run_native_linear_layer(
    std::size_t layer_index, const NativeWeightStore& weights,
    const NativeDecodeWorkspace& workspace,
    const NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, int cu_count, void* stream_value,
    bool synchronize, const NativeDecodeLinearLayer0Observer* observer) {
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
  void* input_norm = invocations.tensor_pointer(base, "out");
  void* projection = invocations.tensor_pointer(base + 1, "out");
  void* conv_state_before =
      invocations.tensor_pointer(base + 1, "state_in");
  void* conv_state_after =
      invocations.tensor_pointer(base + 1, "state_out");
  void* z_projection = invocations.tensor_pointer(base + 3, "z");
  void* a_projection = invocations.tensor_pointer(base + 2, "a");
  void* b_projection = invocations.tensor_pointer(base + 2, "b");
  void* recurrent_state = invocations.tensor_pointer(base + 2, "h0");
  void* recurrent_output = invocations.tensor_pointer(base + 2, "o");
  const auto& packed_state_indices = require_typed_workspace(
      workspace, "native.linear.packed_ssm_state_indices",
      40 * sizeof(std::int32_t), DecodeTensorDtype::kInt32);
  void* direct_conv_state_index =
      static_cast<unsigned char*>(packed_state_indices.device_pointer) +
      3 * sizeof(std::int32_t);
  auto* projection_bytes = static_cast<unsigned char*>(projection);
  void* projected_qkv = projection_bytes;
  void* projected_z =
      projection_bytes + kLinearQkv * sizeof(__hip_bfloat16);
  void* projected_a =
      projection_bytes + (kLinearQkv + kLinearValue) *
                             sizeof(__hip_bfloat16);
  void* projected_b =
      projection_bytes + (kLinearQkv + kLinearValue + kLinearGate) *
                             sizeof(__hip_bfloat16);
  if (projected_z != z_projection || projected_a != a_projection ||
      projected_b != b_projection) {
    throw std::runtime_error(
        "native linear projection slice layout changed");
  }
  const std::string prefix =
      "model.language_model.layers." + suffix;
  const auto& output_weight = require_weight(
      weights, prefix + ".linear_attn.out_proj.weight",
      kHidden * kLinearValue * sizeof(__hip_bfloat16));
  const auto& qkv_weight = require_weight(
      weights, prefix + ".linear_attn.in_proj_qkv.weight",
      kLinearQkv * kHidden * sizeof(__hip_bfloat16));
  const auto& z_weight = require_weight(
      weights, prefix + ".linear_attn.in_proj_z.weight",
      kLinearValue * kHidden * sizeof(__hip_bfloat16));
  const auto& a_weight = require_weight(
      weights, prefix + ".linear_attn.in_proj_a.weight",
      kLinearGate * kHidden * sizeof(__hip_bfloat16));
  const auto& b_weight = require_weight(
      weights, prefix + ".linear_attn.in_proj_b.weight",
      kLinearGate * kHidden * sizeof(__hip_bfloat16));
  const auto& conv_weight = require_weight(
      weights, prefix + ".linear_attn.conv1d.weight",
      kLinearQkv * 4 * sizeof(__hip_bfloat16));
  const auto& linear_norm_weight = require_weight(
      weights, prefix + ".linear_attn.norm.weight",
      128 * sizeof(__hip_bfloat16));
  const auto& shared_down_weight = require_weight(
      weights, prefix + ".mlp.shared_expert.down_proj.weight",
      kHidden * kSharedIntermediate * sizeof(__hip_bfloat16));

  const auto started = std::chrono::steady_clock::now();
  hipStream_t stream = static_cast<hipStream_t>(stream_value);
  NativeLinearLayerMetrics metrics;
  metrics.layer_index = layer_index;
  executor.launch(launches[base], stream);
  observe_boundary(observer, "input_norm", input_norm,
                   kHidden * sizeof(__hip_bfloat16),
                   DecodeTensorDtype::kBfloat16);
  observe_boundary(observer, "conv_state_before", conv_state_before,
                   kLinearQkv * 3ULL * sizeof(__hip_bfloat16),
                   DecodeTensorDtype::kBfloat16);
  // Current vLLM evaluates the four source projections with its wvSplitK
  // provider, then runs causal_conv1d_update in-place. The captured historical
  // fused launch uses a different convolution reduction and is retained only
  // as schedule evidence.
  launch_bf16_wvsplitk(qkv_weight.device_pointer, input_norm, nullptr,
                       projected_qkv, kLinearQkv, kHidden, cu_count, stream);
  launch_bf16_wvsplitk(z_weight.device_pointer, input_norm, nullptr,
                       projected_z, kLinearValue, kHidden, cu_count, stream);
  launch_bf16_wvsplitk(a_weight.device_pointer, input_norm, nullptr,
                       projected_a, kLinearGate, kHidden, cu_count, stream);
  launch_bf16_wvsplitk(b_weight.device_pointer, input_norm, nullptr,
                       projected_b, kLinearGate, kHidden, cu_count, stream);
  metrics.native_projection_launches += 4;
  observe_boundary(observer, "qkv_projection", projected_qkv,
                   kLinearQkv * sizeof(__hip_bfloat16),
                   DecodeTensorDtype::kBfloat16);
  observe_boundary(observer, "z_projection", z_projection,
                   kLinearValue * sizeof(__hip_bfloat16),
                   DecodeTensorDtype::kBfloat16);
  observe_boundary(observer, "a_projection", a_projection,
                   kLinearGate * sizeof(__hip_bfloat16),
                   DecodeTensorDtype::kBfloat16);
  observe_boundary(observer, "b_projection", b_projection,
                   kLinearGate * sizeof(__hip_bfloat16),
                   DecodeTensorDtype::kBfloat16);
  check_hip(hipMemcpyAsync(
                conv_state_after, conv_state_before,
                kLinearQkv * 3ULL * sizeof(__hip_bfloat16),
                hipMemcpyDeviceToDevice, stream),
            "hipMemcpyAsync native causal-conv state");
  launch_current_causal_conv(projected_qkv, conv_weight.device_pointer,
                             conv_state_after, direct_conv_state_index,
                             executor, stream);
  observe_boundary(observer, "post_conv_mixed_qkv", projected_qkv,
                   kLinearQkv * sizeof(__hip_bfloat16),
                   DecodeTensorDtype::kBfloat16);
  observe_boundary(observer, "conv_state_after", conv_state_after,
                   kLinearQkv * 3ULL * sizeof(__hip_bfloat16),
                   DecodeTensorDtype::kBfloat16);
  observe_boundary(observer, "recurrent_state_before", recurrent_state,
                   kRecurrentStateBytes, DecodeTensorDtype::kFloat32);
  launch_packed_recurrent(layer_index, workspace, invocations, executor, stream);
  observe_boundary(observer, "recurrent_output", recurrent_output,
                   kLinearValue * sizeof(__hip_bfloat16),
                   DecodeTensorDtype::kBfloat16);
  observe_boundary(observer, "recurrent_state_after", recurrent_state,
                   kRecurrentStateBytes, DecodeTensorDtype::kFloat32);
  // Current vLLM uses one row per Triton program for decode RMSNormGated.
  // The attention-output scratch is still dead here and supplies the 32-value
  // FP32 Rstd side output without growing the resident workspace.
  launch_current_linear_gated_norm(
      recurrent_output, z_projection, linear_norm_weight.device_pointer,
      gated.device_pointer, attention_output.device_pointer, executor, stream);
  observe_boundary(observer, "gated_norm", gated.device_pointer,
                   kLinearValue * sizeof(__hip_bfloat16),
                   DecodeTensorDtype::kBfloat16);
  metrics.aot_launches += 4;
  launch_bf16_wvsplitk(
      output_weight.device_pointer, gated.device_pointer, nullptr,
      attention_output.device_pointer, kHidden, kLinearValue, cu_count, stream);
  ++metrics.native_projection_launches;
  observe_boundary(observer, "attention_output",
                   attention_output.device_pointer,
                   kHidden * sizeof(__hip_bfloat16),
                   DecodeTensorDtype::kBfloat16);
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
