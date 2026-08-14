// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_linear_prefill.h"

#include "aima/bf16_gemm.h"
#include "aima/native_decode_bindings.h"
#include "aima/native_pointwise.h"
#include "aima/native_prefill_gemm_plans.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima {
namespace {

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kLinearQkv = 8192;
constexpr std::size_t kLinearKey = 2048;
constexpr std::size_t kLinearValue = 4096;
constexpr std::size_t kLinearHeads = 32;
constexpr std::size_t kStateElements = 32 * 128 * 128;
constexpr std::size_t kLinearConvChannels = 8192;
constexpr std::size_t kLinearConvStateTokens = 3;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorName(status) + " (" +
                             hipGetErrorString(status) + ")");
  }
}

const NativeTensorView& require_weight(const NativeWeightStore& weights,
                                       const std::string& name,
                                       std::uint64_t bytes) {
  const NativeTensorView* view = weights.find(name);
  if (view == nullptr || view->device_pointer == nullptr ||
      view->payload_bytes != bytes) {
    throw std::runtime_error("native linear prefill weight mismatch: " + name);
  }
  return *view;
}

void* require_workspace(const NativePrefillWorkspace& workspace,
                        const char* name, std::uint64_t bytes) {
  const NativePrefillWorkspaceView* view = workspace.find(name);
  if (view == nullptr || view->device_pointer == nullptr ||
      view->payload_bytes < bytes) {
    throw std::runtime_error(
        "native linear prefill workspace mismatch: " + std::string(name));
  }
  return view->device_pointer;
}

void require_symbol(const std::vector<PreparedDecodeInvocation>& launches,
                    std::size_t sequence, const char* symbol) {
  if (sequence >= launches.size() || launches[sequence].launch == nullptr ||
      std::string(launches[sequence].launch->symbol) != symbol) {
    throw std::runtime_error(
        "native linear prefill schedule symbol mismatch at sequence " +
        std::to_string(sequence));
  }
}

std::size_t find_linear_layer_base(
    const std::vector<PreparedDecodeInvocation>& launches,
    std::size_t layer_index, std::size_t expected_launches) {
  if (layer_index >= 40 || layer_index % 4 == 3) {
    throw std::invalid_argument(
        "native linear prefill requires a linear-attention layer index");
  }
  for (std::size_t sequence = 0; sequence < launches.size(); ++sequence) {
    const auto* launch = launches[sequence].launch;
    if (launch != nullptr &&
        launch->layer_index == static_cast<std::int16_t>(layer_index)) {
      if (sequence + expected_launches > launches.size()) break;
      for (std::size_t offset = 0; offset < expected_launches; ++offset) {
        if (launches[sequence + offset].launch == nullptr ||
            launches[sequence + offset].launch->layer_index !=
                static_cast<std::int16_t>(layer_index)) {
          throw std::runtime_error(
              "native linear prefill layer schedule is not contiguous");
        }
      }
      return sequence;
    }
  }
  throw std::runtime_error("native linear prefill layer is absent from schedule");
}

bool gate(const NativeOracleComparison& value) {
  return value.finite_elements == value.elements &&
         value.relative_l2_error <= 0.002 &&
         value.cosine_similarity >= 0.999;
}

__global__ void repair_padded_conv_state_kernel(
    const __hip_bfloat16* raw, const __hip_bfloat16* initial_state,
    __hip_bfloat16* state, std::size_t active_tokens,
    std::size_t raw_row_stride) {
  const std::size_t channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= kLinearConvChannels) return;
  const std::size_t retained_initial =
      active_tokens >= kLinearConvStateTokens
          ? 0
          : kLinearConvStateTokens - active_tokens;
  for (std::size_t slot = 0; slot < kLinearConvStateTokens; ++slot) {
    if (slot < retained_initial) {
      state[channel * kLinearConvStateTokens + slot] =
          initial_state[channel * kLinearConvStateTokens + active_tokens +
                        slot];
    } else {
      const std::size_t token = slot - retained_initial +
                                (active_tokens >= kLinearConvStateTokens
                                     ? active_tokens - kLinearConvStateTokens
                                     : 0);
      state[channel * kLinearConvStateTokens + slot] =
          raw[token * raw_row_stride + channel];
    }
  }
}

__global__ void exact_linear_b_projection_kernel(
    const __hip_bfloat16* input, const __hip_bfloat16* weight,
    __hip_bfloat16* output, std::size_t token_count) {
  const std::size_t token = blockIdx.x;
  const std::size_t column = blockIdx.y;
  const unsigned lane = threadIdx.x;
  if (token >= token_count || column >= kLinearHeads) return;

  float sum = 0.0f;
  for (unsigned hidden = lane; hidden < kHidden; hidden += 32) {
    sum = fmaf(
        __bfloat162float(input[token * kHidden + hidden]),
        __bfloat162float(weight[column * kHidden + hidden]), sum);
  }
#pragma unroll
  for (unsigned offset = 16; offset != 0; offset >>= 1) {
    sum += __shfl_down(sum, offset, 32);
  }
  if (lane == 0) {
    output[token * kLinearHeads + column] = __float2bfloat16(sum);
  }
}

void launch_exact_linear_b_projection(
    const void* input, const void* weight, void* output,
    std::size_t token_count) {
  hipLaunchKernelGGL(
      exact_linear_b_projection_kernel,
      dim3(static_cast<unsigned>(token_count), kLinearHeads), dim3(32), 0,
      nullptr, static_cast<const __hip_bfloat16*>(input),
      static_cast<const __hip_bfloat16*>(weight),
      static_cast<__hip_bfloat16*>(output), token_count);
  check_hip(hipGetLastError(), "exact_linear_b_projection_kernel");
}

__global__ void extract_compact_linear_ab_kernel(
    const __hip_bfloat16* compact, __hip_bfloat16* a,
    __hip_bfloat16* b, std::size_t token_count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t elements = token_count * 2 * kLinearHeads;
  if (index >= elements) return;
  const std::size_t token = index / (2 * kLinearHeads);
  const std::size_t column = index % (2 * kLinearHeads);
  if (column < kLinearHeads) {
    a[token * kLinearHeads + column] = compact[index];
  } else {
    b[token * kLinearHeads + column - kLinearHeads] = compact[index];
  }
}

void launch_extract_compact_linear_ab(
    const void* compact, void* a, void* b, std::size_t token_count) {
  const std::size_t elements = token_count * 2 * kLinearHeads;
  hipLaunchKernelGGL(
      extract_compact_linear_ab_kernel,
      dim3(static_cast<unsigned>((elements + 255) / 256)), dim3(256), 0,
      nullptr, static_cast<const __hip_bfloat16*>(compact),
      static_cast<__hip_bfloat16*>(a), static_cast<__hip_bfloat16*>(b),
      token_count);
  check_hip(hipGetLastError(), "extract_compact_linear_ab_kernel");
}

}  // namespace

NativeLinearPrefillOracleResult
probe_native_q8192_linear_prefill_layer0_oracle(
    const std::filesystem::path& oracle_dir,
    const NativeWeightStore& weights,
    const NativePrefillWorkspace& workspace,
    NativePrefillInvocations& invocations,
    NativeDecodeExecutor& executor,
    const NativeLinearPrefillOracleOptions& options) {
  if (!weights.loaded() || !workspace.built() || !executor.loaded() ||
      (invocations.launches().size() != 431 &&
       invocations.launches().size() != 401)) {
    throw std::invalid_argument(
        "native linear prefill oracle requires complete resident owners");
  }
  if (!options.collect_oracle_comparisons &&
      (options.run_output_projection_diagnostic ||
       !options.boundary_oracle_dir.empty())) {
    throw std::invalid_argument(
        "native linear prefill execution cannot run oracle diagnostics");
  }
  const std::size_t bucket_tokens = workspace.context_tokens();
  const std::size_t tokens =
      options.active_tokens == 0 ? bucket_tokens : options.active_tokens;
  const std::size_t comparison_tokens =
      options.comparison_tokens == 0 ? tokens : options.comparison_tokens;
  const std::size_t exact_b_tokens = options.exact_b_projection_tokens;
  Bf16GemmPlan* logical_ab_gemm_plan = options.logical_ab_gemm_plan;
  const void* logical_ab_weight = options.logical_ab_weight;
  void* logical_ab_output = options.logical_ab_output;
  const bool logical_ab_enabled = logical_ab_gemm_plan != nullptr;
  if (bucket_tokens == 0 || bucket_tokens > 262144 || tokens == 0 ||
      tokens > bucket_tokens ||
      comparison_tokens == 0 || comparison_tokens > tokens ||
      exact_b_tokens > tokens ||
      (logical_ab_enabled != (logical_ab_weight != nullptr)) ||
      (logical_ab_enabled != (logical_ab_output != nullptr)) ||
      (logical_ab_enabled &&
       (logical_ab_gemm_plan->m() != comparison_tokens ||
        logical_ab_gemm_plan->n() != 2 * kLinearHeads ||
        logical_ab_gemm_plan->k() != kHidden)) ||
      (tokens != bucket_tokens && options.collect_oracle_comparisons) ||
      (bucket_tokens != 8192 && options.collect_oracle_comparisons)) {
    throw std::invalid_argument(
        "native linear prefill context or oracle mode is unsupported");
  }
  const auto& launches = invocations.launches();
  const bool q8192_schedule = bucket_tokens == 8192;
  const bool q1024_official_fla = bucket_tokens == 1024;
  const bool split_projection_tail =
      !q8192_schedule && launches.size() > 1 &&
      launches[1].launch != nullptr &&
      std::string(launches[1].launch->symbol) ==
          "_causal_conv1d_fwd_kernel";
  const bool split_projections = q8192_schedule || split_projection_tail;
  if (logical_ab_enabled && split_projections) {
    throw std::invalid_argument(
        "native logical A/B GEMM requires the direct q1024 projection");
  }
  const std::size_t linear_launches = q8192_schedule ? 13 : 12;
  const std::size_t attention_launches = q8192_schedule ? 11 : 10;
  const std::size_t base = find_linear_layer_base(
      launches, options.layer_index, linear_launches);
  const std::array<const char*, 11> q8192_symbols = {
      "triton_rmsnorm_kernel",
      "_causal_conv1d_fwd_kernel",
      "_fused_post_conv_kernel",
      "chunk_local_cumsum_scalar_kernel",
      "chunk_scaled_dot_kkt_fwd_kernel",
      "merge_16x16_to_32x32_inverse_kernel",
      "recompute_w_u_fwd_kernel",
      "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
      "chunk_fwd_kernel_o",
      "triton_linear_gated_norm_from_invstd_kernel",
      "triton_prefill_fused_add_rmsnorm_kernel",
  };
  const std::array<const char*, 10> q32768_symbols = {
      "triton_rmsnorm_kernel",
      "triton_prefill_direct_conv_kernel",
      "_fused_post_conv_kernel",
      "chunk_local_cumsum_scalar_kernel",
      "chunk_scaled_dot_kkt_fwd_kernel",
      "merge_16x16_to_32x32_inverse_kernel",
      "recompute_w_u_fwd_kernel",
      "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
      "chunk_fwd_kernel_o",
      "triton_prefill_fused_add_rmsnorm_kernel",
  };
  const std::array<const char*, 10> split_tail_symbols = {
      "triton_rmsnorm_kernel",
      "_causal_conv1d_fwd_kernel",
      "_fused_post_conv_kernel",
      "chunk_local_cumsum_scalar_kernel",
      "chunk_scaled_dot_kkt_fwd_kernel",
      "merge_16x16_to_32x32_inverse_kernel",
      "recompute_w_u_fwd_kernel",
      "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
      "chunk_fwd_kernel_o",
      "triton_prefill_fused_add_rmsnorm_kernel",
  };
  for (std::size_t sequence = 0; sequence < attention_launches; ++sequence) {
    const char* symbol = q8192_schedule
                             ? q8192_symbols[sequence]
                             : (split_projection_tail
                                    ? split_tail_symbols[sequence]
                                    : q32768_symbols[sequence]);
    if (q1024_official_fla && sequence == 5) {
      symbol = "merge_16x16_to_64x64_inverse_kernel";
    }
    require_symbol(launches, base + sequence, symbol);
  }

  NativeLinearPrefillOracleResult result;
  if (options.decode_state_workspace != nullptr) {
    if (!options.decode_state_workspace->built()) {
      throw std::invalid_argument(
          "native linear prefill decode state owner is not built");
    }
    const std::string layer = std::to_string(options.layer_index);
    const NativeDecodeWorkspaceView* conv =
        options.decode_state_workspace->find(
            "linear_attention_initial_conv_states." + layer);
    const NativeDecodeWorkspaceView* ssm =
        options.decode_state_workspace->find(
            "linear_attention_initial_ssm_states_vllm." + layer);
    if (conv == nullptr || conv->device_pointer == nullptr ||
        conv->payload_bytes != 8192ULL * 3ULL * sizeof(std::uint16_t) ||
        conv->dtype != DecodeTensorDtype::kBfloat16 ||
        ssm == nullptr || ssm->device_pointer == nullptr ||
        ssm->payload_bytes != kStateElements * sizeof(float) ||
        ssm->dtype != DecodeTensorDtype::kFloat32) {
      throw std::runtime_error(
          "native linear prefill decode state geometry mismatch");
    }
    if (split_projections) {
      invocations.rebind_tensor(base + 1, "initial_states_ptr",
                                conv->device_pointer);
    } else {
      invocations.rebind_tensor(base + 1, "state_in", conv->device_pointer);
      invocations.rebind_tensor(base + 1, "state_out", conv->device_pointer);
    }
    invocations.rebind_tensor(base + 7, "ht", ssm->device_pointer);
    result.layer.resident_state_direct_bindings = split_projections ? 2 : 3;
    result.layer.resident_state_payload_bytes =
        conv->payload_bytes + ssm->payload_bytes;
  }

  const std::string prefix = "model.language_model.layers." +
                             std::to_string(options.layer_index) + ".";
  const NativeTensorView& input_norm_weight = require_weight(
      weights, prefix + "input_layernorm.weight", 4096ULL);
  const NativeTensorView& post_attention_norm_weight = require_weight(
      weights, prefix + "post_attention_layernorm.weight", 4096ULL);
  const NativeTensorView* qkv_weight = nullptr;
  const NativeTensorView* z_weight = nullptr;
  const NativeTensorView* a_weight = nullptr;
  const NativeTensorView* b_weight = nullptr;
  const NativeDecodeBindingView* fused_weight = nullptr;
  if (split_projections) {
    qkv_weight = &require_weight(
        weights, prefix + "linear_attn.in_proj_qkv.weight", 33554432ULL);
    z_weight = &require_weight(
        weights, prefix + "linear_attn.in_proj_z.weight", 16777216ULL);
    a_weight = &require_weight(
        weights, prefix + "linear_attn.in_proj_a.weight", 131072ULL);
    b_weight = &require_weight(
        weights, prefix + "linear_attn.in_proj_b.weight", 131072ULL);
  } else {
    if (options.bindings == nullptr) {
      throw std::invalid_argument(
          "native direct linear prefill requires derived bindings");
    }
    const std::string binding = "layer_weights." +
        std::to_string(options.layer_index) +
        ".tensors.linear_input_proj_fused_t";
    fused_weight = options.bindings->find(binding);
    if (fused_weight == nullptr || fused_weight->device_pointer == nullptr ||
        fused_weight->payload_bytes != 50593792ULL ||
        fused_weight->dtype != DecodeTensorDtype::kBfloat16) {
      throw std::runtime_error(
          "native direct fused linear input weight is missing");
    }
    if (q1024_official_fla && exact_b_tokens != 0) {
      b_weight = &require_weight(
          weights, prefix + "linear_attn.in_proj_b.weight", 131072ULL);
    }
  }
  const NativeTensorView& output_weight = require_weight(
      weights, prefix + "linear_attn.out_proj.weight", 16777216ULL);
  const NativeTensorView& linear_norm_weight = require_weight(
      weights, prefix + "linear_attn.norm.weight", 256ULL);

  void* x = invocations.tensor_pointer(base, "x");
  void* h1 = invocations.tensor_pointer(base, "out");
  void* qkv = invocations.tensor_pointer(
      base + 1, split_projections ? "x_ptr" : "raw");
  void* z = nullptr;
  if (q8192_schedule) {
    z = invocations.tensor_pointer(base + 9, "z");
  } else if (split_projection_tail) {
    z = require_workspace(
        workspace, "native.tail_linear_gate",
        tokens * kLinearValue * sizeof(std::uint16_t));
  } else {
    z = static_cast<unsigned char*>(qkv) +
        8192 * sizeof(std::uint16_t);
  }
  void* a = invocations.tensor_pointer(base + 2, "a_ptr");
  void* b = invocations.tensor_pointer(base + 2, "b_ptr");
  void* final_state = invocations.tensor_pointer(base + 7, "ht");
  void* core = invocations.tensor_pointer(
      base + (q8192_schedule ? 9 : 8),
      q8192_schedule ? "core" : "o");
  void* invstd = q8192_schedule
                     ? invocations.tensor_pointer(base + 9, "invstd")
                     : nullptr;
  void* gated = q8192_schedule
                    ? invocations.tensor_pointer(base + 9, "out")
                    : core;
  const std::size_t residual_offset = q8192_schedule ? 10 : 9;
  void* attention_output =
      invocations.tensor_pointer(base + residual_offset, "residual");
  void* after_attention =
      invocations.tensor_pointer(base + residual_offset, "residual_out");
  void* h2 =
      invocations.tensor_pointer(base + residual_offset, "norm_out");
  // PyTorch's allocator reused the same captured address for the long-lived Z
  // gate and the short-lived FLA v_new tensor.  Pointer-only trace
  // canonicalization cannot express overlapping semantic lifetimes.  Reuse
  // the now-dead pre-convolution V storage for v_new so Z remains live through
  // gated norm without increasing the workspace allocation.
  if (q8192_schedule) {
    void* v_new_storage = invocations.tensor_pointer(base + 2, "v_ptr");
    invocations.rebind_tensor(base + 7, "v_new", v_new_storage);
    invocations.rebind_tensor(base + 8, "v", v_new_storage);
    result.layer.semantic_alias_rebindings = 2;
  }

  const std::filesystem::path fixture =
      oracle_dir.empty() ? std::filesystem::path{}
                         : std::filesystem::absolute(oracle_dir);
  const auto oracle_file = [&fixture, &options](const char* label) {
    return find_native_oracle_tensor_file(
        fixture, options.oracle_label_prefix + label);
  };
  const std::filesystem::path boundary_fixture =
      options.boundary_oracle_dir.empty()
          ? std::filesystem::path{}
          : std::filesystem::absolute(options.boundary_oracle_dir);
  const auto boundary_file = [&boundary_fixture, &options](const char* label) {
    return find_native_oracle_tensor_file(
        boundary_fixture, options.boundary_oracle_label_prefix + label);
  };
  const std::filesystem::path tail_fixture =
      options.tail_oracle_dir.empty()
          ? std::filesystem::path{}
          : std::filesystem::absolute(options.tail_oracle_dir);
  const auto tail_file = [&tail_fixture, &options](const char* label) {
    return find_native_oracle_tensor_file(
        tail_fixture, options.tail_oracle_label_prefix + label);
  };
  const auto optional_tail_file = [&tail_fixture, &options](const char* label) {
    return find_native_oracle_tensor_file_if_present(
        tail_fixture, options.tail_oracle_label_prefix + label);
  };
  const std::filesystem::path sequence_fixture =
      options.sequence_oracle_dir.empty()
          ? std::filesystem::path{}
          : std::filesystem::absolute(options.sequence_oracle_dir);
  const auto optional_sequence_file = [&sequence_fixture, &options](
                                          const char* label) {
    return find_native_oracle_tensor_file_if_present(
        sequence_fixture, options.sequence_oracle_label_prefix + label);
  };
  const auto compare_optional_sequence_typed = [&] (
      const char* comparison_label, const char* dtype, const void* pointer,
      std::size_t elements_per_token, const char* oracle_label) {
    if (sequence_fixture.empty()) return;
    const std::filesystem::path expected =
        optional_sequence_file(oracle_label);
    if (expected.empty()) return;
    const std::size_t element_bytes =
        std::string_view(dtype) == "float32" ? sizeof(float)
                                               : sizeof(std::uint16_t);
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        options.sequence_oracle_label_prefix + comparison_label,
        dtype, pointer,
        comparison_tokens * elements_per_token * element_bytes,
        expected));
  };
  const auto compare_optional_sequence = [&] (
      const char* comparison_label, const void* pointer,
      std::size_t elements_per_token, const char* oracle_label) {
    compare_optional_sequence_typed(
        comparison_label, "bfloat16", pointer, elements_per_token,
        oracle_label);
  };
  const auto compare_optional_sequence_storage = [&] (
      const char* comparison_label, const char* dtype, const void* pointer,
      const char* oracle_label) {
    if (sequence_fixture.empty()) return;
    const std::filesystem::path expected =
        optional_sequence_file(oracle_label);
    if (expected.empty()) return;
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        options.sequence_oracle_label_prefix + comparison_label,
        dtype, pointer, std::filesystem::file_size(expected), expected));
  };
  const auto compare_tail = [&](const char* comparison_label,
                                const void* pointer,
                                const char* oracle_label) {
    const auto* bytes = static_cast<const unsigned char*>(pointer);
    return compare_native_oracle_tensor(
        options.tail_oracle_label_prefix + comparison_label,
        "bfloat16",
        bytes + (comparison_tokens - 1) * kHidden * sizeof(std::uint16_t),
        kHidden * sizeof(std::uint16_t), tail_file(oracle_label));
  };
  const auto compare_optional_stage_tail = [&] (
      const char* comparison_label, const char* dtype, const void* pointer,
      std::size_t row_elements, std::size_t row_stride_elements,
      const char* oracle_label) {
    if (tail_fixture.empty()) return;
    const std::filesystem::path expected = optional_tail_file(oracle_label);
    if (expected.empty()) return;
    const std::size_t element_bytes =
        std::string(dtype) == "float32" ? sizeof(float)
                                         : sizeof(std::uint16_t);
    const auto* bytes = static_cast<const unsigned char*>(pointer);
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        options.tail_oracle_label_prefix + comparison_label, dtype,
        bytes + (comparison_tokens - 1) * row_stride_elements * element_bytes,
        row_elements * element_bytes, expected));
  };
  const std::string residual_launch_prefix =
      q8192_schedule ? "launch-010-" : "launch-009-";
  result.layer.layer_index = options.layer_index;
  result.layer.tokens = tokens;
  result.layer_input_seeded = options.seed_layer_input;
  if (options.seed_layer_input) {
    if (options.layer_input_oracle_label.empty()) {
      throw std::invalid_argument(
          "native linear prefill layer-input oracle label is empty");
    }
    if (comparison_tokens != tokens) {
      check_hip(hipMemset(x, 0,
                          tokens * kHidden * sizeof(std::uint16_t)),
                "hipMemset padded seeded linear input");
    }
    result.seed_bytes += seed_native_oracle_tensor(
        oracle_file(options.layer_input_oracle_label.c_str()), x,
        comparison_tokens * kHidden * sizeof(std::uint16_t));
    ++result.seed_tensors;
  }

  if (split_projections) {
    // Re-establish the captured cold single-sequence convolution metadata.
    const std::int32_t cache_index = 0;
    const std::uint8_t has_initial_state =
        options.has_initial_state ? 1 : 0;
    const std::array<std::int32_t, 2> query_start = {
        0, static_cast<std::int32_t>(tokens)};
    std::array<std::int32_t, 1024> batch{};
    std::array<std::int32_t, 1024> chunk_offset{};
    for (std::size_t index = 0; index < chunk_offset.size(); ++index) {
      chunk_offset[index] = static_cast<std::int32_t>(index);
    }
    check_hip(hipMemcpy(invocations.tensor_pointer(base + 1, "cache_indices_ptr"),
                        &cache_index, sizeof(cache_index),
                        hipMemcpyHostToDevice),
              "hipMemcpy prefill cache index");
    check_hip(hipMemcpy(
                  invocations.tensor_pointer(base + 1,
                                             "has_initial_states_ptr"),
                  &has_initial_state, sizeof(has_initial_state),
                  hipMemcpyHostToDevice),
              "hipMemcpy prefill initial-state flag");
    check_hip(hipMemcpy(
                  invocations.tensor_pointer(base + 1,
                                             "query_start_loc_ptr"),
                  query_start.data(), sizeof(query_start),
                  hipMemcpyHostToDevice),
              "hipMemcpy prefill query starts");
    check_hip(hipMemcpy(invocations.tensor_pointer(base + 1, "batch_ptr"),
                        batch.data(), sizeof(batch), hipMemcpyHostToDevice),
              "hipMemcpy prefill batch map");
    check_hip(hipMemcpy(
                  invocations.tensor_pointer(base + 1,
                                             "token_chunk_offset_ptr"),
                  chunk_offset.data(), sizeof(chunk_offset),
                  hipMemcpyHostToDevice),
              "hipMemcpy prefill chunk offsets");
  }
  if (!options.has_initial_state) {
    check_hip(hipMemset(invocations.tensor_pointer(
                            base + 1,
                          split_projections ? "initial_states_ptr" : "state_in"),
                        0,
                        8192 * 3 * sizeof(std::uint16_t)),
              "hipMemset prefill initial convolution state");
  }
  std::unique_ptr<NativeQ8192PrefillGemmPlans> local_gemm_plans;
  NativeQ8192PrefillGemmPlans* gemm_plans = options.gemm_plans;
  if (gemm_plans == nullptr) {
    local_gemm_plans =
        std::make_unique<NativeQ8192PrefillGemmPlans>(tokens);
    gemm_plans = local_gemm_plans.get();
  }
  if (gemm_plans->token_count() != tokens) {
    throw std::invalid_argument(
        "native linear prefill GEMM context mismatch");
  }
  Bf16GemmPlan* qkv_plan = nullptr;
  Bf16GemmPlan* z_plan = nullptr;
  Bf16GemmPlan* ab_plan = nullptr;
  Bf16GemmPlan* fused_plan = nullptr;
  if (split_projections) {
    qkv_plan = &gemm_plans->linear_qkv();
    z_plan = &gemm_plans->linear_z();
    ab_plan = &gemm_plans->linear_ab();
  } else {
    fused_plan = &gemm_plans->linear_fused_input();
  }
  Bf16GemmPlan& output_plan = gemm_plans->linear_output();
  result.layer.gemm_workspace_bytes = output_plan.workspace_bytes() +
      (split_projections
           ? qkv_plan->workspace_bytes() + z_plan->workspace_bytes() +
                 ab_plan->workspace_bytes()
           : fused_plan->workspace_bytes());

  const auto started = std::chrono::steady_clock::now();
  if (q1024_official_fla) {
    launch_prefill_rmsnorm_2048(
        x, input_norm_weight.device_pointer, h1, tokens);
    ++result.layer.native_pointwise_launches;
  } else {
    executor.launch(launches[base]);
  }
  NativeOracleComparison tail_input_comparison;
  NativeOracleComparison tail_input_norm_comparison;
  NativeOracleComparison tail_attention_output_comparison;
  NativeOracleComparison tail_post_attention_comparison;
  NativeOracleComparison tail_post_attention_norm_comparison;
  if (!tail_fixture.empty()) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native linear input-norm tail oracle");
    tail_input_comparison = compare_tail(
        "input_last_token", x, "launch-000-x");
    tail_input_norm_comparison = compare_tail(
        "input_norm_last_token", h1, "launch-000-out");
    result.boundary_comparisons.push_back(tail_input_comparison);
    result.boundary_comparisons.push_back(tail_input_norm_comparison);
  }
  compare_optional_sequence(
      "input_norm_full_sequence", h1, kHidden, "launch-000-out");
  NativeOracleComparison attention_input_comparison;
  if (options.collect_oracle_comparisons) {
    attention_input_comparison = compare_native_oracle_tensor(
          "attention_input", "bfloat16", h1,
          tokens * kHidden * sizeof(std::uint16_t),
          oracle_file("launch-000-out"));
  }
  if (split_projections) {
    qkv_plan->launch(h1, qkv_weight->device_pointer, qkv);
    z_plan->launch(h1, z_weight->device_pointer, z);
    ab_plan->launch(h1, a_weight->device_pointer, a);
    ab_plan->launch(h1, b_weight->device_pointer, b);
    result.layer.dense_gemm_launches += 4;
  } else {
    fused_plan->launch(h1, fused_weight->device_pointer, qkv);
    launch_extract_linear_ab_fused(qkv, a, b, tokens);
    ++result.layer.dense_gemm_launches;
    ++result.layer.native_pointwise_launches;
    if (logical_ab_enabled) {
      logical_ab_gemm_plan->launch(
          h1, logical_ab_weight, logical_ab_output);
      launch_extract_compact_linear_ab(
          logical_ab_output, a, b, comparison_tokens);
      ++result.layer.dense_gemm_launches;
      ++result.layer.native_pointwise_launches;
    }
    if (q1024_official_fla && exact_b_tokens != 0) {
      // vLLM projects B/A as a separate 64-column merged linear.  At this
      // gfx1151 padded shape, hipBLASLt changes BF16 B roundings at the
      // blocking video boundary. Preserve the large fused GEMM for QKVZ/A
      // and overwrite only the logical VL prefix with a stable reduction.
      launch_exact_linear_b_projection(
          h1, b_weight->device_pointer, b, exact_b_tokens);
      ++result.layer.native_pointwise_launches;
    }
  }
  if (split_projections) {
    compare_optional_sequence(
        "linear_projection_qkv_full_sequence", qkv, kLinearQkv,
        "diagnostic-qkv");
    compare_optional_sequence(
        "linear_projection_z_full_sequence", z, kLinearValue,
        "diagnostic-z");
  } else {
    compare_optional_sequence(
        "linear_projection_fused_full_sequence", qkv, 12352,
        "diagnostic-fused-input");
  }
  compare_optional_sequence(
      "linear_projection_a_full_sequence", a, kLinearHeads,
      "diagnostic-a");
  compare_optional_sequence(
      "linear_projection_b_full_sequence", b, kLinearHeads,
      "diagnostic-b");
  if (!split_projections) {
    compare_optional_stage_tail(
        "linear_projection_qkv_last_token", "bfloat16", qkv,
        kLinearQkv, 12352, "launch-001-raw");
    compare_optional_stage_tail(
        "linear_projection_a_last_token", "bfloat16", a,
        kLinearHeads, kLinearHeads, "launch-002-a_ptr");
    compare_optional_stage_tail(
        "linear_projection_b_last_token", "bfloat16", b,
        kLinearHeads, kLinearHeads, "launch-002-b_ptr");
  }
  if (options.collect_oracle_comparisons || !boundary_fixture.empty()) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native linear input projections");
  }
  if (!boundary_fixture.empty()) {
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_qkv_projection", "bfloat16", qkv,
        tokens * kLinearQkv * sizeof(std::uint16_t),
        boundary_file("launch-001-x_ptr")));
  }
  NativeOracleComparison projected_gate_comparison;
  if (options.collect_oracle_comparisons) {
    projected_gate_comparison = compare_native_oracle_tensor(
          "projected_linear_gate", "bfloat16", z,
          tokens * kLinearValue * sizeof(std::uint16_t),
          oracle_file("launch-009-z"));
  }
  executor.launch(launches[base + 1]);
  compare_optional_sequence(
      "linear_convolution_full_sequence",
      invocations.tensor_pointer(
          base + 1, split_projection_tail ? "o_ptr" : "out"),
      kLinearQkv, "diagnostic-conv");
  if (tokens != 8192) {
    compare_optional_stage_tail(
        "linear_convolution_last_token", "bfloat16",
        invocations.tensor_pointer(
            base + 1, split_projection_tail ? "o_ptr" : "out"),
        kLinearQkv,
        kLinearQkv, "launch-001-out");
  }
  if (!boundary_fixture.empty()) {
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_convolution", "bfloat16",
        invocations.tensor_pointer(base + 1, "o_ptr"),
        tokens * kLinearQkv * sizeof(std::uint16_t),
        boundary_file("launch-001-o_ptr")));
  }
  executor.launch(launches[base + 2]);
  compare_optional_sequence(
      "fla_q_full_sequence", invocations.tensor_pointer(base + 2, "q_ptr"),
      kLinearKey, "diagnostic-q");
  compare_optional_sequence(
      "fla_k_full_sequence", invocations.tensor_pointer(base + 2, "k_ptr"),
      kLinearKey, "diagnostic-k");
  compare_optional_sequence(
      "fla_v_full_sequence", invocations.tensor_pointer(base + 2, "v_ptr"),
      kLinearValue, "diagnostic-v");
  compare_optional_sequence_typed(
      "fla_g_full_sequence", "float32",
      invocations.tensor_pointer(base + 2, "g_ptr"), kLinearHeads,
      "diagnostic-g");
  compare_optional_sequence_typed(
      "fla_beta_full_sequence", "float32",
      invocations.tensor_pointer(base + 2, "beta_ptr"), kLinearHeads,
      "diagnostic-beta");
  if (tokens != 8192) {
    compare_optional_stage_tail(
        "fla_q_last_token", "bfloat16",
        invocations.tensor_pointer(base + 2, "q_ptr"), kLinearKey,
        kLinearKey, "launch-002-q_ptr");
    compare_optional_stage_tail(
        "fla_k_last_token", "bfloat16",
        invocations.tensor_pointer(base + 2, "k_ptr"), kLinearKey,
        kLinearKey, "launch-002-k_ptr");
    compare_optional_stage_tail(
        "fla_v_last_token", "bfloat16",
        invocations.tensor_pointer(base + 2, "v_ptr"), kLinearValue,
        kLinearValue, "launch-002-v_ptr");
    compare_optional_stage_tail(
        "fla_g_last_token", "float32",
        invocations.tensor_pointer(base + 2, "g_ptr"), kLinearHeads,
        kLinearHeads, "launch-002-g_ptr");
    compare_optional_stage_tail(
        "fla_beta_last_token", "float32",
        invocations.tensor_pointer(base + 2, "beta_ptr"), kLinearHeads,
        kLinearHeads, "launch-002-beta_ptr");
  }
  if (!boundary_fixture.empty()) {
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_q", "bfloat16", invocations.tensor_pointer(base + 2, "q_ptr"),
        tokens * kLinearKey * sizeof(std::uint16_t),
        boundary_file("launch-002-q_ptr")));
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_k", "bfloat16", invocations.tensor_pointer(base + 2, "k_ptr"),
        tokens * kLinearKey * sizeof(std::uint16_t),
        boundary_file("launch-002-k_ptr")));
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_v", "bfloat16", invocations.tensor_pointer(base + 2, "v_ptr"),
        tokens * kLinearValue * sizeof(std::uint16_t),
        boundary_file("launch-002-v_ptr")));
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_g", "float32", invocations.tensor_pointer(base + 2, "g_ptr"),
        tokens * kLinearHeads * sizeof(float),
        boundary_file("launch-002-g_ptr")));
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_beta", "float32",
        invocations.tensor_pointer(base + 2, "beta_ptr"),
        tokens * kLinearHeads * sizeof(float),
        boundary_file("launch-002-beta_ptr")));
  }
  executor.launch(launches[base + 3]);
  compare_optional_sequence_storage(
      "fla_g_cumsum_storage", "float32",
      invocations.tensor_pointer(base + 3, "o"), "diagnostic-g-cumsum");
  if (tokens != 8192) {
    compare_optional_stage_tail(
        "fla_g_cumsum_last_token", "float32",
        invocations.tensor_pointer(base + 3, "o"), kLinearHeads,
        kLinearHeads, "launch-003-o");
  }
  if (!boundary_fixture.empty()) {
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_g_cumsum", "float32",
        invocations.tensor_pointer(base + 3, "o"),
        tokens * kLinearHeads * sizeof(float),
        boundary_file("launch-003-o")));
  }
  executor.launch(launches[base + 4]);
  compare_optional_sequence_storage(
      "fla_chunk_matrix_storage", "float32",
      invocations.tensor_pointer(base + 4, "A"), "diagnostic-chunk-matrix");
  if (tokens != 8192) {
    compare_optional_stage_tail(
        "fla_chunk_matrix_last_token", "float32",
        invocations.tensor_pointer(base + 4, "A"), 32 * 32, 32 * 32,
        "launch-004-A");
  }
  if (!boundary_fixture.empty()) {
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_chunk_matrix", "float32",
        invocations.tensor_pointer(base + 4, "A"),
        invocations.tensor_storage_bytes(base + 4, "A"),
        boundary_file("launch-004-A")));
  }
  // The inverse kernel writes only the structurally valid part of Ai.  This
  // storage survives as unrelated BF16 data across layer iterations, so the
  // untouched cells must be re-established at their captured zero value
  // immediately before the kernel consumes A and produces Ai.
  const std::size_t inverse_scratch_bytes =
      invocations.tensor_storage_bytes(base + 5, "Ai");
  check_hip(hipMemset(invocations.tensor_pointer(base + 5, "Ai"), 0,
                      inverse_scratch_bytes),
            "hipMemset prefill FLA chunk-inverse scratch");
  result.layer.state_scratch_zero_operations = 1;
  result.layer.state_scratch_zero_bytes =
      inverse_scratch_bytes;
  executor.launch(launches[base + 5]);
  compare_optional_sequence_storage(
      "fla_chunk_matrix_inverse_storage", "bfloat16",
      invocations.tensor_pointer(base + 5, "Ai"),
      "diagnostic-chunk-matrix-inverse");
  if (tokens != 8192) {
    compare_optional_stage_tail(
        "fla_chunk_matrix_inverse_last_token", "bfloat16",
        invocations.tensor_pointer(base + 5, "Ai"), 32 * 32, 32 * 32,
        "launch-005-Ai");
  }
  if (!boundary_fixture.empty()) {
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_chunk_matrix_inverse", "bfloat16",
        invocations.tensor_pointer(base + 5, "Ai"), 16777216,
        boundary_file("launch-005-Ai")));
  }
  executor.launch(launches[base + 6]);
  compare_optional_sequence_storage(
      "fla_w_storage", "bfloat16",
      invocations.tensor_pointer(base + 6, "w"), "diagnostic-w");
  compare_optional_sequence_storage(
      "fla_u_storage", "bfloat16",
      invocations.tensor_pointer(base + 6, "u"), "diagnostic-u");
  if (tokens != 8192) {
    compare_optional_stage_tail(
        "fla_w_last_token", "bfloat16",
        invocations.tensor_pointer(base + 6, "w"), kLinearValue,
        kLinearValue, "launch-006-w");
    compare_optional_stage_tail(
        "fla_u_last_token", "bfloat16",
        invocations.tensor_pointer(base + 6, "u"), kLinearValue,
        kLinearValue, "launch-006-u");
  }
  if (!boundary_fixture.empty()) {
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_w", "bfloat16", invocations.tensor_pointer(base + 6, "w"),
        tokens * kLinearValue * sizeof(std::uint16_t),
        boundary_file("launch-006-w")));
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_u", "bfloat16", invocations.tensor_pointer(base + 6, "u"),
        tokens * kLinearValue * sizeof(std::uint16_t),
        boundary_file("launch-006-u")));
  }
  // The q32768 capture reuses transient.1 for the now-dead A projection and
  // the zero initial chunk state.  Clear at the semantic lifetime boundary,
  // after post-conv has consumed A/B, instead of before the projections.
  void* initial_state = invocations.tensor_pointer(base + 7, "h0");
  if (options.has_initial_state) {
    check_hip(hipMemcpy(initial_state, final_state,
                        kStateElements * sizeof(float),
                        hipMemcpyDeviceToDevice),
              "hipMemcpy prefill carried SSM state");
  } else {
    check_hip(hipMemset(initial_state, 0,
                        kStateElements * sizeof(float)),
              "hipMemset prefill initial SSM state");
  }
  executor.launch(launches[base + 7]);
  compare_optional_sequence_storage(
      "fla_chunk_state_storage", "bfloat16",
      invocations.tensor_pointer(base + 7, "h"), "diagnostic-chunk-state");
  compare_optional_sequence_storage(
      "fla_v_new_storage", "bfloat16",
      invocations.tensor_pointer(base + 7, "v_new"), "diagnostic-v-new");
  compare_optional_sequence_storage(
      "fla_final_state_storage", "float32", final_state,
      "diagnostic-final-state");
  if (tokens != 8192 && !tail_fixture.empty()) {
    compare_optional_stage_tail(
        "fla_v_new_last_token", "bfloat16",
        invocations.tensor_pointer(base + 7, "v_new"), kLinearValue,
        kLinearValue, "launch-007-v_new");
    const std::filesystem::path chunk_state_expected =
        optional_tail_file("launch-007-h");
    if (!chunk_state_expected.empty()) {
      const std::size_t chunk_count = tokens / 32;
      const auto* final_chunk = static_cast<const unsigned char*>(
          invocations.tensor_pointer(base + 7, "h")) +
          (chunk_count - 1) * kStateElements * sizeof(std::uint16_t);
      result.boundary_comparisons.push_back(compare_native_oracle_tensor(
          options.tail_oracle_label_prefix +
              "fla_chunk_state_last_chunk",
          "bfloat16", final_chunk,
          kStateElements * sizeof(std::uint16_t), chunk_state_expected));
    }
    const std::filesystem::path final_state_expected =
        optional_tail_file("launch-007-ht");
    if (!final_state_expected.empty()) {
      result.boundary_comparisons.push_back(compare_native_oracle_tensor(
          options.tail_oracle_label_prefix + "fla_final_state", "float32",
          final_state, kStateElements * sizeof(float), final_state_expected));
    }
  }
  if (!boundary_fixture.empty()) {
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_v_new", "bfloat16",
        invocations.tensor_pointer(base + 7, "v_new"),
        tokens * kLinearValue * sizeof(std::uint16_t),
        boundary_file("launch-007-v_new")));
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_chunk_state", "bfloat16",
        invocations.tensor_pointer(base + 7, "h"), 268435456,
        boundary_file("launch-007-h")));
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_final_state", "float32", final_state,
        kStateElements * sizeof(float),
        boundary_file("launch-007-ht")));
  }
  executor.launch(launches[base + 8]);
  compare_optional_sequence(
      "fla_core_full_sequence", core, kLinearValue, "launch-008-o");
  if (tokens != 8192) {
    compare_optional_stage_tail(
        "fla_core_last_token", "bfloat16", core, kLinearValue,
        kLinearValue, "launch-008-o");
  }
  if (!boundary_fixture.empty()) {
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        "fla_core_output", "bfloat16", core,
        tokens * kLinearValue * sizeof(std::uint16_t),
        boundary_file("launch-008-o")));
  }
  if (tokens == 32768 && !sequence_fixture.empty()) {
    const std::filesystem::path variance_expected =
        optional_sequence_file("return-linear_attention-variance");
    if (!variance_expected.empty()) {
      const std::size_t rows = tokens * kLinearHeads;
      void* diagnostic_variance = nullptr;
      check_hip(hipMalloc(&diagnostic_variance, rows * sizeof(float)),
                "hipMalloc linear variance diagnostic");
      try {
        launch_bf16_rowwise_variance_128_pytorch(
            core, diagnostic_variance, rows);
        result.boundary_comparisons.push_back(compare_native_oracle_tensor(
            options.sequence_oracle_label_prefix +
                "linear_variance_full_sequence",
            "float32", diagnostic_variance, rows * sizeof(float),
            variance_expected));
      } catch (...) {
        (void)hipFree(diagnostic_variance);
        throw;
      }
      check_hip(hipFree(diagnostic_variance),
                "hipFree linear variance diagnostic");
    }
  }
  if (q8192_schedule) {
    launch_bf16_rowwise_invstd_128(core, invstd, tokens * kLinearHeads);
    ++result.layer.native_pointwise_launches;
    executor.launch(launches[base + 9]);
  } else if (split_projection_tail) {
    launch_linear_gated_norm_separate(
        core, z, linear_norm_weight.device_pointer, gated, tokens);
    ++result.layer.native_pointwise_launches;
    compare_optional_stage_tail(
        "linear_gated_output_last_token", "bfloat16", gated,
        kLinearValue, kLinearValue,
        "return-linear_attention-gated_out");
  } else {
    launch_linear_gated_norm_fused(
        core, qkv, linear_norm_weight.device_pointer, gated, tokens);
    ++result.layer.native_pointwise_launches;
    compare_optional_stage_tail(
        "linear_gated_output_last_token", "bfloat16", gated,
        kLinearValue, kLinearValue,
        "return-linear_attention-gated_out");
  }
  compare_optional_sequence(
      "linear_gated_output_full_sequence", gated, kLinearValue,
      "return-linear_attention-gated_out");
  if (options.collect_oracle_comparisons) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native linear gated norm");
  }
  NativeOracleComparison core_comparison;
  NativeOracleComparison gate_comparison;
  NativeOracleComparison norm_weight_comparison;
  NativeOracleComparison invstd_comparison;
  NativeOracleComparison gated_comparison;
  if (options.collect_oracle_comparisons) {
    core_comparison = compare_native_oracle_tensor(
          "linear_core", "bfloat16", core,
          tokens * kLinearValue * sizeof(std::uint16_t),
          oracle_file("launch-009-core"));
    gate_comparison = compare_native_oracle_tensor(
          "linear_gate", "bfloat16", z,
          tokens * kLinearValue * sizeof(std::uint16_t),
          oracle_file("launch-009-z"));
    norm_weight_comparison = compare_native_oracle_tensor(
          "linear_norm_weight", "bfloat16",
          linear_norm_weight.device_pointer, 256,
          oracle_file("launch-009-weight"));
    invstd_comparison = compare_native_oracle_tensor(
          "linear_invstd", "float32", invstd,
          tokens * kLinearHeads * sizeof(float),
          oracle_file("launch-009-invstd"));
    gated_comparison = compare_native_oracle_tensor(
          "linear_gated_output", "bfloat16", gated,
          tokens * kLinearValue * sizeof(std::uint16_t),
          oracle_file("launch-009-out"));
  }
  output_plan.launch(gated, output_weight.device_pointer, attention_output);
  ++result.layer.dense_gemm_launches;
  compare_optional_sequence(
      "attention_output_full_sequence", attention_output, kHidden,
      "return-linear_attention-output");
  if (!tail_fixture.empty()) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native linear output tail oracle");
    tail_attention_output_comparison = compare_tail(
        "attention_output_last_token", attention_output,
        (residual_launch_prefix + "residual").c_str());
  }
  if (q1024_official_fla) {
    launch_prefill_add_rmsnorm_2048(
        attention_output, x, post_attention_norm_weight.device_pointer,
        after_attention, h2, tokens);
    ++result.layer.native_pointwise_launches;
  } else {
    executor.launch(launches[base + residual_offset]);
  }
  compare_optional_sequence(
      "post_attention_full_sequence", after_attention, kHidden,
      (residual_launch_prefix + "residual_out").c_str());
  compare_optional_sequence(
      "post_attention_norm_full_sequence", h2, kHidden,
      (residual_launch_prefix + "norm_out").c_str());
  result.layer.aot_launches =
      attention_launches - (q1024_official_fla ? 2 : 0);
  if (options.collect_oracle_comparisons || !tail_fixture.empty()) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native linear prefill oracle");
  }
  if (!tail_fixture.empty()) {
    tail_post_attention_comparison = compare_tail(
        "post_attention_last_token", after_attention,
        (residual_launch_prefix + "residual_out").c_str());
    tail_post_attention_norm_comparison = compare_tail(
        "post_attention_norm_last_token", h2,
        (residual_launch_prefix + "norm_out").c_str());
  }
  result.layer.wall_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();

  if (!tail_fixture.empty()) {
    result.boundary_comparisons.push_back(
        std::move(tail_attention_output_comparison));
    result.boundary_comparisons.push_back(
        std::move(tail_post_attention_comparison));
    result.boundary_comparisons.push_back(
        std::move(tail_post_attention_norm_comparison));
  }

  if (!options.collect_oracle_comparisons) {
    result.all_finite = true;
    return result;
  }

  const std::size_t hidden_bytes =
      tokens * kHidden * sizeof(std::uint16_t);
  const NativeOracleComparison final_state_comparison =
      compare_native_oracle_tensor(
          "linear_final_state", "float32", final_state,
          kStateElements * sizeof(float), oracle_file("launch-007-ht"));
  const NativeOracleComparison attention_output_comparison =
      compare_native_oracle_tensor(
          "attention_output", "bfloat16", attention_output, hidden_bytes,
          oracle_file("return-linear_attention-output"));
  const NativeOracleComparison residual_comparison =
      compare_native_oracle_tensor(
          "post_attention_residual", "bfloat16", after_attention,
          hidden_bytes, oracle_file("launch-010-residual_out"));
  const NativeOracleComparison post_norm_comparison =
      compare_native_oracle_tensor(
          "post_attention_norm", "bfloat16", h2, hidden_bytes,
          oracle_file("launch-010-norm_out"));

  result.comparisons = {
      compare_native_oracle_tensor(
          "layer_input", "bfloat16", x, hidden_bytes,
          oracle_file("launch-000-x")),
      compare_native_oracle_tensor(
          "input_norm_weight", "bfloat16",
          input_norm_weight.device_pointer, 4096,
          oracle_file("launch-000-weight")),
      attention_input_comparison,
      projected_gate_comparison,
      core_comparison,
      gate_comparison,
      norm_weight_comparison,
      invstd_comparison,
      gated_comparison,
      final_state_comparison,
      attention_output_comparison,
      residual_comparison,
      post_norm_comparison,
  };
  // Isolate the dense output projection from all preceding kernels.  This
  // oracle-seeded launch is diagnostic only.  It must be skipped by a
  // multi-layer chain so the production boundary remains untouched.
  if (options.run_output_projection_diagnostic) {
    result.seed_bytes += seed_native_oracle_tensor(
        oracle_file("launch-009-out"), gated,
        tokens * kLinearValue * sizeof(std::uint16_t));
    ++result.seed_tensors;
    output_plan.launch(gated, output_weight.device_pointer, attention_output);
    ++result.layer.diagnostic_gemm_launches;
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize seeded output projection diagnostic");
    result.comparisons.push_back(compare_native_oracle_tensor(
        "seeded_output_projection", "bfloat16", attention_output,
        hidden_bytes, oracle_file("return-linear_attention-output")));
    result.output_projection_diagnostic_ran = true;
  }
  result.all_finite = true;
  for (const NativeOracleComparison& comparison : result.comparisons) {
    result.all_finite =
        result.all_finite && comparison.finite_elements == comparison.elements;
  }
  result.final_state_gate_passed = gate(result.comparisons[9]);
  result.attention_output_gate_passed = gate(result.comparisons[10]);
  result.post_attention_gate_passed =
      gate(result.comparisons[11]) && gate(result.comparisons[12]);
  return result;
}

NativeLinearPrefillStateRepairMetrics repair_native_linear_prefill_padded_state(
    const NativePrefillWorkspace& workspace,
    NativePrefillInvocations& invocations, NativeDecodeExecutor& executor,
    std::size_t layer_index, std::size_t active_tokens,
    const void* initial_conv_state) {
  const std::size_t bucket_tokens = workspace.context_tokens();
  if (!workspace.built() || !executor.loaded() ||
      initial_conv_state == nullptr || active_tokens == 0 ||
      active_tokens >= bucket_tokens) {
    throw std::invalid_argument(
        "native padded linear-state repair geometry is invalid");
  }
  const auto& launches = invocations.launches();
  const bool q8192_schedule = bucket_tokens == 8192;
  const bool split_projection_tail =
      !q8192_schedule && launches.size() > 1 && launches[1].launch != nullptr &&
      std::string(launches[1].launch->symbol) == "_causal_conv1d_fwd_kernel";
  const bool split_projections = q8192_schedule || split_projection_tail;
  const std::size_t base =
      find_linear_layer_base(launches, layer_index, q8192_schedule ? 13 : 12);
  const void* raw =
      invocations.tensor_pointer(base + 1, split_projections ? "x_ptr" : "raw");
  void* conv_state = invocations.tensor_pointer(
      base + 1, split_projections ? "initial_states_ptr" : "state_out");
  if (raw == nullptr || conv_state == nullptr) {
    throw std::runtime_error(
        "native padded linear-state repair owners are missing");
  }
  constexpr unsigned kThreads = 256;
  hipLaunchKernelGGL(repair_padded_conv_state_kernel,
                     dim3((kLinearConvChannels + kThreads - 1) / kThreads),
                     dim3(kThreads), 0, nullptr,
                     static_cast<const __hip_bfloat16*>(raw),
                     static_cast<const __hip_bfloat16*>(initial_conv_state),
                     static_cast<__hip_bfloat16*>(conv_state), active_tokens,
                     split_projections ? kLinearConvChannels : 12352);
  check_hip(hipGetLastError(), "repair_padded_conv_state_kernel");

  const std::size_t recurrent_sequence = base + 7;
  invocations.set_int32_argument(recurrent_sequence, "T",
                                 static_cast<std::int32_t>(active_tokens));
  try {
    executor.launch(invocations.launches()[recurrent_sequence]);
  } catch (...) {
    invocations.set_int32_argument(recurrent_sequence, "T",
                                   static_cast<std::int32_t>(bucket_tokens));
    throw;
  }
  invocations.set_int32_argument(recurrent_sequence, "T",
                                 static_cast<std::int32_t>(bucket_tokens));

  NativeLinearPrefillStateRepairMetrics metrics;
  metrics.aot_launches = 1;
  metrics.native_pointwise_launches = 1;
  return metrics;
}

}  // namespace aima
