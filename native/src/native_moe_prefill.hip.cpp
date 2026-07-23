// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_moe_prefill.h"

#include "aima/bf16_gemm.h"
#include "aima/native_pointwise.h"
#include "aima/native_prefill_gemm_plans.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace aima {
namespace {

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kExperts = 256;
constexpr std::size_t kTopK = 8;
constexpr std::size_t kSharedIntermediate = 512;
constexpr std::size_t kExpertGateUp = 1024;
constexpr std::size_t kMoeBlock = 32;
constexpr unsigned kThreads = 256;

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
    throw std::runtime_error("native MoE prefill weight mismatch: " + name);
  }
  return *view;
}

const NativePrefillWorkspaceView& require_workspace(
    const NativePrefillWorkspace& workspace, const std::string& name,
    std::uint64_t bytes) {
  const NativePrefillWorkspaceView* view = workspace.find(name);
  if (view == nullptr || view->device_pointer == nullptr ||
      view->payload_bytes < bytes) {
    throw std::runtime_error("native MoE prefill workspace mismatch: " + name);
  }
  return *view;
}

void* require_invocation_tensor(const NativePrefillInvocations& invocations,
                                std::size_t sequence, const char* name,
                                std::uint64_t bytes) {
  void* pointer = invocations.tensor_pointer(sequence, name);
  if (pointer == nullptr ||
      invocations.tensor_storage_bytes(sequence, name) < bytes) {
    throw std::runtime_error(
        "native MoE prefill invocation scratch mismatch: " +
        std::string(name));
  }
  return pointer;
}

void require_symbol(const std::vector<PreparedDecodeInvocation>& launches,
                    std::size_t sequence, const char* symbol) {
  if (sequence >= launches.size() || launches[sequence].launch == nullptr ||
      std::string(launches[sequence].launch->symbol) != symbol) {
    throw std::runtime_error(
        "native MoE prefill schedule symbol mismatch at sequence " +
        std::to_string(sequence));
  }
}

struct PrefillLayerSchedule {
  std::size_t base = 0;
  std::size_t launch_count = 0;
  std::size_t moe_offset = 0;
};

PrefillLayerSchedule find_prefill_layer_schedule(
    const std::vector<PreparedDecodeInvocation>& launches,
    std::size_t layer_index) {
  if (layer_index >= 40) {
    throw std::invalid_argument("native MoE prefill layer index is out of range");
  }
  for (std::size_t sequence = 0; sequence < launches.size(); ++sequence) {
    const auto* launch = launches[sequence].launch;
    if (launch == nullptr ||
        launch->layer_index != static_cast<std::int16_t>(layer_index)) {
      continue;
    }
    std::size_t count = 0;
    while (sequence + count < launches.size() &&
           launches[sequence + count].launch != nullptr &&
           launches[sequence + count].launch->layer_index ==
               static_cast<std::int16_t>(layer_index)) {
      ++count;
    }
    const bool valid = layer_index % 4 == 3
                           ? count == 4
                           : (count == 13 || count == 12);
    if (!valid) {
      throw std::runtime_error(
          "native MoE prefill layer schedule length is unsupported");
    }
    return {sequence, count, count - 2};
  }
  throw std::runtime_error("native MoE prefill layer is absent from schedule");
}

void* layer_output_pointer(
    const std::vector<PreparedDecodeInvocation>& launches,
    const NativePrefillWorkspace& workspace,
    const NativePrefillInvocations& invocations, std::size_t layer_index,
    std::size_t tokens) {
  if (layer_index + 1 < 40) {
    const PrefillLayerSchedule next =
        find_prefill_layer_schedule(launches, layer_index + 1);
    return invocations.tensor_pointer(next.base, "x");
  }
  const std::size_t hidden_bytes =
      tokens * kHidden * sizeof(std::uint16_t);
  if (tokens == 8192) {
    const NativePrefillWorkspaceView* terminal =
        workspace.find("transient.31");
    if (terminal != nullptr && terminal->device_pointer != nullptr &&
        terminal->payload_bytes >= hidden_bytes) {
      return terminal->device_pointer;
    }
  } else {
    const PrefillLayerSchedule scratch =
        find_prefill_layer_schedule(launches, 0);
    return require_invocation_tensor(
        invocations, scratch.base + 8, "o", hidden_bytes);
  }
  throw std::runtime_error(
      "native MoE prefill terminal output workspace is absent");
}

std::string launch_label(std::size_t sequence, const char* argument) {
  std::string result = "launch-";
  if (sequence < 100) result += '0';
  if (sequence < 10) result += '0';
  result += std::to_string(sequence);
  result += '-';
  result += argument;
  return result;
}

bool gate(const NativeOracleComparison& value) {
  return value.finite_elements == value.elements &&
         value.relative_l2_error <= 0.002 &&
         value.cosine_similarity >= 0.999;
}

std::size_t count_exact_int64_row_sets(
    const void* actual_device, const std::filesystem::path& expected_path,
    std::size_t rows, std::size_t width) {
  if (actual_device == nullptr || rows == 0 || width != kTopK) {
    throw std::invalid_argument(
        "native MoE router-set comparison geometry is invalid");
  }
  const std::size_t elements = rows * width;
  std::vector<std::int64_t> expected(elements);
  std::ifstream stream(expected_path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() !=
                     static_cast<std::streamoff>(elements * sizeof(std::int64_t))) {
    throw std::runtime_error(
        "native MoE router-set oracle byte count mismatch");
  }
  stream.seekg(0, std::ios::beg);
  if (!stream.read(reinterpret_cast<char*>(expected.data()),
                   static_cast<std::streamsize>(elements *
                                                sizeof(std::int64_t)))) {
    throw std::runtime_error("cannot read native MoE router-set oracle");
  }
  std::vector<std::int64_t> actual(elements);
  check_hip(hipMemcpy(actual.data(), actual_device,
                      elements * sizeof(std::int64_t),
                      hipMemcpyDeviceToHost),
            "hipMemcpy native MoE router-set comparison");
  std::size_t exact_rows = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    std::array<std::int64_t, kTopK> expected_set{};
    std::array<std::int64_t, kTopK> actual_set{};
    std::copy_n(expected.data() + row * width, width, expected_set.begin());
    std::copy_n(actual.data() + row * width, width, actual_set.begin());
    std::sort(expected_set.begin(), expected_set.end());
    std::sort(actual_set.begin(), actual_set.end());
    if (expected_set == actual_set) ++exact_rows;
  }
  return exact_rows;
}

__global__ void shared_silu_multiply_batched_kernel(
    const __hip_bfloat16* gate, const __hip_bfloat16* up,
    __hip_bfloat16* output, std::size_t elements) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float gate_value = __bfloat162float(gate[index]);
  const float up_value = __bfloat162float(up[index]);
  const float silu = gate_value / (1.0f + expf(-gate_value));
  output[index] = __float2bfloat16(silu * up_value);
}

__global__ void shared_sigmoid_scale_batched_kernel(
    const __hip_bfloat16* gate, const __hip_bfloat16* down,
    __hip_bfloat16* output, std::size_t elements) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const std::size_t token = index / kHidden;
  const float sigmoid =
      1.0f / (1.0f + expf(-__bfloat162float(gate[token])));
  const __hip_bfloat16 gate_bf16 = __float2bfloat16(sigmoid);
  output[index] = __float2bfloat16(
      __bfloat162float(gate_bf16) * __bfloat162float(down[index]));
}

__global__ void router_topk8_softmax_256_kernel(
    const __hip_bfloat16* logits, float* scores, std::int64_t* indices_i64,
    std::int32_t* indices_i32, __hip_bfloat16* weights) {
  if (threadIdx.x != 0) return;
  const std::size_t token = blockIdx.x;
  const __hip_bfloat16* row = logits + token * kExperts;
  float ranked_scores[kTopK];
  int ranked_indices[kTopK];
  for (std::size_t rank = 0; rank < kTopK; ++rank) {
    float best = -std::numeric_limits<float>::infinity();
    int best_index = -1;
    for (int expert = 0; expert < static_cast<int>(kExperts); ++expert) {
      bool used = false;
#pragma unroll
      for (std::size_t prior = 0; prior < rank; ++prior) {
        used = used || ranked_indices[prior] == expert;
      }
      if (used) continue;
      const float value = __bfloat162float(row[expert]);
      if (value > best) {
        best = value;
        best_index = expert;
      }
    }
    ranked_scores[rank] = best;
    ranked_indices[rank] = best_index;
  }

  // PyTorch's ROCm topk first gathers every value strictly above the kth
  // threshold in source-index order, then fills the remaining slots with
  // threshold ties in source-index order.  Its in-kernel bitonic sort swaps
  // equal keys on descending comparators, yielding a deterministic (but not
  // stable) tie order.  Reproduce both stages so selected experts remain
  // bit-exact with the frozen engine instead of merely matching topk values.
  const float threshold = ranked_scores[kTopK - 1];
  float selected_scores[kTopK];
  int selected_indices[kTopK];
  std::size_t selected = 0;
  for (int expert = 0; expert < static_cast<int>(kExperts); ++expert) {
    const float value = __bfloat162float(row[expert]);
    if (value > threshold) {
      selected_scores[selected] = value;
      selected_indices[selected] = expert;
      ++selected;
    }
  }
  for (int expert = 0;
       expert < static_cast<int>(kExperts) && selected < kTopK; ++expert) {
    const float value = __bfloat162float(row[expert]);
    if (value == threshold) {
      selected_scores[selected] = value;
      selected_indices[selected] = expert;
      ++selected;
    }
  }
  for (int width = 2; width <= static_cast<int>(kTopK); width *= 2) {
    for (int stride = width / 2; stride > 0; stride /= 2) {
      for (int left = 0; left < static_cast<int>(kTopK); ++left) {
        const int right = left ^ stride;
        if (right <= left) continue;
        const bool ascending = (left & width) != 0;
        const bool swap =
            (selected_scores[left] > selected_scores[right]) == ascending;
        if (swap) {
          const float score = selected_scores[left];
          selected_scores[left] = selected_scores[right];
          selected_scores[right] = score;
          const int index = selected_indices[left];
          selected_indices[left] = selected_indices[right];
          selected_indices[right] = index;
        }
      }
    }
  }
  float denominator = 0.0f;
#pragma unroll
  for (std::size_t rank = 0; rank < kTopK; ++rank) {
    denominator += expf(selected_scores[rank] - selected_scores[0]);
  }
  const std::size_t base = token * kTopK;
#pragma unroll
  for (std::size_t rank = 0; rank < kTopK; ++rank) {
    scores[base + rank] = selected_scores[rank];
    indices_i64[base + rank] = selected_indices[rank];
    indices_i32[base + rank] = selected_indices[rank];
    const float probability =
        expf(selected_scores[rank] - selected_scores[0]) / denominator;
    weights[base + rank] = __float2bfloat16(probability);
  }
}

__global__ void moe_align_block32_256_kernel(
    const std::int32_t* topk_ids, std::int32_t* sorted_token_ids,
    std::int32_t* expert_ids, std::int32_t* num_tokens_post_padded,
    std::size_t routed_rows, std::size_t sorted_capacity,
    std::size_t expert_block_capacity) {
  __shared__ std::int32_t counts[kExperts];
  __shared__ std::int32_t block_offsets[kExperts + 1];
  __shared__ std::int32_t write_positions[kExperts];
  const unsigned thread = threadIdx.x;
  counts[thread] = 0;
  for (std::size_t index = thread; index < sorted_capacity;
       index += blockDim.x) {
    sorted_token_ids[index] = static_cast<std::int32_t>(routed_rows);
  }
  for (std::size_t index = thread; index < expert_block_capacity;
       index += blockDim.x) {
    expert_ids[index] = -1;
  }
  __syncthreads();

  const std::size_t per_thread =
      (routed_rows + kExperts - 1) / kExperts;
  const std::size_t begin = thread * per_thread;
  const std::size_t candidate_end = begin + per_thread;
  const std::size_t end =
      candidate_end < routed_rows ? candidate_end : routed_rows;
  for (std::size_t index = begin; index < end; ++index) {
    atomicAdd(&counts[topk_ids[index]], 1);
  }
  __syncthreads();

  if (thread == 0) {
    block_offsets[0] = 0;
    for (std::size_t expert = 0; expert < kExperts; ++expert) {
      block_offsets[expert + 1] =
          block_offsets[expert] +
          (counts[expert] + static_cast<int>(kMoeBlock) - 1) /
              static_cast<int>(kMoeBlock);
      write_positions[expert] =
          block_offsets[expert] * static_cast<int>(kMoeBlock);
    }
    *num_tokens_post_padded =
        block_offsets[kExperts] * static_cast<int>(kMoeBlock);
  }
  __syncthreads();

  for (int block = block_offsets[thread];
       block < block_offsets[thread + 1]; ++block) {
    expert_ids[block] = static_cast<std::int32_t>(thread);
  }
  __syncthreads();

  for (std::size_t index = begin; index < end; ++index) {
    const int expert = topk_ids[index];
    const int destination = atomicAdd(&write_positions[expert], 1);
    sorted_token_ids[destination] = static_cast<std::int32_t>(index);
  }
}

__global__ void expert_silu_multiply_kernel(
    const __hip_bfloat16* gate_up, __hip_bfloat16* activated,
    std::size_t elements) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const std::size_t row = index / kSharedIntermediate;
  const std::size_t column = index - row * kSharedIntermediate;
  const std::size_t base = row * kExpertGateUp;
  const float gate = __bfloat162float(gate_up[base + column]);
  const float up =
      __bfloat162float(gate_up[base + kSharedIntermediate + column]);
  const float silu = gate / (1.0f + expf(-gate));
  // vLLM's BF16 silu_and_mul rounds the SiLU result to BF16 before the
  // multiply.  Preserve that boundary; the shared-expert path intentionally
  // keeps its different FP32-intermediate semantics above.
  const __hip_bfloat16 silu_bf16 = __float2bfloat16(silu);
  activated[index] =
      __float2bfloat16(__bfloat162float(silu_bf16) * up);
}

__global__ void moe_sum8_kernel(const __hip_bfloat16* input,
                                __hip_bfloat16* output) {
  const std::size_t token = blockIdx.x;
  for (std::size_t hidden = threadIdx.x; hidden < kHidden;
       hidden += blockDim.x) {
    const std::size_t base = token * kTopK * kHidden + hidden;
    const float sum04 =
        __bfloat162float(input[base]) +
        __bfloat162float(input[base + 4 * kHidden]);
    const float sum15 =
        __bfloat162float(input[base + kHidden]) +
        __bfloat162float(input[base + 5 * kHidden]);
    const float sum26 =
        __bfloat162float(input[base + 2 * kHidden]) +
        __bfloat162float(input[base + 6 * kHidden]);
    const float sum37 =
        __bfloat162float(input[base + 3 * kHidden]) +
        __bfloat162float(input[base + 7 * kHidden]);
    // Match ATen's vectorized middle-axis reduction tree for
    // [tokens, top_k, hidden]: (((r0+r4)+(r1+r5))+(r2+r6))+(r3+r7).
    // A rank-ordered FP32 fold differs only at rare BF16 midpoints, but that
    // single bit can select a different expert in a later layer.
    float sum = sum04 + sum15;
    sum += sum26;
    sum += sum37;
    output[token * kHidden + hidden] = __float2bfloat16(sum);
  }
}

void launch_shared_activation(const void* gate, const void* up, void* output,
                              std::size_t tokens) {
  const std::size_t elements = tokens * kSharedIntermediate;
  hipLaunchKernelGGL(
      shared_silu_multiply_batched_kernel,
      dim3(static_cast<unsigned>((elements + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, nullptr,
      static_cast<const __hip_bfloat16*>(gate),
      static_cast<const __hip_bfloat16*>(up),
      static_cast<__hip_bfloat16*>(output), elements);
  check_hip(hipGetLastError(), "shared_silu_multiply_batched_kernel");
}

void launch_shared_gate(const void* gate, const void* down, void* output,
                        std::size_t tokens) {
  const std::size_t elements = tokens * kHidden;
  hipLaunchKernelGGL(
      shared_sigmoid_scale_batched_kernel,
      dim3(static_cast<unsigned>((elements + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, nullptr,
      static_cast<const __hip_bfloat16*>(gate),
      static_cast<const __hip_bfloat16*>(down),
      static_cast<__hip_bfloat16*>(output), elements);
  check_hip(hipGetLastError(), "shared_sigmoid_scale_batched_kernel");
}

void launch_router(const void* logits, void* scores, void* indices_i64,
                   void* indices_i32, void* weights, std::size_t tokens) {
  hipLaunchKernelGGL(
      router_topk8_softmax_256_kernel, dim3(tokens), dim3(64), 0, nullptr,
      static_cast<const __hip_bfloat16*>(logits), static_cast<float*>(scores),
      static_cast<std::int64_t*>(indices_i64),
      static_cast<std::int32_t*>(indices_i32),
      static_cast<__hip_bfloat16*>(weights));
  check_hip(hipGetLastError(), "router_topk8_softmax_256_kernel");
}

void launch_dispatch(const void* indices_i32, void* sorted_token_ids,
                     void* expert_ids, void* num_tokens_post_padded,
                     std::size_t routed_rows, std::size_t sorted_capacity,
                     std::size_t expert_block_capacity) {
  hipLaunchKernelGGL(
      moe_align_block32_256_kernel, dim3(1), dim3(kExperts), 0, nullptr,
      static_cast<const std::int32_t*>(indices_i32),
      static_cast<std::int32_t*>(sorted_token_ids),
      static_cast<std::int32_t*>(expert_ids),
      static_cast<std::int32_t*>(num_tokens_post_padded), routed_rows,
      sorted_capacity, expert_block_capacity);
  check_hip(hipGetLastError(), "moe_align_block32_256_kernel");
}

void launch_expert_activation(const void* gate_up, void* activated,
                              std::size_t routed_rows) {
  const std::size_t elements = routed_rows * kSharedIntermediate;
  hipLaunchKernelGGL(
      expert_silu_multiply_kernel,
      dim3(static_cast<unsigned>((elements + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, nullptr,
      static_cast<const __hip_bfloat16*>(gate_up),
      static_cast<__hip_bfloat16*>(activated), elements);
  check_hip(hipGetLastError(), "expert_silu_multiply_kernel");
}

void launch_moe_sum(const void* input, void* output, std::size_t tokens) {
  hipLaunchKernelGGL(
      moe_sum8_kernel, dim3(tokens), dim3(1024), 0, nullptr,
      static_cast<const __hip_bfloat16*>(input),
      static_cast<__hip_bfloat16*>(output));
  check_hip(hipGetLastError(), "moe_sum8_kernel");
}

}  // namespace

void* native_prefill_terminal_hidden_pointer(
    const NativePrefillWorkspace& workspace,
    const NativePrefillInvocations& invocations) {
  if (!workspace.built() || invocations.launches().empty()) {
    throw std::invalid_argument(
        "native prefill terminal hidden owner is incomplete");
  }
  return layer_output_pointer(
      invocations.launches(), workspace, invocations, 39,
      workspace.context_tokens());
}

void* native_prefill_layer_input_pointer(
    const NativePrefillWorkspace& workspace,
    const NativePrefillInvocations& invocations,
    std::size_t layer_index) {
  if (!workspace.built() || invocations.launches().empty()) {
    throw std::invalid_argument(
        "native prefill layer input owner is incomplete");
  }
  const PrefillLayerSchedule layer = find_prefill_layer_schedule(
      invocations.launches(), layer_index);
  return invocations.tensor_pointer(layer.base, "x");
}

void* native_prefill_layer_output_pointer(
    const NativePrefillWorkspace& workspace,
    const NativePrefillInvocations& invocations,
    std::size_t layer_index) {
  if (!workspace.built() || invocations.launches().empty()) {
    throw std::invalid_argument(
        "native prefill layer output owner is incomplete");
  }
  return layer_output_pointer(invocations.launches(), workspace, invocations,
                              layer_index, workspace.context_tokens());
}

NativeMoePrefillOracleResult probe_native_q8192_moe_prefill_layer0_oracle(
    const std::filesystem::path& oracle_dir,
    const NativeWeightStore& weights,
    const NativePrefillWorkspace& workspace,
    NativePrefillInvocations& invocations,
    NativeDecodeExecutor& executor,
    const NativeMoePrefillOracleOptions& options) {
  if (!weights.loaded() || !workspace.built() || !executor.loaded() ||
      (invocations.launches().size() != 431 &&
       invocations.launches().size() != 401)) {
    throw std::invalid_argument(
        "native MoE prefill oracle requires complete resident owners");
  }
  if (!options.collect_oracle_comparisons &&
      (options.run_routing_diagnostic || options.seed_post_attention ||
       !options.boundary_oracle_dir.empty())) {
    throw std::invalid_argument(
        "native MoE prefill execution cannot run oracle diagnostics");
  }
  const std::size_t tokens = workspace.context_tokens();
  if (tokens == 0 || tokens > 262144 ||
      (tokens != 8192 && options.collect_oracle_comparisons)) {
    throw std::invalid_argument(
        "native MoE prefill context or oracle mode is unsupported");
  }
  const std::size_t routed_rows = tokens * kTopK;
  const auto& launches = invocations.launches();
  const PrefillLayerSchedule layer_schedule =
      find_prefill_layer_schedule(launches, options.layer_index);
  const std::size_t moe_first =
      layer_schedule.base + layer_schedule.moe_offset;
  const std::size_t fused_add_sequence = moe_first - 1;
  require_symbol(launches, moe_first, "fused_moe_kernel");
  require_symbol(launches, moe_first + 1, "fused_moe_kernel");
  const std::size_t sorted_capacity =
      invocations.tensor_storage_bytes(
          moe_first, "sorted_token_ids_ptr") / sizeof(std::int32_t);
  const std::size_t expert_block_capacity =
      invocations.tensor_storage_bytes(
          moe_first, "expert_ids_ptr") / sizeof(std::int32_t);
  if (sorted_capacity < routed_rows || expert_block_capacity < kExperts) {
    throw std::runtime_error(
        "native MoE captured dispatch capacity is insufficient");
  }

  const std::string prefix = "model.language_model.layers." +
                             std::to_string(options.layer_index) + ".mlp.";
  const auto& shared_gate_weight = require_weight(
      weights, prefix + "shared_expert_gate.weight", 4096ULL);
  const auto& shared_gate_proj_weight = require_weight(
      weights, prefix + "shared_expert.gate_proj.weight", 2097152ULL);
  const auto& shared_up_proj_weight = require_weight(
      weights, prefix + "shared_expert.up_proj.weight", 2097152ULL);
  const auto& shared_down_proj_weight = require_weight(
      weights, prefix + "shared_expert.down_proj.weight", 2097152ULL);
  const auto& router_weight = require_weight(
      weights, prefix + "gate.weight", 1048576ULL);

  void* h2 = invocations.tensor_pointer(moe_first, "a_ptr");
  void* topk_weights =
      invocations.tensor_pointer(moe_first, "topk_weights_ptr");
  void* sorted_token_ids =
      invocations.tensor_pointer(moe_first, "sorted_token_ids_ptr");
  void* expert_ids =
      invocations.tensor_pointer(moe_first, "expert_ids_ptr");
  void* num_tokens_post_padded =
      invocations.tensor_pointer(moe_first, "num_tokens_post_padded_ptr");
  void* expert_gate_up = invocations.tensor_pointer(moe_first, "c_ptr");
  void* expert_activated =
      invocations.tensor_pointer(moe_first + 1, "a_ptr");
  void* expert_down =
      invocations.tensor_pointer(moe_first + 1, "c_ptr");

  void* after_attention =
      invocations.tensor_pointer(fused_add_sequence, "residual_out");
  const bool q8192 = tokens == 8192;
  void* shared_gate = nullptr;
  void* shared_projected_gate = nullptr;
  void* shared_projected_up = nullptr;
  void* shared_activated = nullptr;
  void* shared_down = nullptr;
  void* router_logits = nullptr;
  void* router_scores = nullptr;
  void* router_indices_i32 = nullptr;
  void* router_indices_i64 = nullptr;
  void* routed_moe = nullptr;
  void* combined_moe = nullptr;
  if (q8192) {
    shared_gate = require_workspace(
        workspace, "transient.7",
        tokens * sizeof(std::uint16_t)).device_pointer;
    shared_projected_gate = require_workspace(
        workspace, "transient.9",
        tokens * kSharedIntermediate * sizeof(std::uint16_t)).device_pointer;
    shared_projected_up = require_workspace(
        workspace, "transient.10",
        tokens * kSharedIntermediate * sizeof(std::uint16_t)).device_pointer;
    shared_activated = require_workspace(
        workspace, "transient.16",
        tokens * kSharedIntermediate * sizeof(std::uint16_t)).device_pointer;
    shared_down = require_workspace(
        workspace, "transient.15",
        tokens * kHidden * sizeof(std::uint16_t)).device_pointer;
    router_logits = require_workspace(
        workspace, "transient.33",
        tokens * kExperts * sizeof(std::uint16_t)).device_pointer;
    router_scores = require_workspace(
        workspace, "transient.24",
        tokens * kTopK * sizeof(float)).device_pointer;
    router_indices_i32 = require_workspace(
        workspace, "transient.30",
        tokens * kTopK * sizeof(std::int32_t)).device_pointer;
    router_indices_i64 = require_workspace(
        workspace, "transient.6",
        tokens * kTopK * sizeof(std::int64_t)).device_pointer;
    routed_moe = require_workspace(
        workspace, "transient.34",
        tokens * kHidden * sizeof(std::uint16_t)).device_pointer;
    combined_moe = require_workspace(
        workspace, "transient.35",
        tokens * kHidden * sizeof(std::uint16_t)).device_pointer;
  } else {
    const PrefillLayerSchedule scratch =
        find_prefill_layer_schedule(launches, 0);
    shared_gate = require_invocation_tensor(
        invocations, scratch.base + 2, "g_ptr",
        tokens * sizeof(std::uint16_t));
    shared_projected_gate = require_invocation_tensor(
        invocations, scratch.base + 2, "q_ptr",
        tokens * kSharedIntermediate * sizeof(std::uint16_t));
    shared_projected_up = require_invocation_tensor(
        invocations, scratch.base + 2, "k_ptr",
        tokens * kSharedIntermediate * sizeof(std::uint16_t));
    shared_activated = require_invocation_tensor(
        invocations, scratch.base + 2, "v_ptr",
        tokens * kSharedIntermediate * sizeof(std::uint16_t));
    shared_down = require_invocation_tensor(
        invocations, scratch.base + 6, "w",
        tokens * kHidden * sizeof(std::uint16_t));
    router_logits = require_invocation_tensor(
        invocations, scratch.base + 4, "A",
        tokens * kExperts * sizeof(std::uint16_t));
    router_scores = require_invocation_tensor(
        invocations, scratch.base + 3, "o",
        tokens * kTopK * sizeof(float));
    router_indices_i32 = require_invocation_tensor(
        invocations, scratch.base + 2, "beta_ptr",
        tokens * kTopK * sizeof(std::int32_t));
    router_indices_i64 = require_invocation_tensor(
        invocations, scratch.base + 2, "a_ptr",
        tokens * kTopK * sizeof(std::int64_t));
    routed_moe = require_invocation_tensor(
        invocations, scratch.base + 6, "u",
        tokens * kHidden * sizeof(std::uint16_t));
    combined_moe = require_invocation_tensor(
        invocations, scratch.base + 7, "v_new",
        tokens * kHidden * sizeof(std::uint16_t));
  }
  void* layer_output =
      layer_output_pointer(launches, workspace, invocations,
                           options.layer_index, tokens);
  // Once the attention residual and normalized MoE input have been produced,
  // the current layer input is dead.  Reuse that exact per-layer buffer for
  // the scaled shared-expert output.  A fixed captured transient is unsafe:
  // transient.31 is also the layer-1 output / layer-2 input, so writing the
  // shared branch there corrupts the next-layer state.
  void* shared_scaled =
      invocations.tensor_pointer(layer_schedule.base, "x");
  if (shared_scaled == layer_output || shared_scaled == after_attention ||
      shared_scaled == h2) {
    throw std::runtime_error(
        "native MoE prefill scratch aliases a live layer boundary");
  }

  const std::filesystem::path fixture =
      oracle_dir.empty() ? std::filesystem::path{}
                         : std::filesystem::absolute(oracle_dir);
  const auto oracle_file = [&fixture, &options](const std::string& label) {
    return find_native_oracle_tensor_file(
        fixture, options.oracle_label_prefix + label);
  };
  const std::filesystem::path boundary_fixture =
      options.boundary_oracle_dir.empty()
      ? std::filesystem::path{}
      : std::filesystem::absolute(options.boundary_oracle_dir);
  if (!boundary_fixture.empty() && !options.run_routing_diagnostic) {
    throw std::invalid_argument(
        "native MoE expert boundary checks require the routing diagnostic");
  }
  const auto boundary_file = [&boundary_fixture, &options](
                                 const std::string& label) {
    return find_native_oracle_tensor_file(
        boundary_fixture, options.boundary_oracle_label_prefix + label);
  };

  NativeMoePrefillOracleResult result;
  result.layer.layer_index = options.layer_index;
  result.layer.tokens = tokens;
  const auto diagnostic_stage = [&options](const char* stage) {
    if (!options.synchronize_substages) return;
    std::fprintf(stderr,
                 "{\"event\":\"native_moe_prefill_stage\","
                 "\"layer_index\":%zu,\"stage\":\"%s\"}\n",
                 options.layer_index, stage);
    std::fflush(stderr);
  };
  result.post_attention_seeded = options.seed_post_attention;
  if (options.seed_post_attention) {
    result.seed_bytes += seed_native_oracle_tensor(
        oracle_file("return-layer_body-h2"), h2,
        tokens * kHidden * sizeof(std::uint16_t));
    ++result.seed_tensors;
    result.seed_bytes += seed_native_oracle_tensor(
        oracle_file("return-layer_body-after_attn"),
        after_attention,
        tokens * kHidden * sizeof(std::uint16_t));
    ++result.seed_tensors;
  }

  // Match PyTorch's qualified hipBLASLt N=1 preference exactly.  Besides the
  // direct N=1 layout in Bf16GemmPlan, its ROCm allocator exposes 76 MiB here.
  std::unique_ptr<NativeQ8192PrefillGemmPlans> local_gemm_plans;
  NativeQ8192PrefillGemmPlans* gemm_plans = options.gemm_plans;
  if (gemm_plans == nullptr) {
    local_gemm_plans =
        std::make_unique<NativeQ8192PrefillGemmPlans>(tokens);
    gemm_plans = local_gemm_plans.get();
  }
  if (gemm_plans->token_count() != tokens) {
    throw std::invalid_argument("native MoE prefill GEMM context mismatch");
  }
  diagnostic_stage("before_shared_gate_plan");
  Bf16GemmPlan& shared_gate_plan = gemm_plans->moe_shared_gate();
  diagnostic_stage("after_shared_gate_plan");
  Bf16GemmPlan& shared_projection_plan =
      gemm_plans->moe_shared_projection();
  diagnostic_stage("after_shared_projection_plan");
  Bf16GemmPlan& shared_down_plan = gemm_plans->moe_shared_down();
  diagnostic_stage("after_shared_down_plan");
  Bf16GemmPlan& router_plan = gemm_plans->moe_router();
  diagnostic_stage("after_router_plan");
  result.layer.gemm_workspace_bytes =
      shared_gate_plan.workspace_bytes() +
      2 * shared_projection_plan.workspace_bytes() +
      shared_down_plan.workspace_bytes() + router_plan.workspace_bytes();

  const auto started = std::chrono::steady_clock::now();
  shared_gate_plan.launch(h2, shared_gate_weight.device_pointer,
                          shared_gate);
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native MoE shared gate");
  }
  diagnostic_stage("after_shared_gate");
  shared_projection_plan.launch(h2, shared_gate_proj_weight.device_pointer,
                                shared_projected_gate);
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native MoE shared gate projection");
  }
  diagnostic_stage("after_shared_gate_projection");
  shared_projection_plan.launch(h2, shared_up_proj_weight.device_pointer,
                                shared_projected_up);
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native MoE shared up projection");
  }
  diagnostic_stage("after_shared_up_projection");
  result.layer.dense_gemm_launches += 3;
  launch_shared_activation(shared_projected_gate,
                           shared_projected_up,
                           shared_activated, tokens);
  ++result.layer.native_pointwise_launches;
  shared_down_plan.launch(shared_activated,
                          shared_down_proj_weight.device_pointer,
                          shared_down);
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native MoE shared down projection");
  }
  diagnostic_stage("after_shared_down_projection");
  ++result.layer.dense_gemm_launches;
  launch_shared_gate(shared_gate, shared_down,
                     shared_scaled, tokens);
  ++result.layer.native_pointwise_launches;

  router_plan.launch(h2, router_weight.device_pointer,
                     router_logits);
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native MoE router projection");
  }
  diagnostic_stage("after_router_projection");
  ++result.layer.dense_gemm_launches;
  launch_router(router_logits, router_scores,
                router_indices_i64,
                router_indices_i32, topk_weights, tokens);
  ++result.layer.native_router_launches;
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native MoE router topk");
  }
  diagnostic_stage("after_router_topk");
  launch_dispatch(router_indices_i32, sorted_token_ids,
                  expert_ids, num_tokens_post_padded, routed_rows,
                  sorted_capacity, expert_block_capacity);
  ++result.layer.native_dispatch_launches;
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native MoE dispatch");
  }
  diagnostic_stage("after_dispatch");
  if (options.collect_oracle_comparisons) {
    check_hip(hipMemcpy(&result.layer.padded_routed_rows,
                        num_tokens_post_padded, sizeof(std::int32_t),
                        hipMemcpyDeviceToHost),
              "hipMemcpy native MoE padded row count");
  }

  executor.launch(launches[moe_first]);
  ++result.layer.aot_launches;
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native MoE expert gate-up");
  }
  diagnostic_stage("after_expert_gate_up");
  launch_expert_activation(expert_gate_up, expert_activated, routed_rows);
  ++result.layer.native_pointwise_launches;
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native MoE expert activation");
  }
  diagnostic_stage("after_expert_activation");
  executor.launch(launches[moe_first + 1]);
  ++result.layer.aot_launches;
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native MoE expert down");
  }
  diagnostic_stage("after_expert_down");
  launch_moe_sum(expert_down, routed_moe, tokens);
  ++result.layer.native_pointwise_launches;
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native MoE expert sum");
  }
  diagnostic_stage("after_expert_sum");
  launch_bf16_add_pair(
      routed_moe, shared_scaled,
      after_attention, combined_moe,
      layer_output, tokens * kHidden);
  ++result.layer.native_pointwise_launches;
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native MoE output add");
  }
  diagnostic_stage("after_output_add");
  if (options.collect_oracle_comparisons ||
      !options.chain_output_oracle_dir.empty()) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native MoE prefill oracle");
  }
  result.layer.wall_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();

  const std::size_t shared_bytes =
      tokens * kSharedIntermediate * sizeof(std::uint16_t);
  const std::size_t hidden_bytes =
      tokens * kHidden * sizeof(std::uint16_t);
  if (options.collect_oracle_comparisons) {
    result.comparisons = {
      compare_native_oracle_tensor(
          "shared_gate", "bfloat16", shared_gate,
          tokens * sizeof(std::uint16_t),
          oracle_file("return-shared_expert-shared_gate")),
      compare_native_oracle_tensor(
          "shared_gate_projection", "bfloat16",
          shared_projected_gate, shared_bytes,
          oracle_file("return-shared_expert-gate")),
      compare_native_oracle_tensor(
          "shared_up_projection", "bfloat16",
          shared_projected_up, shared_bytes,
          oracle_file("return-shared_expert-up")),
      compare_native_oracle_tensor(
          "shared_activation", "bfloat16",
          shared_activated, shared_bytes,
          oracle_file("return-shared_expert-activated")),
      compare_native_oracle_tensor(
          "shared_down", "bfloat16", shared_down,
          hidden_bytes, oracle_file("return-shared_expert-shared_out")),
      compare_native_oracle_tensor(
          "shared_output", "bfloat16", shared_scaled,
          hidden_bytes, oracle_file("return-shared_expert-output")),
      compare_native_oracle_tensor(
          "router_scores", "float32", router_scores,
          tokens * kTopK * sizeof(float),
          oracle_file("return-layer_body-scores")),
      compare_native_oracle_tensor(
          "router_indices", "int64", router_indices_i64,
          tokens * kTopK * sizeof(std::int64_t),
          oracle_file("return-layer_body-indices")),
      compare_native_oracle_tensor(
          "router_weights", "bfloat16", topk_weights,
          tokens * kTopK * sizeof(std::uint16_t),
          oracle_file(launch_label(layer_schedule.moe_offset,
                                   "topk_weights_ptr"))),
      compare_native_oracle_tensor(
          "dispatch_sorted_tokens", "int32", sorted_token_ids,
          sorted_capacity * sizeof(std::int32_t),
          oracle_file(launch_label(layer_schedule.moe_offset,
                                   "sorted_token_ids_ptr"))),
      compare_native_oracle_tensor(
          "dispatch_expert_ids", "int32", expert_ids,
          expert_block_capacity * sizeof(std::int32_t),
          oracle_file(launch_label(layer_schedule.moe_offset,
                                   "expert_ids_ptr"))),
      compare_native_oracle_tensor(
          "dispatch_padded_rows", "int32", num_tokens_post_padded,
          sizeof(std::int32_t),
          oracle_file(launch_label(layer_schedule.moe_offset,
                                   "num_tokens_post_padded_ptr"))),
      compare_native_oracle_tensor(
          "combined_moe_output", "bfloat16", combined_moe,
          hidden_bytes, oracle_file("return-layer_body-moe_out")),
      compare_native_oracle_tensor(
          "layer_output", "bfloat16", layer_output,
          hidden_bytes, oracle_file("return-layer_body-output")),
    };
    result.router_expert_set_rows = tokens;
    result.router_expert_set_rows_exact = count_exact_int64_row_sets(
        router_indices_i64,
        oracle_file("return-layer_body-indices"), tokens, kTopK);
    result.router_expert_sets_exact =
        result.router_expert_set_rows_exact == result.router_expert_set_rows;
  }
  if (!options.chain_output_oracle_dir.empty()) {
    if (options.chain_output_oracle_label.empty()) {
      throw std::invalid_argument(
          "native MoE chain-output oracle label is required");
    }
    const std::size_t comparison_bytes =
        options.chain_output_last_token_only
            ? kHidden * sizeof(std::uint16_t)
            : hidden_bytes;
    const auto* comparison_pointer =
        static_cast<const unsigned char*>(layer_output) +
        (options.chain_output_last_token_only
             ? (tokens - 1) * kHidden * sizeof(std::uint16_t)
             : 0);
    const std::filesystem::path expected =
        find_native_oracle_tensor_file_if_present(
            options.chain_output_oracle_dir,
            options.chain_output_oracle_label);
    if (!expected.empty()) {
      result.chain_output_comparison = compare_native_oracle_tensor(
          options.chain_output_last_token_only
              ? options.chain_output_oracle_label + ":last_token"
              : "same_request_layer_output",
          "bfloat16", comparison_pointer, comparison_bytes, expected);
      result.chain_output_comparison_provided = true;
    }

    const std::string output_suffix = "output";
    if (options.chain_output_oracle_label.size() >= output_suffix.size() &&
        options.chain_output_oracle_label.compare(
            options.chain_output_oracle_label.size() - output_suffix.size(),
            output_suffix.size(), output_suffix) == 0) {
      const std::string label_prefix =
          options.chain_output_oracle_label.substr(
              0, options.chain_output_oracle_label.size() -
                     output_suffix.size());
      const auto compare_stage_if_present =
          [&](const std::string& suffix, const char* dtype,
              const void* pointer, std::size_t bytes) {
            const std::string label = label_prefix + suffix;
            const std::filesystem::path stage_expected =
                find_native_oracle_tensor_file_if_present(
                    options.chain_output_oracle_dir, label);
            if (!stage_expected.empty()) {
              result.comparisons.push_back(compare_native_oracle_tensor(
                  label, dtype, pointer, bytes, stage_expected));
            }
          };
      compare_stage_if_present("h2", "bfloat16", h2, hidden_bytes);
      compare_stage_if_present("scores", "float32", router_scores,
                               tokens * kTopK * sizeof(float));
      compare_stage_if_present("indices", "int64", router_indices_i64,
                               tokens * kTopK * sizeof(std::int64_t));
      compare_stage_if_present("shared_out", "bfloat16", shared_scaled,
                               hidden_bytes);
      compare_stage_if_present("routed_moe", "bfloat16", routed_moe,
                               hidden_bytes);
      compare_stage_if_present("moe_out", "bfloat16", combined_moe,
                               hidden_bytes);
    }
  }

  if (!options.collect_oracle_comparisons) {
    result.all_finite =
        std::all_of(
            result.comparisons.begin(), result.comparisons.end(),
            [](const NativeOracleComparison& comparison) {
              return comparison.finite_elements == comparison.elements;
            }) &&
        (!result.chain_output_comparison_provided ||
         result.chain_output_comparison.finite_elements ==
             result.chain_output_comparison.elements);
    result.final_hidden_gate_passed =
        result.chain_output_comparison_provided &&
        gate(result.chain_output_comparison);
    return result;
  }

  std::size_t oracle_seeded_combined_moe_index = 0;
  if (options.run_routing_diagnostic) {
    // Isolate the embedded expert GEMMs, native expert activation, and top-8
    // reduction from router tie-order behavior.  These qualification-only
    // reruns seed the exact reference routing tensors and are excluded from
    // production counts.  They mutate layer_output and therefore cannot run
    // inside a multi-layer production chain.
    result.seed_bytes += seed_native_oracle_tensor(
        oracle_file(launch_label(layer_schedule.moe_offset,
                                 "topk_weights_ptr")),
        topk_weights,
        tokens * kTopK * sizeof(std::uint16_t));
    result.seed_bytes += seed_native_oracle_tensor(
        oracle_file(launch_label(layer_schedule.moe_offset,
                                 "sorted_token_ids_ptr")),
        sorted_token_ids,
        sorted_capacity * sizeof(std::int32_t));
    result.seed_bytes += seed_native_oracle_tensor(
        oracle_file(launch_label(layer_schedule.moe_offset,
                                 "expert_ids_ptr")),
        expert_ids,
        expert_block_capacity * sizeof(std::int32_t));
    result.seed_bytes += seed_native_oracle_tensor(
        oracle_file(launch_label(layer_schedule.moe_offset,
                                 "num_tokens_post_padded_ptr")),
        num_tokens_post_padded, sizeof(std::int32_t));
    result.seed_tensors += 4;
    executor.launch(launches[moe_first]);
    if (!boundary_fixture.empty()) {
      result.comparisons.push_back(compare_native_oracle_tensor(
          "oracle_seeded_expert_gate_up", "bfloat16", expert_gate_up,
          routed_rows * kExpertGateUp * sizeof(std::uint16_t),
          boundary_file(launch_label(layer_schedule.moe_offset, "c_ptr"))));
    }
    launch_expert_activation(expert_gate_up, expert_activated, routed_rows);
    if (!boundary_fixture.empty()) {
      result.comparisons.push_back(compare_native_oracle_tensor(
          "oracle_seeded_expert_activation", "bfloat16", expert_activated,
          routed_rows * kSharedIntermediate * sizeof(std::uint16_t),
          boundary_file(launch_label(layer_schedule.moe_offset + 1,
                                     "a_ptr"))));
    }
    executor.launch(launches[moe_first + 1]);
    if (!boundary_fixture.empty()) {
      result.comparisons.push_back(compare_native_oracle_tensor(
          "oracle_seeded_expert_down", "bfloat16", expert_down,
          routed_rows * kHidden * sizeof(std::uint16_t),
          boundary_file(launch_label(layer_schedule.moe_offset + 1,
                                     "c_ptr"))));
    }
    launch_moe_sum(expert_down, routed_moe, tokens);
    launch_bf16_add_pair(
        routed_moe, shared_scaled,
        after_attention, combined_moe,
        layer_output, tokens * kHidden);
    result.layer.diagnostic_aot_launches = 2;
    result.layer.diagnostic_pointwise_launches = 3;
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize oracle-seeded native MoE diagnostic");
    oracle_seeded_combined_moe_index = result.comparisons.size();
    result.comparisons.push_back(compare_native_oracle_tensor(
        "oracle_seeded_combined_moe_output", "bfloat16",
        combined_moe, hidden_bytes,
        oracle_file("return-layer_body-moe_out")));
    result.routing_diagnostic_ran = true;
  }
  result.all_finite = true;
  for (const auto& comparison : result.comparisons) {
    result.all_finite = result.all_finite &&
                        comparison.finite_elements == comparison.elements;
  }
  result.router_ids_exact =
      result.comparisons[7].exact_elements == result.comparisons[7].elements;
  result.router_weights_gate_passed = gate(result.comparisons[8]);
  result.dispatch_count_exact =
      result.comparisons[11].exact_elements == result.comparisons[11].elements;
  result.shared_expert_gate_passed = gate(result.comparisons[5]);
  result.combined_moe_gate_passed = gate(result.comparisons[12]);
  result.final_hidden_gate_passed = gate(result.comparisons[13]);
  result.expert_boundaries_provided = !boundary_fixture.empty();
  if (result.expert_boundaries_provided) {
    result.expert_boundaries_gate_passed =
        gate(result.comparisons[14]) && gate(result.comparisons[15]) &&
        gate(result.comparisons[16]);
  }
  result.oracle_seeded_combined_moe_gate_passed =
      !result.routing_diagnostic_ran ||
      gate(result.comparisons[oracle_seeded_combined_moe_index]);
  return result;
}

}  // namespace aima
