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

__global__ void shared_silu_multiply_batched_v151_kernel(
    const __hip_bfloat16* gate, const __hip_bfloat16* up,
    __hip_bfloat16* output, std::size_t elements) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float gate_value = __bfloat162float(gate[index]);
  const float up_value = __bfloat162float(up[index]);
  const float silu = gate_value / (1.0f + expf(-gate_value));
  output[index] = __float2bfloat16(silu * up_value);
}

__global__ void shared_silu_multiply_batched_kernel(
    const __hip_bfloat16* gate, const __hip_bfloat16* up,
    __hip_bfloat16* output, std::size_t elements) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float gate_value = __bfloat162float(gate[index]);
  const float up_value = __bfloat162float(up[index]);
  const float silu = gate_value / (1.0f + expf(-gate_value));
  // vLLM's CUDA-like SiluAndMul materializes the SiLU branch in BF16 before
  // multiplying it by the BF16 up branch.  This is also the routed-expert
  // activation boundary below; keeping FP32 SiLU here moves the seeded shared
  // expert outside the product acceptance threshold.
  const __hip_bfloat16 silu_bf16 = __float2bfloat16(silu);
  output[index] =
      __float2bfloat16(__bfloat162float(silu_bf16) * up_value);
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

template <bool kWeightsAreBfloat16>
__global__ void router_topk8_softmax_256_text_kernel(
    const __hip_bfloat16* logits, float* scores, std::int64_t* indices_i64,
    std::int32_t* indices_i32, void* weights) {
  constexpr int kRouterThreads = 64;
  constexpr int kRouterWave = 32;
  constexpr int kRouterWaves = kRouterThreads / kRouterWave;
  constexpr int kValuesPerThread = kExperts / kRouterThreads;
  static_assert(kRouterWaves == 2 && kValuesPerThread == 4);
  const std::size_t token = blockIdx.x;
  const __hip_bfloat16* row = logits + token * kExperts;
  const int thread = threadIdx.x;
  const int wave_lane = thread % kRouterWave;
  const int wave = thread / kRouterWave;
  __shared__ float row_values[kExperts];
  __shared__ float wave_scores[kRouterWaves];
  __shared__ int wave_indices[kRouterWaves];
  __shared__ float ranked_scores[kTopK];
  __shared__ int ranked_indices[kTopK];

#pragma unroll
  for (int value = 0; value < kValuesPerThread; ++value) {
    const int expert = thread + value * kRouterThreads;
    row_values[expert] = __bfloat162float(row[expert]);
  }
  __syncthreads();

  // Preserve the serial v1.5.1 selection rule while distributing each of its
  // eight maximum searches across two wave32s. Equal values explicitly pick
  // the lower expert index, matching the source-order scan below.
  for (std::size_t rank = 0; rank < kTopK; ++rank) {
    float best = -std::numeric_limits<float>::infinity();
    int best_index = -1;
#pragma unroll
    for (int value = 0; value < kValuesPerThread; ++value) {
      const int expert = thread + value * kRouterThreads;
      bool used = false;
#pragma unroll
      for (std::size_t prior = 0; prior < rank; ++prior) {
        used = used || ranked_indices[prior] == expert;
      }
      if (used) continue;
      const float candidate = row_values[expert];
      if (candidate > best ||
          (candidate == best && expert < best_index)) {
        best = candidate;
        best_index = expert;
      }
    }
#pragma unroll
    for (int offset = kRouterWave / 2; offset > 0; offset /= 2) {
      const float other_score = __shfl_down(best, offset, kRouterWave);
      const int other_index = __shfl_down(best_index, offset, kRouterWave);
      if (wave_lane + offset < kRouterWave &&
          (other_score > best ||
           (other_score == best && other_index < best_index))) {
        best = other_score;
        best_index = other_index;
      }
    }
    if (wave_lane == 0) {
      wave_scores[wave] = best;
      wave_indices[wave] = best_index;
    }
    __syncthreads();
    if (thread == 0) {
      const bool second_is_better =
          wave_scores[1] > wave_scores[0] ||
          (wave_scores[1] == wave_scores[0] &&
           wave_indices[1] < wave_indices[0]);
      ranked_scores[rank] =
          second_is_better ? wave_scores[1] : wave_scores[0];
      ranked_indices[rank] =
          second_is_better ? wave_indices[1] : wave_indices[0];
    }
    __syncthreads();
  }

  if (thread != 0) return;

  // Preserve the frozen ROCm top-k tie order: gather values above the kth
  // threshold, append threshold ties in source order, then apply the same
  // deterministic bitonic swaps used by the v1.5.1 product.
  const float threshold = ranked_scores[kTopK - 1];
  float selected_scores[kTopK];
  int selected_indices[kTopK];
  std::size_t selected = 0;
  for (int expert = 0; expert < static_cast<int>(kExperts); ++expert) {
    const float value = row_values[expert];
    if (value > threshold) {
      selected_scores[selected] = value;
      selected_indices[selected] = expert;
      ++selected;
    }
  }
  for (int expert = 0;
       expert < static_cast<int>(kExperts) && selected < kTopK; ++expert) {
    const float value = row_values[expert];
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
    // v1.5.1 materialized routed weights as BF16.  New q1024 captures expose
    // FP32 storage to the fused expert kernels, but that ABI expansion must
    // not silently change text arithmetic: round first, then widen when the
    // destination buffer is FP32.  VL routing retains its current-vLLM FP32
    // value in router_topk8_softmax_256_kernel below.
    const __hip_bfloat16 rounded_probability =
        __float2bfloat16(probability);
    if constexpr (kWeightsAreBfloat16) {
      static_cast<__hip_bfloat16*>(weights)[base + rank] =
          rounded_probability;
    } else {
      static_cast<float*>(weights)[base + rank] =
          __bfloat162float(rounded_probability);
    }
  }
}

__global__ void router_topk8_softmax_256_kernel(
    const __hip_bfloat16* logits, float* scores, std::int64_t* indices_i64,
    std::int32_t* indices_i32, void* weights,
    bool weights_are_bfloat16) {
  constexpr int kRouterWave = 32;
  constexpr int kValuesPerLane = kExperts / kRouterWave;
  const std::size_t token = blockIdx.x;
  const int lane = threadIdx.x;
  const __hip_bfloat16* row = logits + token * kExperts;
  float row_chunk[kValuesPerLane];
#pragma unroll
  for (int value = 0; value < kValuesPerLane; ++value) {
    row_chunk[value] =
        __bfloat162float(row[lane * kValuesPerLane + value]);
  }

  // Match vLLM's gfx1151 topkGating<8, 256, ..., BF16, Softmax>
  // arithmetic.  Each 32-lane wave owns one row and each lane consumes eight
  // contiguous experts before the XOR reductions.  Computing only the chosen
  // eight logits is mathematically equivalent, but does not preserve the two
  // visible FP32 rounding boundaries below.
  float thread_max = row_chunk[0];
#pragma unroll
  for (int value = 1; value < kValuesPerLane; ++value) {
    thread_max = fmaxf(thread_max, row_chunk[value]);
  }
#pragma unroll
  for (int mask = kRouterWave / 2; mask > 0; mask /= 2) {
    thread_max = fmaxf(
        thread_max, __shfl_xor(thread_max, mask, kRouterWave));
  }

  float row_sum = 0.0f;
#pragma unroll
  for (int value = 0; value < kValuesPerLane; ++value) {
    row_chunk[value] = expf(row_chunk[value] - thread_max);
    row_sum += row_chunk[value];
  }
#pragma unroll
  for (int mask = kRouterWave / 2; mask > 0; mask /= 2) {
    row_sum += __shfl_xor(row_sum, mask, kRouterWave);
  }
  const float reciprocal_row_sum = 1.0f / row_sum;
#pragma unroll
  for (int value = 0; value < kValuesPerLane; ++value) {
    row_chunk[value] *= reciprocal_row_sum;
  }

  float selected_probabilities[kTopK];
  int selected_indices[kTopK];
  float selected_sum = 0.0f;
  for (int rank = 0; rank < static_cast<int>(kTopK); ++rank) {
    float max_probability = row_chunk[0];
    int expert = lane * kValuesPerLane;
#pragma unroll
    for (int value = 0; value < kValuesPerLane; ++value) {
      const float candidate = row_chunk[value];
      if (candidate > max_probability) {
        max_probability = candidate;
        expert = lane * kValuesPerLane + value;
      }
    }
#pragma unroll
    for (int mask = kRouterWave / 2; mask > 0; mask /= 2) {
      const float other_probability =
          __shfl_xor(max_probability, mask, kRouterWave);
      const int other_expert = __shfl_xor(expert, mask, kRouterWave);
      if (other_probability > max_probability ||
          (other_probability == max_probability && other_expert < expert)) {
        max_probability = other_probability;
        expert = other_expert;
      }
    }
    if (lane == 0) {
      selected_probabilities[rank] = max_probability;
      selected_indices[rank] = expert;
      selected_sum += max_probability;
    }
    if (expert / kValuesPerLane == lane) {
      row_chunk[expert % kValuesPerLane] = -10000.0f;
    }
  }

  if (lane == 0) {
    const float denominator = selected_sum > 0.0f ? selected_sum : 1.0f;
    const std::size_t base = token * kTopK;
#pragma unroll
    for (std::size_t rank = 0; rank < kTopK; ++rank) {
      const int expert = selected_indices[rank];
      scores[base + rank] = __bfloat162float(row[expert]);
      indices_i64[base + rank] = expert;
      indices_i32[base + rank] = expert;
      // vLLM first rounds the full 256-way softmax probability, then divides
      // the chosen eight probabilities by their FP32 sum when renormalize is
      // enabled.  This second division is observable by the expert kernels.
      const float weight = selected_probabilities[rank] / denominator;
      if (weights_are_bfloat16) {
        static_cast<__hip_bfloat16*>(weights)[base + rank] =
            __float2bfloat16(weight);
      } else {
        static_cast<float*>(weights)[base + rank] = weight;
      }
    }
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
  // vLLM's BF16 SiluAndMul rounds the SiLU result to BF16 before the
  // multiply.  Preserve that boundary in both expert implementations.
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
                              bool use_vl_shared_expert_semantics,
                              std::size_t tokens) {
  const std::size_t elements = tokens * kSharedIntermediate;
  const dim3 grid(
      static_cast<unsigned>((elements + kThreads - 1) / kThreads));
  if (use_vl_shared_expert_semantics) {
    hipLaunchKernelGGL(
        shared_silu_multiply_batched_kernel, grid, dim3(kThreads), 0, nullptr,
        static_cast<const __hip_bfloat16*>(gate),
        static_cast<const __hip_bfloat16*>(up),
        static_cast<__hip_bfloat16*>(output), elements);
    check_hip(hipGetLastError(), "shared_silu_multiply_batched_kernel");
  } else {
    hipLaunchKernelGGL(
        shared_silu_multiply_batched_v151_kernel, grid, dim3(kThreads), 0,
        nullptr, static_cast<const __hip_bfloat16*>(gate),
        static_cast<const __hip_bfloat16*>(up),
        static_cast<__hip_bfloat16*>(output), elements);
    check_hip(
        hipGetLastError(), "shared_silu_multiply_batched_v151_kernel");
  }
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
                   void* indices_i32, void* weights,
                   bool weights_are_bfloat16,
                   bool use_vl_router_semantics, std::size_t tokens) {
  if (use_vl_router_semantics) {
    hipLaunchKernelGGL(
        router_topk8_softmax_256_kernel, dim3(tokens), dim3(32), 0, nullptr,
        static_cast<const __hip_bfloat16*>(logits),
        static_cast<float*>(scores),
        static_cast<std::int64_t*>(indices_i64),
        static_cast<std::int32_t*>(indices_i32), weights,
        weights_are_bfloat16);
    check_hip(hipGetLastError(), "router_topk8_softmax_256_kernel");
  } else {
    if (weights_are_bfloat16) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(router_topk8_softmax_256_text_kernel<true>),
          dim3(tokens), dim3(64), 0, nullptr,
          static_cast<const __hip_bfloat16*>(logits),
          static_cast<float*>(scores),
          static_cast<std::int64_t*>(indices_i64),
          static_cast<std::int32_t*>(indices_i32), weights);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(router_topk8_softmax_256_text_kernel<false>),
          dim3(tokens), dim3(64), 0, nullptr,
          static_cast<const __hip_bfloat16*>(logits),
          static_cast<float*>(scores),
          static_cast<std::int64_t*>(indices_i64),
          static_cast<std::int32_t*>(indices_i32), weights);
    }
    check_hip(hipGetLastError(),
              "router_topk8_softmax_256_text_kernel");
  }
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
      (options.run_routing_diagnostic ||
       !options.boundary_oracle_dir.empty())) {
    throw std::invalid_argument(
        "native MoE prefill execution cannot run oracle diagnostics");
  }
  const std::size_t bucket_tokens = workspace.context_tokens();
  const std::size_t tokens =
      options.active_tokens == 0 ? bucket_tokens : options.active_tokens;
  const std::size_t comparison_tokens =
      options.comparison_tokens == 0 ? tokens : options.comparison_tokens;
  if (bucket_tokens == 0 || bucket_tokens > 262144 || tokens == 0 ||
      tokens > bucket_tokens || comparison_tokens == 0 ||
      comparison_tokens > tokens ||
      (tokens != bucket_tokens && options.collect_oracle_comparisons) ||
      (bucket_tokens != 8192 && options.collect_oracle_comparisons)) {
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
  const std::uint64_t gate_up_router_weight_storage_bytes =
      invocations.tensor_storage_bytes(moe_first, "topk_weights_ptr");
  const std::uint64_t down_router_weight_storage_bytes =
      invocations.tensor_storage_bytes(moe_first + 1, "topk_weights_ptr");
  const std::uint64_t router_weight_elements = bucket_tokens * kTopK;
  // The recaptured q1024 closure consumes FP32 routed weights, while the
  // frozen larger closures consume BF16. The AOT tensor storage contract is
  // authoritative here: writing every bucket as FP32 both changes the load
  // bit pattern and overruns the larger closures' buffers.
  const bool router_weights_are_bfloat16 =
      down_router_weight_storage_bytes ==
      router_weight_elements * sizeof(std::uint16_t);
  const bool router_weights_are_float32 =
      down_router_weight_storage_bytes ==
      router_weight_elements * sizeof(float);
  if (sorted_capacity < routed_rows || expert_block_capacity < kExperts) {
    throw std::runtime_error(
        "native MoE captured dispatch capacity is insufficient");
  }
  if (gate_up_router_weight_storage_bytes !=
          down_router_weight_storage_bytes ||
      (!router_weights_are_bfloat16 && !router_weights_are_float32)) {
    throw std::runtime_error(
        "native MoE captured router-weight ABI is unsupported");
  }
  const char* router_weight_dtype =
      router_weights_are_bfloat16 ? "bfloat16" : "float32";
  const std::size_t router_weight_element_bytes =
      router_weights_are_bfloat16 ? sizeof(std::uint16_t) : sizeof(float);

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
  if (topk_weights !=
      invocations.tensor_pointer(moe_first + 1, "topk_weights_ptr")) {
    throw std::runtime_error(
        "native MoE captured router-weight bindings disagree");
  }
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
  const bool q8192 = bucket_tokens == 8192;
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
    if (options.post_attention_h2_oracle_label.empty() ||
        options.post_attention_residual_oracle_label.empty()) {
      throw std::invalid_argument(
          "native MoE post-attention seed labels are required");
    }
    if (comparison_tokens != tokens) {
      check_hip(hipMemset(h2, 0, tokens * kHidden * sizeof(std::uint16_t)),
                "hipMemset padded seeded MoE H2");
      check_hip(
          hipMemset(after_attention, 0,
                    tokens * kHidden * sizeof(std::uint16_t)),
          "hipMemset padded seeded MoE residual");
    }
    result.seed_bytes += seed_native_oracle_tensor(
        oracle_file(options.post_attention_h2_oracle_label), h2,
        comparison_tokens * kHidden * sizeof(std::uint16_t));
    ++result.seed_tensors;
    result.seed_bytes += seed_native_oracle_tensor(
        oracle_file(options.post_attention_residual_oracle_label),
        after_attention,
        comparison_tokens * kHidden * sizeof(std::uint16_t));
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
  const std::size_t gemm_tokens = gemm_plans->token_count();
  if (gemm_tokens < tokens || gemm_tokens > bucket_tokens) {
    throw std::invalid_argument("native MoE prefill GEMM context mismatch");
  }
  NativeQ8192PrefillGemmPlans* logical_router_gemm_plans =
      options.logical_router_gemm_plans;
  if (logical_router_gemm_plans != nullptr &&
      (logical_router_gemm_plans->token_count() < comparison_tokens ||
       logical_router_gemm_plans->token_count() > bucket_tokens ||
       logical_router_gemm_plans->token_count() != gemm_tokens)) {
    throw std::invalid_argument(
        "native MoE logical router GEMM context mismatch");
  }
  diagnostic_stage("before_shared_gate_plan");
  Bf16GemmPlan& shared_gate_plan = gemm_plans->moe_shared_gate();
  diagnostic_stage("after_shared_gate_plan");
  Bf16GemmPlan& shared_projection_plan =
      gemm_plans->moe_shared_projection();
  diagnostic_stage("after_shared_projection_plan");
  Bf16GemmPlan& shared_down_plan = gemm_plans->moe_shared_down();
  diagnostic_stage("after_shared_down_plan");
  Bf16GemmPlan& router_plan =
      logical_router_gemm_plans == nullptr
          ? gemm_plans->moe_router()
          : logical_router_gemm_plans->moe_router();
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
                           shared_activated,
                           options.use_vl_shared_expert_semantics, tokens);
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

  if (logical_router_gemm_plans != nullptr) {
    check_hip(hipMemsetAsync(
                  router_logits, 0,
                  tokens * kExperts * sizeof(std::uint16_t), nullptr),
              "hipMemsetAsync native MoE padded router logits");
  }
  router_plan.launch(h2, router_weight.device_pointer, router_logits);
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native MoE router projection");
  }
  diagnostic_stage("after_router_projection");
  ++result.layer.dense_gemm_launches;
  launch_router(router_logits, router_scores,
                router_indices_i64,
                router_indices_i32, topk_weights,
                router_weights_are_bfloat16,
                options.use_vl_router_semantics, tokens);
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
  // The embedded fused kernels retain the bucket's maximum launch grid and
  // storage geometry.  Their num_valid_tokens scalar is the semantic guard
  // that excludes dispatch sentinels beyond the causal active prefix.
  invocations.set_int32_argument(
      moe_first, "num_valid_tokens", static_cast<std::int32_t>(routed_rows));
  invocations.set_int32_argument(
      moe_first + 1, "num_valid_tokens",
      static_cast<std::int32_t>(routed_rows));
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
          "router_weights", router_weight_dtype, topk_weights,
          tokens * kTopK * router_weight_element_bytes,
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
            : comparison_tokens * kHidden * sizeof(std::uint16_t);
    const auto* comparison_pointer =
        static_cast<const unsigned char*>(layer_output) +
        (options.chain_output_last_token_only
             ? (comparison_tokens - 1) * kHidden * sizeof(std::uint16_t)
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
      compare_stage_if_present(
          "h2", "bfloat16", h2,
          comparison_tokens * kHidden * sizeof(std::uint16_t));
      compare_stage_if_present("router_logits", "bfloat16", router_logits,
                               comparison_tokens * kExperts *
                                   sizeof(std::uint16_t));
      compare_stage_if_present("router_scores", "float32", router_scores,
                               comparison_tokens * kTopK * sizeof(float));
      compare_stage_if_present(
          "router_weights", router_weight_dtype, topk_weights,
          comparison_tokens * kTopK * router_weight_element_bytes);
      compare_stage_if_present("router_indices", "int64", router_indices_i64,
                               comparison_tokens * kTopK *
                                   sizeof(std::int64_t));
      const std::filesystem::path router_indices_expected =
          find_native_oracle_tensor_file_if_present(
              options.chain_output_oracle_dir,
              label_prefix + "router_indices");
      if (!router_indices_expected.empty()) {
        result.router_expert_set_rows = comparison_tokens;
        result.router_expert_set_rows_exact = count_exact_int64_row_sets(
            router_indices_i64, router_indices_expected, comparison_tokens,
            kTopK);
        result.router_expert_sets_exact =
            result.router_expert_set_rows_exact ==
            result.router_expert_set_rows;
      }
      compare_stage_if_present("shared_out", "bfloat16", shared_scaled,
                               comparison_tokens * kHidden *
                                   sizeof(std::uint16_t));
      compare_stage_if_present("routed_moe", "bfloat16", routed_moe,
                               comparison_tokens * kHidden *
                                   sizeof(std::uint16_t));
      compare_stage_if_present("moe_out", "bfloat16", combined_moe,
                               comparison_tokens * kHidden *
                                   sizeof(std::uint16_t));
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
        tokens * kTopK * router_weight_element_bytes);
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
