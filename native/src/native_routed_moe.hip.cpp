// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_routed_moe.h"

#include "aima/bf16_wvsplitk.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima {
namespace {

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kExperts = 256;
constexpr std::size_t kTopK = 8;
constexpr std::size_t kIntermediate = 512;
constexpr std::size_t kGateUp = 2 * kIntermediate;
constexpr unsigned kThreads = 256;
constexpr char kHybridScalarGateUpKernelHash[] =
    "2aabec08044ef14f7f5f08e4854473bd85f15e17feb31d52b10cdf94a801a4ce";
constexpr char kSparseGateUpCorrectionKernelHash[] =
    "0e6da1f589b4787c411264ea8288e26cb4259c61da6f88e1a1f7d8b4e3e74dab";
constexpr char kDownKernelHash[] =
    "775c54180f9368197b9493aa15e604d3f7622519a20fc322e827e2d51a979b75";
constexpr AotLaunchConfig kHybridScalarGateUpLaunchConfig{8, 256, 1, 8, 32,
                                                          128};
constexpr AotLaunchConfig kSparseGateUpCorrectionLaunchConfig{512, 1, 1, 4,
                                                              32, 16384};
constexpr AotLaunchConfig kDownLaunchConfig{1024, 1, 1, 4, 32, 16384};

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

class SharedExpertOverlap {
 public:
  SharedExpertOverlap() {
    check_hip(hipStreamCreateWithFlags(&side_stream_, hipStreamNonBlocking),
              "hipStreamCreateWithFlags shared expert");
    check_hip(hipEventCreateWithFlags(&input_ready_, hipEventDisableTiming),
              "hipEventCreateWithFlags shared expert input");
    check_hip(hipEventCreateWithFlags(&output_ready_, hipEventDisableTiming),
              "hipEventCreateWithFlags shared expert output");
  }

  ~SharedExpertOverlap() {
    if (output_ready_ != nullptr) {
      const hipError_t ignored = hipEventDestroy(output_ready_);
      static_cast<void>(ignored);
    }
    if (input_ready_ != nullptr) {
      const hipError_t ignored = hipEventDestroy(input_ready_);
      static_cast<void>(ignored);
    }
    if (side_stream_ != nullptr) {
      const hipError_t ignored = hipStreamDestroy(side_stream_);
      static_cast<void>(ignored);
    }
  }

  void* begin(hipStream_t main_stream) {
    check_hip(hipEventRecord(input_ready_, main_stream),
              "hipEventRecord shared expert input");
    check_hip(hipStreamWaitEvent(side_stream_, input_ready_, 0),
              "hipStreamWaitEvent shared expert input");
    return side_stream_;
  }

  void complete(hipStream_t main_stream) {
    check_hip(hipEventRecord(output_ready_, side_stream_),
              "hipEventRecord shared expert output");
    check_hip(hipStreamWaitEvent(main_stream, output_ready_, 0),
              "hipStreamWaitEvent shared expert output");
  }

 private:
  hipStream_t side_stream_ = nullptr;
  hipEvent_t input_ready_ = nullptr;
  hipEvent_t output_ready_ = nullptr;
};

SharedExpertOverlap& shared_expert_overlap() {
  static SharedExpertOverlap overlap;
  return overlap;
}

void require_complete(const NativeDecodeRoutedMoeBuffers& buffers) {
  if (buffers.router_logits_bf16 == nullptr ||
      buffers.router_weights_fp32 == nullptr ||
      buffers.router_indices_i32 == nullptr ||
      buffers.num_tokens_post_padded_i32 == nullptr ||
      buffers.gate_up_bf16 == nullptr || buffers.activation_bf16 == nullptr ||
      buffers.weighted_expert_outputs_bf16 == nullptr ||
      buffers.output_bf16 == nullptr) {
    throw std::invalid_argument("native decode routed-MoE buffers are incomplete");
  }
}

__global__ void router_topk8_softmax_256_kernel(
    const __hip_bfloat16* logits, float* weights, std::int32_t* indices,
    std::int32_t* num_tokens_post_padded) {
  constexpr int kRouterWave = 32;
  constexpr int kValuesPerLane = kExperts / kRouterWave;
  const int lane = threadIdx.x;
  float row_chunk[kValuesPerLane];
#pragma unroll
  for (int value = 0; value < kValuesPerLane; ++value) {
    row_chunk[value] =
        __bfloat162float(logits[lane * kValuesPerLane + value]);
  }

  float thread_max = row_chunk[0];
#pragma unroll
  for (int value = 1; value < kValuesPerLane; ++value) {
    thread_max = fmaxf(thread_max, row_chunk[value]);
  }
#pragma unroll
  for (int mask = kRouterWave / 2; mask > 0; mask /= 2) {
    thread_max = fmaxf(thread_max,
                       __shfl_xor(thread_max, mask, kRouterWave));
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
#pragma unroll
    for (std::size_t rank = 0; rank < kTopK; ++rank) {
      indices[rank] = selected_indices[rank];
      weights[rank] = selected_probabilities[rank] / denominator;
    }
    *num_tokens_post_padded = 128;
  }
}

__global__ void routed_silu_multiply_kernel(
    const __hip_bfloat16* gate_up, __hip_bfloat16* activated) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kTopK * kIntermediate) return;
  const std::size_t row = index / kIntermediate;
  const std::size_t column = index - row * kIntermediate;
  const std::size_t base = row * kGateUp;
  const float gate = __bfloat162float(gate_up[base + column]);
  const float up =
      __bfloat162float(gate_up[base + kIntermediate + column]);
  const float silu = gate / (1.0f + expf(-gate));
  const __hip_bfloat16 silu_bf16 = __float2bfloat16(silu);
  activated[index] =
      __float2bfloat16(__bfloat162float(silu_bf16) * up);
}

__global__ void routed_sum8_kernel(const __hip_bfloat16* input,
                                   __hip_bfloat16* output) {
  const std::size_t hidden = blockIdx.x * blockDim.x + threadIdx.x;
  if (hidden >= kHidden) return;
  const float sum04 = __bfloat162float(input[hidden]) +
                      __bfloat162float(input[4 * kHidden + hidden]);
  const float sum15 = __bfloat162float(input[kHidden + hidden]) +
                      __bfloat162float(input[5 * kHidden + hidden]);
  const float sum26 = __bfloat162float(input[2 * kHidden + hidden]) +
                      __bfloat162float(input[6 * kHidden + hidden]);
  const float sum37 = __bfloat162float(input[3 * kHidden + hidden]) +
                      __bfloat162float(input[7 * kHidden + hidden]);
  float sum = sum04 + sum15;
  sum += sum26;
  sum += sum37;
  output[hidden] = __float2bfloat16(sum);
}

void launch_fused_moe(
    const char* kernel_hash, const AotLaunchConfig& config,
    void* activation, void* weight, void* output, void* topk_weights,
    void* expert_ids, void* num_tokens_post_padded, std::int32_t n,
    std::int32_t k, std::int32_t stride_be, std::int32_t stride_bn,
    std::int32_t stride_am, std::int32_t stride_cm,
    NativeDecodeExecutor& executor, void* stream) {
  std::int32_t em = 128;
  std::int32_t num_valid_tokens = 8;
  std::int32_t zero = 0;
  const std::vector<void*> parameters = {
      &activation,
      &weight,
      &output,
      &topk_weights,
      &expert_ids,
      &num_tokens_post_padded,
      &n,
      &k,
      &em,
      &num_valid_tokens,
      &stride_am,
      &stride_be,
      &stride_bn,
      &stride_cm,
      &zero,
      &zero,
      &zero,
      &zero,
      &zero,
      &zero,
      &zero,
  };
  executor.launch_embedded(kernel_hash, config, parameters, stream);
}

void launch_hybrid_gate_up(void* hidden, void* weight, void* expert_ids,
                           void* output, void* flagged_indices,
                           void* flagged_count,
                           NativeDecodeExecutor& executor, void* stream) {
  check_hip(hipMemsetAsync(flagged_count, 0, sizeof(std::int32_t),
                           static_cast<hipStream_t>(stream)),
            "hipMemsetAsync routed gate/up correction count");
  const std::vector<void*> parameters = {
      &hidden,
      &weight,
      &expert_ids,
      &output,
      &flagged_indices,
      &flagged_count,
  };
  executor.launch_embedded(kHybridScalarGateUpKernelHash,
                           kHybridScalarGateUpLaunchConfig, parameters,
                           stream);
  executor.launch_embedded(kSparseGateUpCorrectionKernelHash,
                           kSparseGateUpCorrectionLaunchConfig, parameters,
                           stream);
}

}  // namespace

void* begin_native_decode_shared_expert_overlap(void* main_stream) {
  return shared_expert_overlap().begin(static_cast<hipStream_t>(main_stream));
}

void complete_native_decode_shared_expert_overlap(void* main_stream) {
  shared_expert_overlap().complete(static_cast<hipStream_t>(main_stream));
}

NativeDecodeRoutedMoeMetrics run_native_decode_routed_moe(
    const void* hidden_bf16, const void* router_weight_bf16,
    const void* gate_up_weight_bf16, const void* down_weight_bf16,
    const NativeDecodeRoutedMoeBuffers& buffers,
    NativeDecodeExecutor& executor, int cu_count, void* stream) {
  if (hidden_bf16 == nullptr || router_weight_bf16 == nullptr ||
      gate_up_weight_bf16 == nullptr || down_weight_bf16 == nullptr ||
      !executor.loaded() || cu_count <= 0) {
    throw std::invalid_argument("native decode routed-MoE owners are incomplete");
  }
  require_complete(buffers);
  NativeDecodeRoutedMoeMetrics metrics;
  launch_bf16_wvsplitk(router_weight_bf16, hidden_bf16, nullptr,
                       buffers.router_logits_bf16, kExperts, kHidden,
                       cu_count, stream);
  ++metrics.native_projection_launches;
  const NativeDecodeRoutedMoeMetrics tail =
      run_native_decode_routed_moe_from_logits(
          hidden_bf16, gate_up_weight_bf16, down_weight_bf16, buffers,
          executor, stream);
  metrics.aot_launches += tail.aot_launches;
  metrics.native_projection_launches += tail.native_projection_launches;
  metrics.native_pointwise_launches += tail.native_pointwise_launches;
  return metrics;
}

NativeDecodeRoutedMoeMetrics run_native_decode_routed_moe_from_logits(
    const void* hidden_bf16, const void* gate_up_weight_bf16,
    const void* down_weight_bf16,
    const NativeDecodeRoutedMoeBuffers& buffers,
    NativeDecodeExecutor& executor, void* stream) {
  if (hidden_bf16 == nullptr || gate_up_weight_bf16 == nullptr ||
      down_weight_bf16 == nullptr || !executor.loaded()) {
    throw std::invalid_argument(
        "native decode routed-MoE owners are incomplete");
  }
  require_complete(buffers);
  NativeDecodeRoutedMoeMetrics metrics;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  hipLaunchKernelGGL(
      router_topk8_softmax_256_kernel, dim3(1), dim3(32), 0, hip_stream,
      static_cast<const __hip_bfloat16*>(buffers.router_logits_bf16),
      static_cast<float*>(buffers.router_weights_fp32),
      static_cast<std::int32_t*>(buffers.router_indices_i32),
      static_cast<std::int32_t*>(buffers.num_tokens_post_padded_i32));
  check_hip(hipGetLastError(), "router_topk8_softmax_256_kernel");
  ++metrics.native_pointwise_launches;

  launch_hybrid_gate_up(
      const_cast<void*>(hidden_bf16),
      const_cast<void*>(gate_up_weight_bf16), buffers.router_indices_i32,
      buffers.gate_up_bf16, buffers.weighted_expert_outputs_bf16,
      buffers.activation_bf16, executor, stream);
  metrics.aot_launches += 2;
  constexpr std::size_t activation_elements = kTopK * kIntermediate;
  hipLaunchKernelGGL(
      routed_silu_multiply_kernel,
      dim3(static_cast<unsigned>((activation_elements + kThreads - 1) /
                                 kThreads)),
      dim3(kThreads), 0, hip_stream,
      static_cast<const __hip_bfloat16*>(buffers.gate_up_bf16),
      static_cast<__hip_bfloat16*>(buffers.activation_bf16));
  check_hip(hipGetLastError(), "routed_silu_multiply_kernel");
  ++metrics.native_pointwise_launches;

  launch_fused_moe(
      kDownKernelHash, kDownLaunchConfig, buffers.activation_bf16,
      const_cast<void*>(down_weight_bf16),
      buffers.weighted_expert_outputs_bf16, buffers.router_weights_fp32,
      buffers.router_indices_i32, buffers.num_tokens_post_padded_i32,
      2048, 512, 1048576, 512, 512, 2048, executor, stream);
  ++metrics.aot_launches;
  hipLaunchKernelGGL(
      routed_sum8_kernel,
      dim3(static_cast<unsigned>((kHidden + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, hip_stream,
      static_cast<const __hip_bfloat16*>(
          buffers.weighted_expert_outputs_bf16),
      static_cast<__hip_bfloat16*>(buffers.output_bf16));
  check_hip(hipGetLastError(), "routed_sum8_kernel");
  ++metrics.native_pointwise_launches;
  return metrics;
}

}  // namespace aima
