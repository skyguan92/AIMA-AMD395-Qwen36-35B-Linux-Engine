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
    "b7457a70c1c87f2e9cbe2a99def59e8873877c7c174f83f84f668d0063886df1";
constexpr char kSparseGateUpCorrectionKernelHash[] =
    "cfe9a4fd5c5b2198d068a7ea38726d99e0f79374da9246109905e8ed316dfbf4";
constexpr char kDownKernelHash[] =
    "775c54180f9368197b9493aa15e604d3f7622519a20fc322e827e2d51a979b75";
constexpr AotLaunchConfig kHybridScalarGateUpLaunchConfig{8, 256, 1, 8, 32,
                                                          128};
constexpr AotLaunchConfig kSparseGateUpCorrectionLaunchConfig{128, 1, 1, 4,
                                                              32, 40960};
constexpr AotLaunchConfig kDownLaunchConfig{1024, 1, 1, 4, 32, 16384};

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

__device__ __forceinline__ float pytorch_rounded_rsqrtf(float value) {
  // Match the correctly rounded FP32 result produced by the pinned
  // PyTorch/ROCm RMSNorm boundary.  This is intentionally identical to the
  // singleton implementation in native_pointwise.hip.cpp: the compiler's
  // direct rsqrtf result can otherwise differ by one ULP.
  const float seed = rsqrtf(value);
  const double seed64 = static_cast<double>(seed);
  const unsigned value_exponent = __float_as_uint(value) & 0x7f800000U;
  bool seed_above_root = false;
  bool seed_below_root = false;
  if (value_exponent == 0U || value_exponent >= 0x7e800000U) {
    const double seed_test =
        static_cast<double>(value) * seed64 * seed64;
    seed_above_root = seed_test > 1.0;
    seed_below_root = seed_test < 1.0;
  } else {
    const float seed_squared = seed * seed;
    const float squared_error = fmaf(seed, seed, -seed_squared);
    const float product = value * seed_squared;
    const float product_error =
        fmaf(value, seed_squared, -product) + value * squared_error;
    seed_above_root =
        product > 1.0f || (product == 1.0f && product_error > 0.0f);
    seed_below_root =
        product < 1.0f || (product == 1.0f && product_error < 0.0f);
  }
  float selected = seed;
  if (seed_above_root) {
    const float lower = nextafterf(seed, 0.0f);
    const double midpoint =
        (static_cast<double>(lower) + seed64) * 0.5;
    const double midpoint_test =
        static_cast<double>(value) * midpoint * midpoint;
    if (midpoint_test > 1.0 ||
        (midpoint_test == 1.0 && (__float_as_uint(lower) & 1U) == 0)) {
      selected = lower;
    }
  } else if (seed_below_root) {
    const float upper = nextafterf(seed, __int_as_float(0x7f800000));
    const double midpoint =
        (seed64 + static_cast<double>(upper)) * 0.5;
    const double midpoint_test =
        static_cast<double>(value) * midpoint * midpoint;
    if (midpoint_test < 1.0 ||
        (midpoint_test == 1.0 && (__float_as_uint(upper) & 1U) == 0)) {
      selected = upper;
    }
  }
  return selected;
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
    std::int32_t* num_tokens_post_padded,
    std::int32_t* gate_up_correction_count) {
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
    *gate_up_correction_count = 0;
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

__global__ void native_decode_moe_tail_kernel(
    const __hip_bfloat16* weighted_expert_outputs,
    const __hip_bfloat16* shared_input,
    const __hip_bfloat16* shared_down,
    const __hip_bfloat16* residual,
    __hip_bfloat16* routed_output,
    __hip_bfloat16* shared_output,
    __hip_bfloat16* combined_output,
    __hip_bfloat16* output) {
  const std::size_t hidden = blockIdx.x * blockDim.x + threadIdx.x;
  if (hidden >= kHidden) return;

  const float sum04 =
      __bfloat162float(weighted_expert_outputs[hidden]) +
      __bfloat162float(weighted_expert_outputs[4 * kHidden + hidden]);
  const float sum15 =
      __bfloat162float(weighted_expert_outputs[kHidden + hidden]) +
      __bfloat162float(weighted_expert_outputs[5 * kHidden + hidden]);
  const float sum26 =
      __bfloat162float(weighted_expert_outputs[2 * kHidden + hidden]) +
      __bfloat162float(weighted_expert_outputs[6 * kHidden + hidden]);
  const float sum37 =
      __bfloat162float(weighted_expert_outputs[3 * kHidden + hidden]) +
      __bfloat162float(weighted_expert_outputs[7 * kHidden + hidden]);
  float routed_sum = sum04 + sum15;
  routed_sum += sum26;
  routed_sum += sum37;
  volatile __hip_bfloat16* volatile_routed_output = routed_output;
  const __hip_bfloat16 routed_converted = __float2bfloat16(routed_sum);
  volatile_routed_output[hidden] =
      static_cast<__hip_bfloat16_raw>(routed_converted);
  const __hip_bfloat16 routed_bf16(
      static_cast<__hip_bfloat16_raw>(volatile_routed_output[hidden]));

  const float gate_fp32 =
      1.0f / (1.0f + expf(-__bfloat162float(shared_input[0])));
  const __hip_bfloat16 gate_bf16 = __float2bfloat16(gate_fp32);
  volatile __hip_bfloat16* volatile_shared_output = shared_output;
  const __hip_bfloat16 shared_converted = __float2bfloat16(
      __bfloat162float(gate_bf16) *
      __bfloat162float(shared_down[hidden]));
  volatile_shared_output[hidden] =
      static_cast<__hip_bfloat16_raw>(shared_converted);
  const __hip_bfloat16 shared_bf16(
      static_cast<__hip_bfloat16_raw>(volatile_shared_output[hidden]));

  volatile __hip_bfloat16* volatile_combined_output = combined_output;
  const __hip_bfloat16 combined_converted = __float2bfloat16(
      __bfloat162float(routed_bf16) + __bfloat162float(shared_bf16));
  volatile_combined_output[hidden] =
      static_cast<__hip_bfloat16_raw>(combined_converted);
  const __hip_bfloat16 combined_bf16(
      static_cast<__hip_bfloat16_raw>(volatile_combined_output[hidden]));
  output[hidden] = __float2bfloat16(
      __bfloat162float(residual[hidden]) +
      __bfloat162float(combined_bf16));
}

__global__ void native_decode_moe_tail_next_rmsnorm_kernel(
    const __hip_bfloat16* weighted_expert_outputs,
    const __hip_bfloat16* shared_input,
    const __hip_bfloat16* shared_down,
    const __hip_bfloat16* residual,
    __hip_bfloat16* routed_output,
    __hip_bfloat16* shared_output,
    __hip_bfloat16* combined_output,
    __hip_bfloat16* output,
    const __hip_bfloat16* next_norm_weight,
    __hip_bfloat16* next_norm_output) {
  constexpr unsigned kVectorWidth = 4;
  constexpr unsigned kBlockThreads = kHidden / kVectorWidth;
  static_assert(kBlockThreads == 512);
  __shared__ float partials[kBlockThreads];
  __shared__ float shared_inverse_rms;

  const unsigned lane = threadIdx.x;
  const unsigned vector_base = lane * kVectorWidth;
  const float gate_fp32 =
      1.0f / (1.0f + expf(-__bfloat162float(shared_input[0])));
  const __hip_bfloat16 gate_bf16 = __float2bfloat16(gate_fp32);
  volatile __hip_bfloat16* volatile_routed_output = routed_output;
  volatile __hip_bfloat16* volatile_shared_output = shared_output;
  volatile __hip_bfloat16* volatile_combined_output = combined_output;
  volatile __hip_bfloat16* volatile_output = output;
  float sum = 0.0f;
#pragma unroll
  for (unsigned component = 0; component < kVectorWidth; ++component) {
    const unsigned hidden = vector_base + component;
    const float sum04 =
        __bfloat162float(weighted_expert_outputs[hidden]) +
        __bfloat162float(weighted_expert_outputs[4 * kHidden + hidden]);
    const float sum15 =
        __bfloat162float(weighted_expert_outputs[kHidden + hidden]) +
        __bfloat162float(weighted_expert_outputs[5 * kHidden + hidden]);
    const float sum26 =
        __bfloat162float(weighted_expert_outputs[2 * kHidden + hidden]) +
        __bfloat162float(weighted_expert_outputs[6 * kHidden + hidden]);
    const float sum37 =
        __bfloat162float(weighted_expert_outputs[3 * kHidden + hidden]) +
        __bfloat162float(weighted_expert_outputs[7 * kHidden + hidden]);
    float routed_sum = sum04 + sum15;
    routed_sum += sum26;
    routed_sum += sum37;
    const __hip_bfloat16 routed_converted = __float2bfloat16(routed_sum);
    volatile_routed_output[hidden] =
        static_cast<__hip_bfloat16_raw>(routed_converted);
    const __hip_bfloat16 routed_bf16(
        static_cast<__hip_bfloat16_raw>(volatile_routed_output[hidden]));

    const __hip_bfloat16 shared_converted = __float2bfloat16(
        __bfloat162float(gate_bf16) *
        __bfloat162float(shared_down[hidden]));
    volatile_shared_output[hidden] =
        static_cast<__hip_bfloat16_raw>(shared_converted);
    const __hip_bfloat16 shared_bf16(
        static_cast<__hip_bfloat16_raw>(volatile_shared_output[hidden]));

    const __hip_bfloat16 combined_converted = __float2bfloat16(
        __bfloat162float(routed_bf16) + __bfloat162float(shared_bf16));
    volatile_combined_output[hidden] =
        static_cast<__hip_bfloat16_raw>(combined_converted);
    const __hip_bfloat16 combined_bf16(
        static_cast<__hip_bfloat16_raw>(volatile_combined_output[hidden]));

    const __hip_bfloat16 output_converted = __float2bfloat16(
        __bfloat162float(residual[hidden]) +
        __bfloat162float(combined_bf16));
    volatile_output[hidden] =
        static_cast<__hip_bfloat16_raw>(output_converted);
    const __hip_bfloat16 output_bf16(
        static_cast<__hip_bfloat16_raw>(volatile_output[hidden]));
    const float value = __bfloat162float(output_bf16);
    volatile float squared = value * value;
    sum = sum + squared;
  }
  partials[lane] = sum;

  // Replay the exact singleton [1, 2048] ATen reduction used by the
  // standalone input RMSNorm.  Wave 0 combines the sixteen groups and
  // evaluates the correctly-rounded rsqrt once, then broadcasts it through
  // LDS.  This preserves the reduction tree while avoiding sixteen copies of
  // the FP64-assisted scalar tail.
  __syncthreads();
  const unsigned wave_lane = lane & 31;
  if (lane < 32) {
    float wave_sums[16];
#pragma unroll
    for (unsigned wave = 0; wave < 16; ++wave) {
      wave_sums[wave] = partials[wave_lane + wave * 32];
    }
    for (unsigned span = 8; span > 0; span >>= 1) {
#pragma unroll
      for (unsigned wave = 0; wave < span; ++wave) {
        wave_sums[wave] = wave_sums[wave] + wave_sums[wave + span];
      }
    }
    float sum = wave_sums[0];
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
      sum = sum + __shfl_down(sum, offset, 32);
    }
    if (wave_lane == 0) {
      volatile float variance =
          sum * (1.0f / static_cast<float>(kHidden));
      volatile float variance_with_epsilon = variance + 1.0e-6f;
      shared_inverse_rms =
          pytorch_rounded_rsqrtf(variance_with_epsilon);
    }
  }
  __syncthreads();
  const float inverse_rms = shared_inverse_rms;

#pragma unroll
  for (unsigned component = 0; component < kVectorWidth; ++component) {
    const unsigned hidden = vector_base + component;
    const __hip_bfloat16 output_bf16(
        static_cast<__hip_bfloat16_raw>(volatile_output[hidden]));
    const float value = __bfloat162float(output_bf16);
    volatile float scaled = value * inverse_rms;
    volatile float weight_plus_one =
        __bfloat162float(next_norm_weight[hidden]) + 1.0f;
    volatile float weighted = scaled * weight_plus_one;
    next_norm_output[hidden] = __float2bfloat16(weighted);
  }
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

void launch_native_decode_moe_tail(
    const void* weighted_expert_outputs_bf16,
    const void* fused_shared_input_bf16, const void* shared_down_bf16,
    const void* residual_bf16, void* routed_output_bf16,
    void* shared_output_bf16, void* combined_output_bf16,
    void* output_bf16, void* stream) {
  if (weighted_expert_outputs_bf16 == nullptr ||
      fused_shared_input_bf16 == nullptr || shared_down_bf16 == nullptr ||
      residual_bf16 == nullptr || routed_output_bf16 == nullptr ||
      shared_output_bf16 == nullptr || combined_output_bf16 == nullptr ||
      output_bf16 == nullptr) {
    throw std::invalid_argument(
        "native decode MoE tail requires non-null pointers");
  }
  hipLaunchKernelGGL(
      native_decode_moe_tail_kernel,
      dim3(static_cast<unsigned>((kHidden + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, static_cast<hipStream_t>(stream),
      static_cast<const __hip_bfloat16*>(weighted_expert_outputs_bf16),
      static_cast<const __hip_bfloat16*>(fused_shared_input_bf16),
      static_cast<const __hip_bfloat16*>(shared_down_bf16),
      static_cast<const __hip_bfloat16*>(residual_bf16),
      static_cast<__hip_bfloat16*>(routed_output_bf16),
      static_cast<__hip_bfloat16*>(shared_output_bf16),
      static_cast<__hip_bfloat16*>(combined_output_bf16),
      static_cast<__hip_bfloat16*>(output_bf16));
  check_hip(hipGetLastError(), "native_decode_moe_tail_kernel");
}

void launch_native_decode_moe_tail_next_rmsnorm(
    const void* weighted_expert_outputs_bf16,
    const void* fused_shared_input_bf16, const void* shared_down_bf16,
    const void* residual_bf16, void* routed_output_bf16,
    void* shared_output_bf16, void* combined_output_bf16,
    void* output_bf16, const void* next_norm_weight_bf16,
    void* next_norm_output_bf16, void* stream) {
  if (weighted_expert_outputs_bf16 == nullptr ||
      fused_shared_input_bf16 == nullptr || shared_down_bf16 == nullptr ||
      residual_bf16 == nullptr || routed_output_bf16 == nullptr ||
      shared_output_bf16 == nullptr || combined_output_bf16 == nullptr ||
      output_bf16 == nullptr || next_norm_weight_bf16 == nullptr ||
      next_norm_output_bf16 == nullptr) {
    throw std::invalid_argument(
        "native decode MoE tail plus next RMSNorm requires non-null pointers");
  }
  hipLaunchKernelGGL(
      native_decode_moe_tail_next_rmsnorm_kernel, dim3(1), dim3(512), 0,
      static_cast<hipStream_t>(stream),
      static_cast<const __hip_bfloat16*>(weighted_expert_outputs_bf16),
      static_cast<const __hip_bfloat16*>(fused_shared_input_bf16),
      static_cast<const __hip_bfloat16*>(shared_down_bf16),
      static_cast<const __hip_bfloat16*>(residual_bf16),
      static_cast<__hip_bfloat16*>(routed_output_bf16),
      static_cast<__hip_bfloat16*>(shared_output_bf16),
      static_cast<__hip_bfloat16*>(combined_output_bf16),
      static_cast<__hip_bfloat16*>(output_bf16),
      static_cast<const __hip_bfloat16*>(next_norm_weight_bf16),
      static_cast<__hip_bfloat16*>(next_norm_output_bf16));
  check_hip(hipGetLastError(),
            "native_decode_moe_tail_next_rmsnorm_kernel");
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
    NativeDecodeExecutor& executor, void* stream, bool defer_sum) {
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
      static_cast<std::int32_t*>(buffers.num_tokens_post_padded_i32),
      static_cast<std::int32_t*>(buffers.activation_bf16));
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
  if (!defer_sum) {
    hipLaunchKernelGGL(
        routed_sum8_kernel,
        dim3(static_cast<unsigned>((kHidden + kThreads - 1) / kThreads)),
        dim3(kThreads), 0, hip_stream,
        static_cast<const __hip_bfloat16*>(
            buffers.weighted_expert_outputs_bf16),
        static_cast<__hip_bfloat16*>(buffers.output_bf16));
    check_hip(hipGetLastError(), "routed_sum8_kernel");
    ++metrics.native_pointwise_launches;
  }
  return metrics;
}

}  // namespace aima
