// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_pointwise.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace aima {
namespace {

constexpr unsigned kThreads = 256;
constexpr std::size_t kHidden = 2048;
constexpr std::size_t kVocabulary = 248320;
constexpr std::size_t kSharedIntermediate = 512;
constexpr std::size_t kFullAttentionFeatures = 4096;
constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kFusedHeadStride = 512;
constexpr unsigned kInvstdWidth = 128;
constexpr std::size_t kMaxPrefillTokens = 262144;
constexpr std::size_t kQueryHeads = 16;
constexpr std::size_t kKvHeads = 2;
constexpr std::size_t kRotaryDim = 64;
constexpr std::size_t kRotaryPairs = kRotaryDim / 2;
constexpr float kRopeTheta = 10000000.0f;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

__device__ __forceinline__ float pytorch_rounded_rsqrtf(float value) {
  // The frozen PyTorch/ROCm build returns the correctly rounded FP32
  // reciprocal square root, while this package's ROCm compiler maps rsqrtf
  // to a result that can be one ULP away. Start with the fast estimate and
  // select its correctly rounded neighbor by testing the exact-root midpoint
  // in FP64. This avoids a throughput-heavy FP64 sqrt/divide.
  const float seed = rsqrtf(value);
  const double seed64 = static_cast<double>(seed);
  const double seed_test = static_cast<double>(value) * seed64 * seed64;
  float selected = seed;
  if (seed_test > 1.0) {
    const float lower = nextafterf(seed, 0.0f);
    const double midpoint =
        (static_cast<double>(lower) + seed64) * 0.5;
    const double midpoint_test =
        static_cast<double>(value) * midpoint * midpoint;
    if (midpoint_test > 1.0 ||
        (midpoint_test == 1.0 && (__float_as_uint(lower) & 1U) == 0)) {
      selected = lower;
    }
  } else if (seed_test < 1.0) {
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

__global__ void prompt_embeddings_kernel(
    const __hip_bfloat16* embedding, const std::uint32_t* token_ids,
    __hip_bfloat16* output, std::size_t token_count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t elements = token_count * kHidden;
  if (index >= elements) return;
  const std::size_t token_index = index / kHidden;
  const std::size_t hidden_index = index - token_index * kHidden;
  output[index] = embedding[
      static_cast<std::size_t>(token_ids[token_index]) * kHidden +
      hidden_index];
}

__global__ void bf16_add_kernel(const __hip_bfloat16* left,
                                const __hip_bfloat16* right,
                                __hip_bfloat16* output,
                                std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  output[index] = __float2bfloat16(
      __bfloat162float(left[index]) + __bfloat162float(right[index]));
}

__global__ void bf16_add_pair_kernel(const __hip_bfloat16* left,
                                     const __hip_bfloat16* right,
                                     const __hip_bfloat16* residual,
                                     __hip_bfloat16* intermediate,
                                     __hip_bfloat16* output,
                                     std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const __hip_bfloat16 combined = __float2bfloat16(
      __bfloat162float(left[index]) + __bfloat162float(right[index]));
  intermediate[index] = combined;
  output[index] = __float2bfloat16(
      __bfloat162float(residual[index]) + __bfloat162float(combined));
}

__global__ void prefill_rmsnorm_2048_kernel(
    const __hip_bfloat16* input, const __hip_bfloat16* weight,
    __hip_bfloat16* output, std::size_t token_count) {
  constexpr unsigned kVectorWidth = 4;
  constexpr unsigned kRowsPerBlock = 16;
  constexpr unsigned kRowThreads = 32;
  constexpr unsigned kIterations =
      kHidden / (kVectorWidth * kRowThreads);
  const unsigned lane = threadIdx.x;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * kRowsPerBlock + threadIdx.y;
  if (row >= token_count) return;
  const std::size_t row_offset =
      row * kHidden;

  // Match ATen's contiguous-input MeanOps reduction for [rows, 2048]: a
  // 32x16 workgroup, four independent vector-component accumulators per x
  // lane, 16 vector loads separated by 128 values, a left-associated merge
  // of the four accumulators, then ROCm's ascending-offset shuffle tree.
  float accumulator[kVectorWidth] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (unsigned iteration = 0; iteration < kIterations; ++iteration) {
    const unsigned vector_base =
        lane * kVectorWidth + iteration * kRowThreads * kVectorWidth;
#pragma unroll
    for (unsigned component = 0; component < kVectorWidth; ++component) {
      const float value =
          __bfloat162float(input[row_offset + vector_base + component]);
      // Eager PyTorch materializes pow(2) before mean.  Keep that FP32
      // boundary so the compiler cannot contract square-plus-accumulate.
      volatile float squared = value * value;
      accumulator[component] = accumulator[component] + squared;
    }
  }
  float sum = accumulator[0];
#pragma unroll
  for (unsigned component = 1; component < kVectorWidth; ++component) {
    sum = sum + accumulator[component];
  }
  for (unsigned offset = 1; offset < kRowThreads; offset <<= 1) {
    sum = sum + __shfl_down(sum, offset, kRowThreads);
  }

  float inverse_rms = 0.0f;
  if (lane == 0) {
    volatile float variance = sum * (1.0f / static_cast<float>(kHidden));
    volatile float variance_with_epsilon = variance + 1.0e-6f;
    inverse_rms = pytorch_rounded_rsqrtf(variance_with_epsilon);
  }
  inverse_rms = __shfl(inverse_rms, 0, kRowThreads);

#pragma unroll
  for (unsigned iteration = 0; iteration < kIterations; ++iteration) {
    const unsigned vector_base =
        lane * kVectorWidth + iteration * kRowThreads * kVectorWidth;
#pragma unroll
    for (unsigned component = 0; component < kVectorWidth; ++component) {
      const unsigned hidden = vector_base + component;
      const float value = __bfloat162float(input[row_offset + hidden]);
      volatile float scaled = value * inverse_rms;
      volatile float weight_plus_one =
          __bfloat162float(weight[hidden]) + 1.0f;
      volatile float weighted = scaled * weight_plus_one;
      output[row_offset + hidden] = __float2bfloat16(weighted);
    }
  }
}

__global__ void singleton_rmsnorm_2048_kernel(
    const __hip_bfloat16* input, const __hip_bfloat16* weight,
    __hip_bfloat16* output) {
  constexpr unsigned kVectorWidth = 4;
  constexpr unsigned kBlockThreads = kHidden / kVectorWidth;
  static_assert(kBlockThreads == 512);
  __shared__ float partials[kBlockThreads];

  const unsigned lane = threadIdx.x;
  const unsigned vector_base = lane * kVectorWidth;
  float sum = 0.0f;
#pragma unroll
  for (unsigned component = 0; component < kVectorWidth; ++component) {
    const float value =
        __bfloat162float(input[vector_base + component]);
    // Eager PyTorch materializes pow(2) before MeanOps reduction.
    volatile float squared = value * value;
    sum = sum + squared;
  }
  partials[lane] = sum;

  // ATen's singleton [1, 2048] reduction vectorizes the input across one
  // 512-thread block. Reduce its sixteen gfx1151 wave32 groups through shared
  // memory, then preserve ROCm's ascending-offset shuffle tree in wave 0.
  for (unsigned offset = kBlockThreads / 2; offset >= 32; offset >>= 1) {
    __syncthreads();
    if (lane < offset) {
      sum = sum + partials[lane + offset];
      partials[lane] = sum;
    }
  }
  __syncthreads();
  if (lane < 32) {
    sum = partials[lane];
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
      sum = sum + __shfl_down(sum, offset, 32);
    }
    if (lane == 0) {
      volatile float variance =
          sum * (1.0f / static_cast<float>(kHidden));
      volatile float variance_with_epsilon = variance + 1.0e-6f;
      partials[0] = pytorch_rounded_rsqrtf(variance_with_epsilon);
    }
  }
  __syncthreads();
  const float inverse_rms = partials[0];

#pragma unroll
  for (unsigned component = 0; component < kVectorWidth; ++component) {
    const unsigned hidden = vector_base + component;
    const float value = __bfloat162float(input[hidden]);
    volatile float scaled = value * inverse_rms;
    volatile float weight_plus_one =
        __bfloat162float(weight[hidden]) + 1.0f;
    volatile float weighted = scaled * weight_plus_one;
    output[hidden] = __float2bfloat16(weighted);
  }
}

__global__ void prefill_add_rmsnorm_2048_kernel(
    const __hip_bfloat16* input, const __hip_bfloat16* residual,
    const __hip_bfloat16* weight, __hip_bfloat16* residual_output,
    __hip_bfloat16* norm_output, std::size_t token_count) {
  constexpr unsigned kVectorWidth = 4;
  constexpr unsigned kRowsPerBlock = 16;
  constexpr unsigned kRowThreads = 32;
  constexpr unsigned kIterations =
      kHidden / (kVectorWidth * kRowThreads);
  const unsigned lane = threadIdx.x;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * kRowsPerBlock + threadIdx.y;
  if (row >= token_count) return;
  const std::size_t row_offset =
      row * kHidden;
  float accumulator[kVectorWidth] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (unsigned iteration = 0; iteration < kIterations; ++iteration) {
    const unsigned vector_base =
        lane * kVectorWidth + iteration * kRowThreads * kVectorWidth;
#pragma unroll
    for (unsigned component = 0; component < kVectorWidth; ++component) {
      const unsigned hidden = vector_base + component;
      const __hip_bfloat16 rounded = __float2bfloat16(
          __bfloat162float(input[row_offset + hidden]) +
          __bfloat162float(residual[row_offset + hidden]));
      residual_output[row_offset + hidden] = rounded;
      const float value = __bfloat162float(rounded);
      volatile float squared = value * value;
      accumulator[component] = accumulator[component] + squared;
    }
  }
  float sum = accumulator[0];
#pragma unroll
  for (unsigned component = 1; component < kVectorWidth; ++component) {
    sum = sum + accumulator[component];
  }
  for (unsigned offset = 1; offset < kRowThreads; offset <<= 1) {
    sum = sum + __shfl_down(sum, offset, kRowThreads);
  }

  float inverse_rms = 0.0f;
  if (lane == 0) {
    volatile float variance = sum * (1.0f / static_cast<float>(kHidden));
    volatile float variance_with_epsilon = variance + 1.0e-6f;
    inverse_rms = pytorch_rounded_rsqrtf(variance_with_epsilon);
  }
  inverse_rms = __shfl(inverse_rms, 0, kRowThreads);

#pragma unroll
  for (unsigned iteration = 0; iteration < kIterations; ++iteration) {
    const unsigned vector_base =
        lane * kVectorWidth + iteration * kRowThreads * kVectorWidth;
#pragma unroll
    for (unsigned component = 0; component < kVectorWidth; ++component) {
      const unsigned hidden = vector_base + component;
      const float value =
          __bfloat162float(residual_output[row_offset + hidden]);
      volatile float scaled = value * inverse_rms;
      volatile float weight_plus_one =
          __bfloat162float(weight[hidden]) + 1.0f;
      volatile float weighted = scaled * weight_plus_one;
      norm_output[row_offset + hidden] = __float2bfloat16(weighted);
    }
  }
}

__global__ void shared_activation_kernel(const __hip_bfloat16* shared_input,
                                         __hip_bfloat16* activated) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kSharedIntermediate) return;
  const float gate = __bfloat162float(shared_input[1 + index]);
  const float up =
      __bfloat162float(shared_input[1 + kSharedIntermediate + index]);
  // vLLM's BF16 SiluAndMul materializes the Silu result in the input dtype
  // before multiplying by the BF16 up projection.  Preserve that rounding
  // boundary instead of carrying Silu in FP32 through the final multiply.
  const __hip_bfloat16 silu_bf16 =
      __float2bfloat16(gate / (1.0f + expf(-gate)));
  activated[index] = __float2bfloat16(__bfloat162float(silu_bf16) * up);
}

__global__ void shared_gate_kernel(const __hip_bfloat16* shared_input,
                                   const __hip_bfloat16* shared_down,
                                   __hip_bfloat16* output) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kHidden) return;
  const float gate_fp32 =
      1.0f / (1.0f + expf(-__bfloat162float(shared_input[0])));
  const __hip_bfloat16 gate_bf16 = __float2bfloat16(gate_fp32);
  output[index] = __float2bfloat16(
      __bfloat162float(gate_bf16) * __bfloat162float(shared_down[index]));
}

__global__ void full_attention_gate_kernel(
    const __hip_bfloat16* attention,
    const __hip_bfloat16* fused_q_gate,
    __hip_bfloat16* output) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kFullAttentionFeatures) return;
  const std::size_t head = index / kHeadDim;
  const std::size_t dimension = index - head * kHeadDim;
  const std::size_t gate_index =
      head * kFusedHeadStride + kHeadDim + dimension;
  const float gate_fp32 =
      1.0f / (1.0f + expf(-__bfloat162float(fused_q_gate[gate_index])));
  const __hip_bfloat16 gate_bf16 = __float2bfloat16(gate_fp32);
  output[index] = __float2bfloat16(
      __bfloat162float(attention[index]) * __bfloat162float(gate_bf16));
}

__global__ void prefill_rotary_table_kernel(float* cosine, float* sine,
                                            std::size_t token_count,
                                            std::size_t position_start) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= token_count * kRotaryPairs) return;
  const std::size_t token = index / kRotaryPairs;
  const std::size_t pair = index - token * kRotaryPairs;
  const float exponent =
      static_cast<float>(2 * pair) / static_cast<float>(kRotaryDim);
  const float inverse_frequency = 1.0f / powf(kRopeTheta, exponent);
  const float angle =
      static_cast<float>(position_start + token) * inverse_frequency;
  cosine[index] = cosf(angle);
  sine[index] = sinf(angle);
}

__device__ __forceinline__ unsigned mrope_axis_for_pair(unsigned pair) {
  // Pinned Qwen3.6 interleaved mrope_section=[11,11,10].  This is the exact
  // predicate used by vLLM's Triton M-RoPE kernel, including its inclusive
  // section bounds.
  const bool height = pair % 3 == 1 && pair <= 3 * 11;
  const bool width = pair % 3 == 2 && pair <= 3 * 10;
  return height ? 1U : (width ? 2U : 0U);
}

__global__ void prefill_mrope_rotary_table_kernel(
    float* cosine, float* sine, const std::int64_t* positions,
    std::size_t token_count, std::size_t position_row_stride) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= token_count * kRotaryPairs) return;
  const std::size_t token = index / kRotaryPairs;
  const unsigned pair =
      static_cast<unsigned>(index - token * kRotaryPairs);
  const unsigned axis = mrope_axis_for_pair(pair);
  const std::int64_t position =
      positions[static_cast<std::size_t>(axis) * position_row_stride + token];
  const float exponent =
      static_cast<float>(2 * pair) / static_cast<float>(kRotaryDim);
  const float inverse_frequency = 1.0f / powf(kRopeTheta, exponent);
  const float angle = static_cast<float>(position) * inverse_frequency;
  // The pinned vLLM cache is created in FP32 and converted to the query's
  // BF16 dtype before triton_mrope consumes it.  Preserve that boundary while
  // retaining the existing FP32 workspace/consumer ABI.
  cosine[index] =
      __bfloat162float(__float2bfloat16(cosf(angle)));
  sine[index] =
      __bfloat162float(__float2bfloat16(sinf(angle)));
}

__device__ __forceinline__ float triton_bf16_product_rtz(float left,
                                                         float right) {
  volatile float product = left * right;
  return __uint_as_float(__float_as_uint(product) & 0xffff0000U);
}

template <std::size_t QRowStride, std::size_t KRowStride,
          std::size_t VRowStride, bool CopyV,
          bool TruncateRotaryProducts>
__global__ void full_attention_head_norm_rope_prefill_kernel(
    const __hip_bfloat16* q_gate, const __hip_bfloat16* k_raw,
    const __hip_bfloat16* v_raw,
    const __hip_bfloat16* q_weight, const __hip_bfloat16* k_weight,
    const float* cosine, const float* sine,
    __hip_bfloat16* q_output, __hip_bfloat16* k_output,
    __hip_bfloat16* v_output) {
  const unsigned lane = threadIdx.x;
  const std::size_t token = blockIdx.x;
  const std::size_t local_head = threadIdx.y;
  const bool query = local_head < kQueryHeads;
  const std::size_t head = query ? local_head : local_head - kQueryHeads;
  const __hip_bfloat16* input =
      query ? q_gate + token * QRowStride +
                         head * kFusedHeadStride
            : k_raw + token * KRowStride + head * kHeadDim;
  const __hip_bfloat16* weight = query ? q_weight : k_weight;

  // Match ATen's vectorized-input reduction for a contiguous 256-wide row.
  // The frozen eager path materializes pow(2) before mean, so keep every
  // square rounded independently instead of allowing a fused multiply-add.
  const unsigned vector_base = lane * 4;
  float accumulator[4];
#pragma unroll
  for (unsigned component = 0; component < 4; ++component) {
    const float first = __bfloat162float(input[vector_base + component]);
    const float second =
        __bfloat162float(input[128 + vector_base + component]);
    volatile float first_squared = first * first;
    volatile float second_squared = second * second;
    accumulator[component] = first_squared + second_squared;
  }
  float sum = accumulator[0];
#pragma unroll
  for (unsigned component = 1; component < 4; ++component) {
    sum = sum + accumulator[component];
  }
  for (unsigned offset = 1; offset < 32; offset <<= 1) {
    sum = sum + __shfl_down(sum, offset, 32);
  }
  float inverse_rms = 0.0f;
  if (lane == 0) {
    // mean, add(epsilon), and rsqrt are separate eager kernels. Preserve
    // their FP32 rounding boundaries while retaining the fused launch.
    volatile float variance = sum * (1.0f / 256.0f);
    volatile float variance_with_epsilon = variance + 1.0e-6f;
    inverse_rms = pytorch_rounded_rsqrtf(variance_with_epsilon);
  }
  inverse_rms = __shfl(inverse_rms, 0, 32);

#pragma unroll
  for (unsigned group = 0; group < 8; ++group) {
    const unsigned dimension = lane + group * 32;
    const float value = __bfloat162float(input[dimension]);
    volatile float scaled_storage = value * inverse_rms;
    volatile float weight_plus_one =
        __bfloat162float(weight[dimension]) + 1.0f;
    volatile float weighted_storage = scaled_storage * weight_plus_one;
    const __hip_bfloat16 normalized_bf16 =
        __float2bfloat16(weighted_storage);
    float output = __bfloat162float(normalized_bf16);
    if (dimension < kRotaryDim) {
      const unsigned mate_dimension =
          dimension < kRotaryPairs ? dimension + kRotaryPairs
                                   : dimension - kRotaryPairs;
      const float mate_value = __bfloat162float(input[mate_dimension]);
      volatile float mate_scaled_storage = mate_value * inverse_rms;
      volatile float mate_weight_plus_one =
          __bfloat162float(weight[mate_dimension]) + 1.0f;
      volatile float mate_weighted_storage =
          mate_scaled_storage * mate_weight_plus_one;
      const float mate = __bfloat162float(
          __float2bfloat16(mate_weighted_storage));
      const unsigned pair =
          dimension < kRotaryPairs ? dimension : dimension - kRotaryPairs;
      const float cos_value = cosine[token * kRotaryPairs + pair];
      const float sin_value = sine[token * kRotaryPairs + pair];
      float first_product = 0.0f;
      float second_product = 0.0f;
      if constexpr (TruncateRotaryProducts) {
        // Triton's AMD BF16 multiply lowers to RTZ, followed by an FP32
        // add/subtract and an RNE BF16 store.  Preserving both boundaries is
        // required for bit-exact Qwen3.6 M-RoPE consumption.
        first_product = triton_bf16_product_rtz(output, cos_value);
        second_product = triton_bf16_product_rtz(mate, sin_value);
      } else {
        volatile float first_product_storage = output * cos_value;
        volatile float second_product_storage = mate * sin_value;
        first_product = first_product_storage;
        second_product = second_product_storage;
      }
      output = dimension < kRotaryPairs
                   ? first_product - second_product
                   : first_product + second_product;
    }
    __hip_bfloat16* destination =
        query ? q_output + (token * kQueryHeads + head) * kHeadDim
              : k_output + (token * kKvHeads + head) * kHeadDim;
    destination[dimension] = __float2bfloat16(output);
    if constexpr (CopyV) {
      if (!query) {
        v_output[(token * kKvHeads + head) * kHeadDim + dimension] =
            v_raw[token * VRowStride + head * kHeadDim + dimension];
      }
    }
  }
}

template <bool TruncateRotaryProducts>
void launch_full_attention_head_norm_rope_prefill_impl(
    const void* q_gate, const void* k_raw, const void* v_raw,
    const void* q_norm_weight, const void* k_norm_weight,
    const void* cosine_fp32, const void* sine_fp32,
    void* q_output, void* k_output, void* v_output,
    std::size_t token_count, std::size_t q_row_stride,
    std::size_t k_row_stride, std::size_t v_row_stride,
    void* stream_value) {
  const bool split_projection =
      q_row_stride == 8192 && k_row_stride == 512 && v_row_stride == 0 &&
      v_raw == nullptr && v_output == nullptr;
  const bool fused_qk_projection =
      q_row_stride == 9216 && k_row_stride == 9216 && v_row_stride == 0 &&
      v_raw == nullptr && v_output == nullptr;
  const bool fused_projection =
      q_row_stride == 9216 && k_row_stride == 9216 &&
      v_row_stride == 9216 && v_raw != nullptr && v_output != nullptr;
  if (q_gate == nullptr || k_raw == nullptr || q_norm_weight == nullptr ||
      k_norm_weight == nullptr || cosine_fp32 == nullptr ||
      sine_fp32 == nullptr || q_output == nullptr || k_output == nullptr ||
      token_count == 0 || token_count > kMaxPrefillTokens ||
      (!split_projection && !fused_qk_projection && !fused_projection)) {
    throw std::invalid_argument(
        "native full-attention head norm geometry is unsupported");
  }
  if (split_projection) {
    hipLaunchKernelGGL(
        (full_attention_head_norm_rope_prefill_kernel<
            8192, 512, 0, false, TruncateRotaryProducts>),
        dim3(token_count), dim3(32, kQueryHeads + kKvHeads), 0,
        static_cast<hipStream_t>(stream_value),
        static_cast<const __hip_bfloat16*>(q_gate),
        static_cast<const __hip_bfloat16*>(k_raw), nullptr,
        static_cast<const __hip_bfloat16*>(q_norm_weight),
        static_cast<const __hip_bfloat16*>(k_norm_weight),
        static_cast<const float*>(cosine_fp32),
        static_cast<const float*>(sine_fp32),
        static_cast<__hip_bfloat16*>(q_output),
        static_cast<__hip_bfloat16*>(k_output), nullptr);
  } else if (fused_qk_projection) {
    hipLaunchKernelGGL(
        (full_attention_head_norm_rope_prefill_kernel<
            9216, 9216, 0, false, TruncateRotaryProducts>),
        dim3(token_count), dim3(32, kQueryHeads + kKvHeads), 0,
        static_cast<hipStream_t>(stream_value),
        static_cast<const __hip_bfloat16*>(q_gate),
        static_cast<const __hip_bfloat16*>(k_raw), nullptr,
        static_cast<const __hip_bfloat16*>(q_norm_weight),
        static_cast<const __hip_bfloat16*>(k_norm_weight),
        static_cast<const float*>(cosine_fp32),
        static_cast<const float*>(sine_fp32),
        static_cast<__hip_bfloat16*>(q_output),
        static_cast<__hip_bfloat16*>(k_output), nullptr);
  } else {
    hipLaunchKernelGGL(
        (full_attention_head_norm_rope_prefill_kernel<
            9216, 9216, 9216, true, TruncateRotaryProducts>),
        dim3(token_count), dim3(32, kQueryHeads + kKvHeads), 0,
        static_cast<hipStream_t>(stream_value),
        static_cast<const __hip_bfloat16*>(q_gate),
        static_cast<const __hip_bfloat16*>(k_raw),
        static_cast<const __hip_bfloat16*>(v_raw),
        static_cast<const __hip_bfloat16*>(q_norm_weight),
        static_cast<const __hip_bfloat16*>(k_norm_weight),
        static_cast<const float*>(cosine_fp32),
        static_cast<const float*>(sine_fp32),
        static_cast<__hip_bfloat16*>(q_output),
        static_cast<__hip_bfloat16*>(k_output),
        static_cast<__hip_bfloat16*>(v_output));
  }
  check_hip(hipGetLastError(),
            "full_attention_head_norm_rope_prefill_kernel");
}

template <std::size_t QRowStride>
__global__ void full_attention_gate_f32_prefill_kernel(
    const float* attention, const __hip_bfloat16* q_gate,
    __hip_bfloat16* attention_bf16, __hip_bfloat16* gated_bf16,
    std::size_t token_count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= token_count * kFullAttentionFeatures) return;
  const std::size_t token = index / kFullAttentionFeatures;
  const std::size_t feature = index - token * kFullAttentionFeatures;
  const std::size_t head = feature / kHeadDim;
  const std::size_t dimension = feature - head * kHeadDim;
  const std::size_t gate_index =
      token * QRowStride +
      head * kFusedHeadStride + kHeadDim + dimension;
  const __hip_bfloat16 rounded_attention =
      __float2bfloat16(attention[index]);
  const __hip_bfloat16 rounded_gate = __float2bfloat16(
      1.0f / (1.0f + expf(-__bfloat162float(q_gate[gate_index]))));
  attention_bf16[index] = rounded_attention;
  gated_bf16[index] = __float2bfloat16(
      __bfloat162float(rounded_attention) * __bfloat162float(rounded_gate));
}

template <std::size_t QRowStride>
__global__ void full_attention_gate_bf16_prefill_kernel(
    const __hip_bfloat16* attention, const __hip_bfloat16* q_gate,
    __hip_bfloat16* gated_bf16, std::size_t token_count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= token_count * kFullAttentionFeatures) return;
  const std::size_t token = index / kFullAttentionFeatures;
  const std::size_t feature = index - token * kFullAttentionFeatures;
  const std::size_t head = feature / kHeadDim;
  const std::size_t dimension = feature - head * kHeadDim;
  const std::size_t gate_index =
      token * QRowStride +
      head * kFusedHeadStride + kHeadDim + dimension;
  const __hip_bfloat16 rounded_gate = __float2bfloat16(
      1.0f / (1.0f + expf(-__bfloat162float(q_gate[gate_index]))));
  gated_bf16[index] = __float2bfloat16(
      __bfloat162float(attention[index]) * __bfloat162float(rounded_gate));
}

__global__ void extract_linear_ab_fused_kernel(
    const __hip_bfloat16* fused, __hip_bfloat16* a, __hip_bfloat16* b,
    std::size_t token_count, std::size_t row_stride) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= token_count * 64) return;
  const std::size_t token = index / 64;
  const std::size_t feature = index - token * 64;
  const __hip_bfloat16 value = fused[token * row_stride + 12288 + feature];
  if (feature < 32) {
    a[token * 32 + feature] = value;
  } else {
    b[token * 32 + feature - 32] = value;
  }
}

__global__ void linear_gated_norm_v151_kernel(
    const __hip_bfloat16* core, const __hip_bfloat16* gate_storage,
    const __hip_bfloat16* weight, __hip_bfloat16* output,
    std::size_t gate_row_stride, std::size_t gate_offset) {
  // Exact source-level preservation of the frozen v1.5.1 PyTorch fallback:
  // 32x16 workgroup, four contiguous values per lane, ascending-offset wave
  // reduction, separately rounded rsqrt, then a BF16 RMSNorm output before
  // the SiLU product.
  __shared__ volatile float row_statistic[16];
  const unsigned lane = threadIdx.x;
  const unsigned row_in_block = threadIdx.y;
  const std::size_t row = blockIdx.x * blockDim.y + row_in_block;
  const std::size_t token = row / 32;
  const std::size_t head = row - token * 32;
  const std::size_t index = row * kInvstdWidth + lane * 4;
  const float value0 = __bfloat162float(core[index]);
  const float value1 = __bfloat162float(core[index + 1]);
  const float value2 = __bfloat162float(core[index + 2]);
  const float value3 = __bfloat162float(core[index + 3]);
  float square_sum = value0 * value0;
  square_sum += value1 * value1;
  square_sum += value2 * value2;
  square_sum += value3 * value3;
  for (unsigned offset = 1; offset < 32; offset <<= 1) {
    square_sum += __shfl_down(square_sum, offset);
  }
  if (lane == 0) {
    row_statistic[row_in_block] = square_sum * (1.0f / 128.0f);
  }
  __syncthreads();
  if (lane == 0) {
    const float variance = row_statistic[row_in_block];
    row_statistic[row_in_block] =
        pytorch_rounded_rsqrtf(variance + 1.0e-6f);
  }
  __syncthreads();
  const float row_inverse_rms = row_statistic[row_in_block];
#pragma unroll
  for (unsigned element = 0; element < 4; ++element) {
    const unsigned dimension = lane * 4 + element;
    const float value = __bfloat162float(core[index + element]);
    const __hip_bfloat16 normalized = __float2bfloat16(
        value * row_inverse_rms * __bfloat162float(weight[dimension]));
    const float gate_value = __bfloat162float(
        gate_storage[token * gate_row_stride + gate_offset +
                     head * kInvstdWidth + dimension]);
    const float silu = gate_value / (1.0f + expf(-gate_value));
    output[index + element] =
        __float2bfloat16(__bfloat162float(normalized) * silu);
  }
}

__global__ void linear_gated_norm_fused_kernel(
    const __hip_bfloat16* core, const __hip_bfloat16* gate_storage,
    const __hip_bfloat16* weight, __hip_bfloat16* output,
    std::size_t gate_row_stride, std::size_t gate_offset) {
  // Match the pinned vLLM Triton RMSNormGated forward path. A single wave
  // covers four rows: each 16-lane half-wave owns one row at a time, with
  // eight contiguous BF16 values per lane and two row iterations. Triton's
  // blocked layout lowers the row reduction to a 16-lane XOR tree. Its
  // rsqrt and sigmoid lower to the native reciprocal-square-root and exp2
  // approximations, respectively; all three details can flip BF16 ties.
  const unsigned lane = threadIdx.x;
  const unsigned lane_in_row = lane & 15U;
  const unsigned row_parity = lane >> 4;
#pragma unroll
  for (unsigned row_repeat = 0; row_repeat < 2; ++row_repeat) {
    const std::size_t row =
        blockIdx.x * 4 + row_parity + row_repeat * 2;
    const std::size_t token = row / 32;
    const std::size_t head = row - token * 32;
    const std::size_t index =
        row * kInvstdWidth + lane_in_row * 8;
    float values[8];
#pragma unroll
    for (unsigned element = 0; element < 8; ++element) {
      values[element] = __bfloat162float(core[index + element]);
    }

    // The first packed pair is extracted high-half then low-half by Triton;
    // subsequent values accumulate in logical dimension order.
    float square_sum = values[1] * values[1];
    square_sum = fmaf(values[0], values[0], square_sum);
#pragma unroll
    for (unsigned element = 2; element < 8; ++element) {
      square_sum = fmaf(values[element], values[element], square_sum);
    }
#pragma unroll
    for (unsigned offset = 8; offset != 0; offset >>= 1) {
      square_sum += __shfl_xor(square_sum, offset, 16);
    }
    const float row_inverse_rms =
        rsqrtf(fmaf(square_sum, 1.0f / 128.0f, 1.0e-6f));

#pragma unroll
    for (unsigned element = 0; element < 8; ++element) {
      const unsigned dimension = lane_in_row * 8 + element;
      const float normalized =
          (values[element] * row_inverse_rms) *
          __bfloat162float(weight[dimension]);
      const float gate_value = __bfloat162float(
          gate_storage[token * gate_row_stride + gate_offset +
                       head * kInvstdWidth + dimension]);
      const float sigmoid =
          1.0f /
          (1.0f + exp2f(-gate_value * 1.4426950408889634074f));
      const float silu = gate_value * sigmoid;
      output[index + element] = __float2bfloat16(normalized * silu);
    }
  }
}

__global__ void bf16_rowwise_variance_128_pytorch_kernel(
    const __hip_bfloat16* input, float* output) {
  const unsigned lane = threadIdx.x;
  const std::size_t row = blockIdx.x * blockDim.y + threadIdx.y;
  const std::size_t index = row * kInvstdWidth + lane * 4;
  const float value0 = __bfloat162float(input[index]);
  const float value1 = __bfloat162float(input[index + 1]);
  const float value2 = __bfloat162float(input[index + 2]);
  const float value3 = __bfloat162float(input[index + 3]);
  float square_sum = value0 * value0;
  square_sum += value1 * value1;
  square_sum += value2 * value2;
  square_sum += value3 * value3;
  for (unsigned offset = 1; offset < 32; offset <<= 1) {
    square_sum += __shfl_down(square_sum, offset);
  }
  if (lane == 0) output[row] = square_sum * (1.0f / 128.0f);
}

__global__ void bf16_rowwise_invstd_128_kernel(
    const __hip_bfloat16* input, float* output, float epsilon) {
  __shared__ float squares[kInvstdWidth];
  const unsigned lane = threadIdx.x;
  const std::size_t row = blockIdx.x;
  const float value = __bfloat162float(input[row * kInvstdWidth + lane]);
  squares[lane] = value * value;
  __syncthreads();
  for (unsigned width = kInvstdWidth / 2; width != 0; width /= 2) {
    if (lane < width) squares[lane] += squares[lane + width];
    __syncthreads();
  }
  if (lane == 0) output[row] = rsqrtf(squares[0] / 128.0f + epsilon);
}

}  // namespace

void launch_prompt_embeddings(const void* embedding_bf16,
                              const std::uint32_t* host_token_ids,
                              void* device_token_ids,
                              void* output_bf16,
                              std::size_t token_count,
                              void* stream_value) {
  if (embedding_bf16 == nullptr || host_token_ids == nullptr ||
      device_token_ids == nullptr || output_bf16 == nullptr ||
      token_count == 0 || token_count > kMaxPrefillTokens) {
    throw std::invalid_argument(
        "native prompt embeddings require valid resident buffers");
  }
  for (std::size_t index = 0; index < token_count; ++index) {
    if (host_token_ids[index] >= kVocabulary) {
      throw std::invalid_argument(
          "native prompt token id is outside the model vocabulary");
    }
  }
  hipStream_t stream = static_cast<hipStream_t>(stream_value);
  check_hip(hipMemcpyAsync(device_token_ids, host_token_ids,
                           token_count * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, stream),
            "hipMemcpyAsync native prompt token ids");
  const std::size_t elements = token_count * kHidden;
  const unsigned blocks =
      static_cast<unsigned>((elements + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      prompt_embeddings_kernel, dim3(blocks), dim3(kThreads), 0, stream,
      static_cast<const __hip_bfloat16*>(embedding_bf16),
      static_cast<const std::uint32_t*>(device_token_ids),
      static_cast<__hip_bfloat16*>(output_bf16), token_count);
  check_hip(hipGetLastError(), "prompt_embeddings_kernel");
}

void launch_bf16_add(const void* left, const void* right, void* output,
                     std::size_t count, void* stream_value) {
  if (left == nullptr || right == nullptr || output == nullptr || count == 0) {
    throw std::invalid_argument("native BF16 add requires non-empty pointers");
  }
  const unsigned blocks =
      static_cast<unsigned>((count + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      bf16_add_kernel, dim3(blocks), dim3(kThreads), 0,
      static_cast<hipStream_t>(stream_value),
      static_cast<const __hip_bfloat16*>(left),
      static_cast<const __hip_bfloat16*>(right),
      static_cast<__hip_bfloat16*>(output), count);
  check_hip(hipGetLastError(), "bf16_add_kernel");
}

void launch_bf16_add_pair(const void* left, const void* right,
                          const void* residual, void* intermediate,
                          void* output, std::size_t count,
                          void* stream_value) {
  if (left == nullptr || right == nullptr || residual == nullptr ||
      intermediate == nullptr || output == nullptr || count == 0) {
    throw std::invalid_argument(
        "native BF16 add pair requires non-empty pointers");
  }
  const unsigned blocks =
      static_cast<unsigned>((count + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      bf16_add_pair_kernel, dim3(blocks), dim3(kThreads), 0,
      static_cast<hipStream_t>(stream_value),
      static_cast<const __hip_bfloat16*>(left),
      static_cast<const __hip_bfloat16*>(right),
      static_cast<const __hip_bfloat16*>(residual),
      static_cast<__hip_bfloat16*>(intermediate),
      static_cast<__hip_bfloat16*>(output), count);
  check_hip(hipGetLastError(), "bf16_add_pair_kernel");
}

void launch_prefill_rmsnorm_2048(const void* input_bf16,
                                 const void* weight_bf16,
                                 void* output_bf16,
                                 std::size_t token_count,
                                 void* stream_value) {
  if (input_bf16 == nullptr || weight_bf16 == nullptr ||
      output_bf16 == nullptr ||
      token_count == 0 || token_count > kMaxPrefillTokens) {
    throw std::invalid_argument("native prefill RMSNorm geometry is invalid");
  }
  if (token_count == 1) {
    hipLaunchKernelGGL(
        singleton_rmsnorm_2048_kernel, dim3(1), dim3(512), 0,
        static_cast<hipStream_t>(stream_value),
        static_cast<const __hip_bfloat16*>(input_bf16),
        static_cast<const __hip_bfloat16*>(weight_bf16),
        static_cast<__hip_bfloat16*>(output_bf16));
    check_hip(hipGetLastError(), "singleton_rmsnorm_2048_kernel");
  } else {
    hipLaunchKernelGGL(
        prefill_rmsnorm_2048_kernel, dim3((token_count + 15) / 16),
        dim3(32, 16), 0,
        static_cast<hipStream_t>(stream_value),
        static_cast<const __hip_bfloat16*>(input_bf16),
        static_cast<const __hip_bfloat16*>(weight_bf16),
        static_cast<__hip_bfloat16*>(output_bf16), token_count);
    check_hip(hipGetLastError(), "prefill_rmsnorm_2048_kernel");
  }
}

void launch_prefill_add_rmsnorm_2048(const void* input_bf16,
                                     const void* residual_bf16,
                                     const void* weight_bf16,
                                     void* residual_output_bf16,
                                     void* norm_output_bf16,
                                     std::size_t token_count,
                                     void* stream_value) {
  if (input_bf16 == nullptr || residual_bf16 == nullptr ||
      weight_bf16 == nullptr || residual_output_bf16 == nullptr ||
      norm_output_bf16 == nullptr ||
      token_count == 0 || token_count > kMaxPrefillTokens) {
    throw std::invalid_argument(
        "native prefill residual RMSNorm geometry is invalid");
  }
  hipLaunchKernelGGL(
      prefill_add_rmsnorm_2048_kernel, dim3((token_count + 15) / 16),
      dim3(32, 16), 0,
      static_cast<hipStream_t>(stream_value),
      static_cast<const __hip_bfloat16*>(input_bf16),
      static_cast<const __hip_bfloat16*>(residual_bf16),
      static_cast<const __hip_bfloat16*>(weight_bf16),
      static_cast<__hip_bfloat16*>(residual_output_bf16),
      static_cast<__hip_bfloat16*>(norm_output_bf16), token_count);
  check_hip(hipGetLastError(), "prefill_add_rmsnorm_2048_kernel");
}

void launch_shared_silu_multiply(const void* fused_shared_input,
                                 void* activated_512, void* stream_value) {
  if (fused_shared_input == nullptr || activated_512 == nullptr) {
    throw std::invalid_argument("native shared activation requires non-null pointers");
  }
  const unsigned blocks = static_cast<unsigned>(
      (kSharedIntermediate + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      shared_activation_kernel, dim3(blocks), dim3(kThreads), 0,
      static_cast<hipStream_t>(stream_value),
      static_cast<const __hip_bfloat16*>(fused_shared_input),
      static_cast<__hip_bfloat16*>(activated_512));
  check_hip(hipGetLastError(), "shared_activation_kernel");
}

void launch_shared_sigmoid_scale(const void* fused_shared_input,
                                 const void* shared_down_2048,
                                 void* output_2048, void* stream_value) {
  if (fused_shared_input == nullptr || shared_down_2048 == nullptr ||
      output_2048 == nullptr) {
    throw std::invalid_argument("native shared gate requires non-null pointers");
  }
  const unsigned blocks =
      static_cast<unsigned>((kHidden + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      shared_gate_kernel, dim3(blocks), dim3(kThreads), 0,
      static_cast<hipStream_t>(stream_value),
      static_cast<const __hip_bfloat16*>(fused_shared_input),
      static_cast<const __hip_bfloat16*>(shared_down_2048),
      static_cast<__hip_bfloat16*>(output_2048));
  check_hip(hipGetLastError(), "shared_gate_kernel");
}

void launch_full_attention_sigmoid_gate(const void* attention_4096,
                                        const void* fused_q_gate_storage,
                                        void* gated_4096,
                                        void* stream_value) {
  if (attention_4096 == nullptr || fused_q_gate_storage == nullptr ||
      gated_4096 == nullptr) {
    throw std::invalid_argument("native full-attention gate requires non-null pointers");
  }
  const unsigned blocks = static_cast<unsigned>(
      (kFullAttentionFeatures + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      full_attention_gate_kernel, dim3(blocks), dim3(kThreads), 0,
      static_cast<hipStream_t>(stream_value),
      static_cast<const __hip_bfloat16*>(attention_4096),
      static_cast<const __hip_bfloat16*>(fused_q_gate_storage),
      static_cast<__hip_bfloat16*>(gated_4096));
  check_hip(hipGetLastError(), "full_attention_gate_kernel");
}

void launch_q8192_rotary_table(void* cosine_fp32, void* sine_fp32,
                               void* stream_value) {
  launch_prefill_rotary_table(cosine_fp32, sine_fp32, 8192, 0,
                              stream_value);
}

void launch_prefill_rotary_table(void* cosine_fp32, void* sine_fp32,
                                 std::size_t token_count,
                                 std::size_t position_start,
                                 void* stream_value) {
  if (cosine_fp32 == nullptr || sine_fp32 == nullptr) {
    throw std::invalid_argument(
        "native q8192 rotary table requires non-null outputs");
  }
  if (token_count == 0 || token_count > kMaxPrefillTokens) {
    throw std::invalid_argument("native rotary table context is unsupported");
  }
  const std::size_t elements = token_count * kRotaryPairs;
  const unsigned blocks =
      static_cast<unsigned>((elements + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      prefill_rotary_table_kernel, dim3(blocks), dim3(kThreads), 0,
      static_cast<hipStream_t>(stream_value),
      static_cast<float*>(cosine_fp32), static_cast<float*>(sine_fp32),
      token_count, position_start);
  check_hip(hipGetLastError(), "prefill_rotary_table_kernel");
}

void launch_prefill_mrope_rotary_table(
    void* cosine_fp32, void* sine_fp32, const void* positions_i64,
    std::size_t token_count, std::size_t position_row_stride,
    void* stream_value) {
  if (cosine_fp32 == nullptr || sine_fp32 == nullptr ||
      positions_i64 == nullptr) {
    throw std::invalid_argument(
        "native M-RoPE table requires non-null inputs and outputs");
  }
  if (token_count == 0 || token_count > kMaxPrefillTokens) {
    throw std::invalid_argument(
        "native M-RoPE table context is unsupported");
  }
  if (position_row_stride == 0) position_row_stride = token_count;
  if (position_row_stride < token_count) {
    throw std::invalid_argument(
        "native M-RoPE position row stride is invalid");
  }
  const std::size_t elements = token_count * kRotaryPairs;
  const unsigned blocks =
      static_cast<unsigned>((elements + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      prefill_mrope_rotary_table_kernel, dim3(blocks), dim3(kThreads), 0,
      static_cast<hipStream_t>(stream_value),
      static_cast<float*>(cosine_fp32), static_cast<float*>(sine_fp32),
      static_cast<const std::int64_t*>(positions_i64), token_count,
      position_row_stride);
  check_hip(hipGetLastError(), "prefill_mrope_rotary_table_kernel");
}

void launch_q8192_full_attention_head_norm_rope(
    const void* q_gate, const void* k_raw,
    const void* q_norm_weight, const void* k_norm_weight,
    const void* cosine_fp32, const void* sine_fp32,
    void* q_output, void* k_output, void* stream_value) {
  launch_full_attention_head_norm_rope_prefill(
      q_gate, k_raw, nullptr, q_norm_weight, k_norm_weight, cosine_fp32,
      sine_fp32, q_output, k_output, nullptr, 8192, 8192, 512, 0,
      stream_value);
}

void launch_full_attention_head_norm_rope_prefill(
    const void* q_gate, const void* k_raw, const void* v_raw,
    const void* q_norm_weight, const void* k_norm_weight,
    const void* cosine_fp32, const void* sine_fp32,
    void* q_output, void* k_output, void* v_output,
    std::size_t token_count, std::size_t q_row_stride,
    std::size_t k_row_stride, std::size_t v_row_stride,
    void* stream_value) {
  launch_full_attention_head_norm_rope_prefill_impl<false>(
      q_gate, k_raw, v_raw, q_norm_weight, k_norm_weight, cosine_fp32,
      sine_fp32, q_output, k_output, v_output, token_count, q_row_stride,
      k_row_stride, v_row_stride, stream_value);
}

void launch_full_attention_head_norm_mrope_prefill(
    const void* q_gate, const void* k_raw, const void* v_raw,
    const void* q_norm_weight, const void* k_norm_weight,
    const void* cosine_fp32, const void* sine_fp32,
    void* q_output, void* k_output, void* v_output,
    std::size_t token_count, std::size_t q_row_stride,
    std::size_t k_row_stride, std::size_t v_row_stride,
    void* stream_value) {
  launch_full_attention_head_norm_rope_prefill_impl<true>(
      q_gate, k_raw, v_raw, q_norm_weight, k_norm_weight, cosine_fp32,
      sine_fp32, q_output, k_output, v_output, token_count, q_row_stride,
      k_row_stride, v_row_stride, stream_value);
}

void launch_q8192_full_attention_sigmoid_gate_f32(
    const void* attention_f32, const void* fused_q_gate_storage,
    void* attention_bf16, void* gated_bf16, void* stream_value) {
  launch_full_attention_sigmoid_gate_f32_prefill(
      attention_f32, fused_q_gate_storage, attention_bf16, gated_bf16,
      8192, 8192, stream_value);
}

void launch_full_attention_sigmoid_gate_f32_prefill(
    const void* attention_f32, const void* fused_q_gate_storage,
    void* attention_bf16, void* gated_bf16, std::size_t token_count,
    std::size_t q_row_stride, void* stream_value) {
  if (attention_f32 == nullptr || fused_q_gate_storage == nullptr ||
      attention_bf16 == nullptr || gated_bf16 == nullptr) {
    throw std::invalid_argument(
        "native q8192 full-attention gate requires non-null pointers");
  }
  if (token_count == 0 || token_count > kMaxPrefillTokens ||
      (q_row_stride != 8192 && q_row_stride != 9216)) {
    throw std::invalid_argument("native full-attention gate geometry is unsupported");
  }
  const std::size_t elements = token_count * kFullAttentionFeatures;
  const unsigned blocks =
      static_cast<unsigned>((elements + kThreads - 1) / kThreads);
  if (q_row_stride == 8192) {
    hipLaunchKernelGGL(
        (full_attention_gate_f32_prefill_kernel<8192>),
        dim3(blocks), dim3(kThreads), 0,
        static_cast<hipStream_t>(stream_value),
        static_cast<const float*>(attention_f32),
        static_cast<const __hip_bfloat16*>(fused_q_gate_storage),
        static_cast<__hip_bfloat16*>(attention_bf16),
        static_cast<__hip_bfloat16*>(gated_bf16), token_count);
  } else {
    hipLaunchKernelGGL(
        (full_attention_gate_f32_prefill_kernel<9216>),
        dim3(blocks), dim3(kThreads), 0,
        static_cast<hipStream_t>(stream_value),
        static_cast<const float*>(attention_f32),
        static_cast<const __hip_bfloat16*>(fused_q_gate_storage),
        static_cast<__hip_bfloat16*>(attention_bf16),
        static_cast<__hip_bfloat16*>(gated_bf16), token_count);
  }
  check_hip(hipGetLastError(),
            "full_attention_gate_f32_prefill_kernel");
}

void launch_full_attention_sigmoid_gate_bf16_prefill(
    const void* attention_bf16, const void* fused_q_gate_storage,
    void* gated_bf16, std::size_t token_count,
    std::size_t q_row_stride, void* stream_value) {
  if (attention_bf16 == nullptr || fused_q_gate_storage == nullptr ||
      gated_bf16 == nullptr) {
    throw std::invalid_argument(
        "native BF16 full-attention gate requires non-null pointers");
  }
  if (token_count == 0 || token_count > kMaxPrefillTokens ||
      (q_row_stride != 8192 && q_row_stride != 9216)) {
    throw std::invalid_argument(
        "native BF16 full-attention gate geometry is unsupported");
  }
  const std::size_t elements = token_count * kFullAttentionFeatures;
  const unsigned blocks =
      static_cast<unsigned>((elements + kThreads - 1) / kThreads);
  if (q_row_stride == 8192) {
    hipLaunchKernelGGL(
        (full_attention_gate_bf16_prefill_kernel<8192>),
        dim3(blocks), dim3(kThreads), 0,
        static_cast<hipStream_t>(stream_value),
        static_cast<const __hip_bfloat16*>(attention_bf16),
        static_cast<const __hip_bfloat16*>(fused_q_gate_storage),
        static_cast<__hip_bfloat16*>(gated_bf16), token_count);
  } else {
    hipLaunchKernelGGL(
        (full_attention_gate_bf16_prefill_kernel<9216>),
        dim3(blocks), dim3(kThreads), 0,
        static_cast<hipStream_t>(stream_value),
        static_cast<const __hip_bfloat16*>(attention_bf16),
        static_cast<const __hip_bfloat16*>(fused_q_gate_storage),
        static_cast<__hip_bfloat16*>(gated_bf16), token_count);
  }
  check_hip(hipGetLastError(),
            "full_attention_gate_bf16_prefill_kernel");
}

void launch_extract_linear_ab_fused(const void* fused_input_bf16,
                                    void* a_bf16, void* b_bf16,
                                    std::size_t token_count,
                                    std::size_t fused_row_stride,
                                    void* stream_value) {
  if (fused_input_bf16 == nullptr || a_bf16 == nullptr || b_bf16 == nullptr ||
      token_count == 0 || token_count > kMaxPrefillTokens ||
      fused_row_stride < 12352) {
    throw std::invalid_argument("native fused linear A/B geometry is invalid");
  }
  const std::size_t elements = token_count * 64;
  const unsigned blocks =
      static_cast<unsigned>((elements + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      extract_linear_ab_fused_kernel, dim3(blocks), dim3(kThreads), 0,
      static_cast<hipStream_t>(stream_value),
      static_cast<const __hip_bfloat16*>(fused_input_bf16),
      static_cast<__hip_bfloat16*>(a_bf16),
      static_cast<__hip_bfloat16*>(b_bf16), token_count, fused_row_stride);
  check_hip(hipGetLastError(), "extract_linear_ab_fused_kernel");
}

void launch_linear_gated_norm_fused(
    const void* core_bf16, const void* fused_input_bf16,
    const void* norm_weight_bf16, void* output_bf16,
    std::size_t token_count, std::size_t fused_row_stride,
    void* stream_value) {
  if (core_bf16 == nullptr || fused_input_bf16 == nullptr ||
      norm_weight_bf16 == nullptr || output_bf16 == nullptr ||
      token_count == 0 || token_count > kMaxPrefillTokens ||
      fused_row_stride < 12352) {
    throw std::invalid_argument("native fused linear gated norm geometry is invalid");
  }
  hipLaunchKernelGGL(
      linear_gated_norm_fused_kernel, dim3(token_count * 8),
      dim3(32), 0, static_cast<hipStream_t>(stream_value),
      static_cast<const __hip_bfloat16*>(core_bf16),
      static_cast<const __hip_bfloat16*>(fused_input_bf16),
      static_cast<const __hip_bfloat16*>(norm_weight_bf16),
      static_cast<__hip_bfloat16*>(output_bf16), fused_row_stride, 8192);
  check_hip(hipGetLastError(), "linear_gated_norm_fused_kernel");
}

void launch_linear_gated_norm_separate(
    const void* core_bf16, const void* gate_bf16,
    const void* norm_weight_bf16, void* output_bf16,
    std::size_t token_count, void* stream_value) {
  if (core_bf16 == nullptr || gate_bf16 == nullptr ||
      norm_weight_bf16 == nullptr || output_bf16 == nullptr ||
      token_count == 0 || token_count > kMaxPrefillTokens) {
    throw std::invalid_argument(
        "native separate linear gated norm geometry is invalid");
  }
  hipLaunchKernelGGL(
      linear_gated_norm_fused_kernel, dim3(token_count * 8),
      dim3(32), 0, static_cast<hipStream_t>(stream_value),
      static_cast<const __hip_bfloat16*>(core_bf16),
      static_cast<const __hip_bfloat16*>(gate_bf16),
      static_cast<const __hip_bfloat16*>(norm_weight_bf16),
      static_cast<__hip_bfloat16*>(output_bf16), 4096, 0);
  check_hip(hipGetLastError(), "linear_gated_norm_separate_kernel");
}

void launch_linear_gated_norm_fused_v151(
    const void* core_bf16, const void* fused_input_bf16,
    const void* norm_weight_bf16, void* output_bf16,
    std::size_t token_count, std::size_t fused_row_stride,
    void* stream_value) {
  if (core_bf16 == nullptr || fused_input_bf16 == nullptr ||
      norm_weight_bf16 == nullptr || output_bf16 == nullptr ||
      token_count == 0 || token_count > kMaxPrefillTokens ||
      fused_row_stride < 12352) {
    throw std::invalid_argument(
        "v1.5.1 fused linear gated norm geometry is invalid");
  }
  hipLaunchKernelGGL(
      linear_gated_norm_v151_kernel, dim3(token_count * 2), dim3(32, 16), 0,
      static_cast<hipStream_t>(stream_value),
      static_cast<const __hip_bfloat16*>(core_bf16),
      static_cast<const __hip_bfloat16*>(fused_input_bf16),
      static_cast<const __hip_bfloat16*>(norm_weight_bf16),
      static_cast<__hip_bfloat16*>(output_bf16), fused_row_stride, 8192);
  check_hip(hipGetLastError(), "linear_gated_norm_v151_fused_kernel");
}

void launch_linear_gated_norm_separate_v151(
    const void* core_bf16, const void* gate_bf16,
    const void* norm_weight_bf16, void* output_bf16,
    std::size_t token_count, void* stream_value) {
  if (core_bf16 == nullptr || gate_bf16 == nullptr ||
      norm_weight_bf16 == nullptr || output_bf16 == nullptr ||
      token_count == 0 || token_count > kMaxPrefillTokens) {
    throw std::invalid_argument(
        "v1.5.1 separate linear gated norm geometry is invalid");
  }
  hipLaunchKernelGGL(
      linear_gated_norm_v151_kernel, dim3(token_count * 2), dim3(32, 16), 0,
      static_cast<hipStream_t>(stream_value),
      static_cast<const __hip_bfloat16*>(core_bf16),
      static_cast<const __hip_bfloat16*>(gate_bf16),
      static_cast<const __hip_bfloat16*>(norm_weight_bf16),
      static_cast<__hip_bfloat16*>(output_bf16), 4096, 0);
  check_hip(hipGetLastError(), "linear_gated_norm_v151_separate_kernel");
}

void launch_bf16_rowwise_variance_128_pytorch(
    const void* rows_bf16, void* variance_fp32,
    std::size_t row_count, void* stream_value) {
  if (rows_bf16 == nullptr || variance_fp32 == nullptr || row_count == 0 ||
      row_count % 16 != 0) {
    throw std::invalid_argument(
        "native PyTorch-order variance requires aligned width-128 rows");
  }
  hipLaunchKernelGGL(
      bf16_rowwise_variance_128_pytorch_kernel, dim3(row_count / 16),
      dim3(32, 16), 0, static_cast<hipStream_t>(stream_value),
      static_cast<const __hip_bfloat16*>(rows_bf16),
      static_cast<float*>(variance_fp32));
  check_hip(hipGetLastError(),
            "bf16_rowwise_variance_128_pytorch_kernel");
}

void launch_bf16_rowwise_invstd_128(const void* rows_bf16,
                                    void* invstd_fp32,
                                    std::size_t row_count, float epsilon,
                                    void* stream_value) {
  if (rows_bf16 == nullptr || invstd_fp32 == nullptr || row_count == 0 ||
      !(epsilon > 0.0f)) {
    throw std::invalid_argument(
        "native rowwise invstd requires valid pointers and geometry");
  }
  hipLaunchKernelGGL(
      bf16_rowwise_invstd_128_kernel, dim3(row_count), dim3(kInvstdWidth), 0,
      static_cast<hipStream_t>(stream_value),
      static_cast<const __hip_bfloat16*>(rows_bf16),
      static_cast<float*>(invstd_fp32), epsilon);
  check_hip(hipGetLastError(), "bf16_rowwise_invstd_128_kernel");
}

}  // namespace aima
