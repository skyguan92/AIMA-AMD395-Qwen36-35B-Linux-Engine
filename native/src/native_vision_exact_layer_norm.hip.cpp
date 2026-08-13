// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_exact_layer_norm.h"

#include "aima/native_vl_processor.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace aima {
namespace {

// Algorithm and launch shape are derived from PyTorch
// aten/src/ATen/native/cuda/layer_norm_kernel.cu at commit
// 8514f05131610dab50233027b2fab9c01235081b (BSD-3-Clause).
constexpr int kVisionHidden = 1152;
constexpr int kVectorSize = 4;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr float kLayerNormEpsilon = 1.0e-6f;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T values[Size];
};

struct WelfordData {
  float mean;
  float sigma2;
  float count;
};

template <bool FastReciprocal>
__device__ float reciprocal(float value) {
  if constexpr (FastReciprocal) {
    return __builtin_amdgcn_rcpf(value);
  }
  return 1.0f / value;
}

template <bool FastReciprocal>
__device__ WelfordData welford_online(float value,
                                      const WelfordData& current) {
  const float delta = value - current.mean;
  const float new_count = current.count + 1.0f;
  const float new_mean =
      current.mean + delta * reciprocal<FastReciprocal>(new_count);
  return WelfordData{new_mean,
                     current.sigma2 + delta * (value - new_mean),
                     new_count};
}

template <bool FastReciprocal>
__device__ WelfordData welford_combine(const WelfordData& data_b,
                                       const WelfordData& data_a) {
  const float delta = data_b.mean - data_a.mean;
  const float count = data_a.count + data_b.count;
  if (count <= 0.0f) return WelfordData{0.0f, 0.0f, 0.0f};
  const float coefficient = reciprocal<FastReciprocal>(count);
  const float n_a = data_a.count * coefficient;
  const float n_b = data_b.count * coefficient;
  return WelfordData{
      n_a * data_a.mean + n_b * data_b.mean,
      data_a.sigma2 + data_b.sigma2 +
          delta * delta * data_a.count * n_b,
      count};
}

template <bool FastReciprocal>
__device__ WelfordData compute_stats(const __hip_bfloat16* input,
                                     float* shared) {
  using Vector = AlignedVector<__hip_bfloat16, kVectorSize>;
  const auto* input_vectors = reinterpret_cast<const Vector*>(input);
  const int linear_thread =
      static_cast<int>(threadIdx.x + threadIdx.y * blockDim.x);
  constexpr int kVectorCount = kVisionHidden / kVectorSize;
  constexpr int kThreads = kWarpSize * kWarpsPerBlock;
  WelfordData value{0.0f, 0.0f, 0.0f};
  for (int vector_index = linear_thread; vector_index < kVectorCount;
       vector_index += kThreads) {
    const Vector vector = input_vectors[vector_index];
#pragma unroll
    for (int element = 0; element < kVectorSize; ++element) {
      value = welford_online<FastReciprocal>(
          __bfloat162float(vector.values[element]), value);
    }
  }
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    const WelfordData other{
        __shfl_down(value.mean, offset, kWarpSize),
        __shfl_down(value.sigma2, offset, kWarpSize),
        __shfl_down(value.count, offset, kWarpSize)};
    value = welford_combine<FastReciprocal>(value, other);
  }

  float* mean_sigma = shared;
  float* counts = shared + kWarpsPerBlock;
  for (int offset = kWarpsPerBlock / 2; offset > 0; offset /= 2) {
    if (threadIdx.x == 0 && threadIdx.y >= offset &&
        threadIdx.y < 2 * offset) {
      const int write_warp = static_cast<int>(threadIdx.y) - offset;
      mean_sigma[2 * write_warp] = value.mean;
      mean_sigma[2 * write_warp + 1] = value.sigma2;
      counts[write_warp] = value.count;
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y < offset) {
      const WelfordData other{
          mean_sigma[2 * threadIdx.y], mean_sigma[2 * threadIdx.y + 1],
          counts[threadIdx.y]};
      value = welford_combine<FastReciprocal>(value, other);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    mean_sigma[0] = value.mean;
    mean_sigma[1] = value.sigma2 / static_cast<float>(kVisionHidden);
  }
  __syncthreads();
  return WelfordData{mean_sigma[0], mean_sigma[1], 0.0f};
}

template <bool FastReciprocal>
__global__ void exact_vision_layer_norm_kernel(
    const __hip_bfloat16* input, const __hip_bfloat16* weight,
    const __hip_bfloat16* bias, __hip_bfloat16* output) {
  extern __shared__ float shared[];
  const std::size_t row = blockIdx.x;
  const auto* row_input = input + row * kVisionHidden;
  auto* row_output = output + row * kVisionHidden;
  const WelfordData stats =
      compute_stats<FastReciprocal>(row_input, shared);
  const float inverse_standard_deviation =
      rsqrtf(stats.sigma2 + kLayerNormEpsilon);

  using Vector = AlignedVector<__hip_bfloat16, kVectorSize>;
  const auto* input_vectors = reinterpret_cast<const Vector*>(row_input);
  const auto* weight_vectors = reinterpret_cast<const Vector*>(weight);
  const auto* bias_vectors = reinterpret_cast<const Vector*>(bias);
  auto* output_vectors = reinterpret_cast<Vector*>(row_output);
  const int linear_thread =
      static_cast<int>(threadIdx.x + threadIdx.y * blockDim.x);
  constexpr int kVectorCount = kVisionHidden / kVectorSize;
  constexpr int kThreads = kWarpSize * kWarpsPerBlock;
  for (int vector_index = linear_thread; vector_index < kVectorCount;
       vector_index += kThreads) {
    const Vector input_vector = input_vectors[vector_index];
    const Vector weight_vector = weight_vectors[vector_index];
    const Vector bias_vector = bias_vectors[vector_index];
    Vector output_vector;
#pragma unroll
    for (int element = 0; element < kVectorSize; ++element) {
      const float result =
          __bfloat162float(weight_vector.values[element]) *
              (inverse_standard_deviation *
               (__bfloat162float(input_vector.values[element]) - stats.mean)) +
          __bfloat162float(bias_vector.values[element]);
      output_vector.values[element] = __float2bfloat16(result);
    }
    output_vectors[vector_index] = output_vector;
  }
}

}  // namespace

struct NativeVisionExactLayerNormPlan::Impl {
  Impl(std::size_t rows, NativeVisionLayerNormReciprocal requested_mode)
      : row_count_value(rows), reciprocal_mode_value(requested_mode) {
    if (rows == 0 || rows > 4 * kNativeVlAggregateTokenLimit) {
      throw std::invalid_argument(
          "native exact vision LayerNorm row count is outside the budget");
    }
  }

  std::size_t row_count_value = 0;
  NativeVisionLayerNormReciprocal reciprocal_mode_value =
      NativeVisionLayerNormReciprocal::kDivision;
};

NativeVisionExactLayerNormPlan::NativeVisionExactLayerNormPlan(
    std::size_t row_count,
    NativeVisionLayerNormReciprocal reciprocal_mode)
    : impl_(std::make_unique<Impl>(row_count, reciprocal_mode)) {}
NativeVisionExactLayerNormPlan::~NativeVisionExactLayerNormPlan() = default;
NativeVisionExactLayerNormPlan::NativeVisionExactLayerNormPlan(
    NativeVisionExactLayerNormPlan&&) noexcept = default;
NativeVisionExactLayerNormPlan& NativeVisionExactLayerNormPlan::operator=(
    NativeVisionExactLayerNormPlan&&) noexcept = default;

void NativeVisionExactLayerNormPlan::launch(
    const void* input_device, const void* weight_device,
    const void* bias_device, void* output_device, void* stream_pointer) const {
  if (!impl_ || input_device == nullptr || weight_device == nullptr ||
      bias_device == nullptr || output_device == nullptr ||
      input_device == output_device || weight_device == output_device ||
      bias_device == output_device) {
    throw std::invalid_argument(
        "native exact vision LayerNorm launch is invalid");
  }
  const dim3 threads(kWarpSize, kWarpsPerBlock, 1);
  constexpr std::size_t kSharedBytes =
      kWarpsPerBlock * 3 / 2 * sizeof(float);
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
  if (impl_->reciprocal_mode_value ==
      NativeVisionLayerNormReciprocal::kFastAmdReciprocal) {
    hipLaunchKernelGGL(
        (exact_vision_layer_norm_kernel<true>),
        dim3(impl_->row_count_value), threads, kSharedBytes, stream,
        static_cast<const __hip_bfloat16*>(input_device),
        static_cast<const __hip_bfloat16*>(weight_device),
        static_cast<const __hip_bfloat16*>(bias_device),
        static_cast<__hip_bfloat16*>(output_device));
  } else {
    hipLaunchKernelGGL(
        (exact_vision_layer_norm_kernel<false>),
        dim3(impl_->row_count_value), threads, kSharedBytes, stream,
        static_cast<const __hip_bfloat16*>(input_device),
        static_cast<const __hip_bfloat16*>(weight_device),
        static_cast<const __hip_bfloat16*>(bias_device),
        static_cast<__hip_bfloat16*>(output_device));
  }
  check_hip(hipGetLastError(), "native exact vision LayerNorm launch");
}

std::size_t NativeVisionExactLayerNormPlan::row_count() const {
  return impl_->row_count_value;
}

NativeVisionLayerNormReciprocal
NativeVisionExactLayerNormPlan::reciprocal_mode() const {
  return impl_->reciprocal_mode_value;
}

}  // namespace aima
