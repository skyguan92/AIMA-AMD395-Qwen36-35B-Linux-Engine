// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors
//
// The projection kernel is adapted from vLLM csrc/rocm/skinny_gemms.cu at
// commit 29e5d102050669d03992a2eb863ad364ea50fab2 (Apache-2.0). Changes:
// remove all Torch/c10 dispatch and allocation, retain only the gfx1151 BF16
// N=1 small-LDS specialization, and expose a resident-pointer native API.

#include "aima/bf16_wvsplitk.h"

#include "aima/sha256.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aima {
namespace {

constexpr int kThreads = 32;
constexpr int kWavesPerGroup = 16;
constexpr int kYTile = 2;
constexpr int kActivationChunk = 8;
constexpr int kUnroll = 2;
constexpr int kLdsBf16Elements = 32 * 1024;
constexpr int kLaunchesPerSample = 200;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

template <typename T>
__device__ __forceinline__ T load_nontemporal(T* address) {
  return __builtin_nontemporal_load(address);
}

__device__ __forceinline__ unsigned min_u32(std::uint32_t left,
                                             std::uint32_t right) {
  return left < right ? left : right;
}

using Scalar8 =
    __attribute__((__vector_size__((kActivationChunk / 2) * sizeof(float))))
    float;

union Vector8 {
  __hip_bfloat16 h[kActivationChunk];
  float f[kActivationChunk / 2];
  float2 f2[kActivationChunk / 4];
  double d[kActivationChunk / 4];
  Scalar8 h8;
};

__device__ __forceinline__ void dot2_accumulate(float& accumulator,
                                                 float packed_left,
                                                 float packed_right) {
  const float2 left =
      __bfloat1622float2(*reinterpret_cast<__hip_bfloat162*>(&packed_left));
  const float2 right =
      __bfloat1622float2(*reinterpret_cast<__hip_bfloat162*>(&packed_right));
  const float2 product = left * right;
  accumulator += product.x + product.y;
}

__global__ void __launch_bounds__(kWavesPerGroup * kThreads)
    bf16_wvsplitk_n1_small_kernel(
        int k, int weight_stride, int activation_stride, int m,
        const __hip_bfloat16* weight, const __hip_bfloat16* activation,
        const __hip_bfloat16* bias, __hip_bfloat16* output,
        int active_waves_per_group, int cu_count) {
  __shared__ __hip_bfloat16 shared_activation[kLdsBf16Elements];

  for (std::uint32_t offset =
           (threadIdx.y * kThreads + threadIdx.x) * kActivationChunk;
       offset < min_u32(static_cast<std::uint32_t>(activation_stride),
                        kLdsBf16Elements);
       offset += kThreads * kWavesPerGroup * kActivationChunk) {
    *reinterpret_cast<Vector8*>(&shared_activation[offset]) =
        *reinterpret_cast<const Vector8*>(&activation[offset]);
  }
  __syncthreads();

  if (threadIdx.y >= active_waves_per_group) return;
  std::uint32_t row =
      (blockIdx.x * active_waves_per_group + threadIdx.y) * kYTile;

  while (row < static_cast<std::uint32_t>(m)) {
    float sum[kYTile] = {};

    for (std::uint32_t k1 = 0; k1 < static_cast<std::uint32_t>(k);
         k1 += kThreads * kActivationChunk * kUnroll) {
      Vector8 activation_chunk[kUnroll] = {};
      Vector8 weight_chunk[kYTile][kUnroll];

#pragma unroll
      for (std::uint32_t k2 = 0; k2 < kUnroll; ++k2) {
        const std::uint32_t base = k1 + k2 * kThreads * kActivationChunk;
        const std::uint32_t lane_offset =
            base + threadIdx.x * kActivationChunk;
        const __hip_bfloat16* weight_lane =
            &weight[min_u32(lane_offset,
                            static_cast<std::uint32_t>(k - kActivationChunk))];
#pragma unroll
        for (int y = 0; y < kYTile; ++y) {
          const std::uint32_t bounded_row =
              min_u32(row + y, static_cast<std::uint32_t>(m - 1));
          weight_chunk[y][k2].h8 = load_nontemporal(
              reinterpret_cast<Scalar8*>(const_cast<__hip_bfloat16*>(
                  &weight_lane[bounded_row * weight_stride])));
        }
      }

#pragma unroll
      for (std::uint32_t k2 = 0; k2 < kUnroll; ++k2) {
        const std::uint32_t base = k1 + k2 * kThreads * kActivationChunk;
        const std::uint32_t lane_offset =
            base + threadIdx.x * kActivationChunk;
        if (lane_offset >= static_cast<std::uint32_t>(k)) break;
        activation_chunk[k2] =
            *reinterpret_cast<const Vector8*>(&shared_activation[lane_offset]);
      }

#pragma unroll
      for (std::uint32_t k2 = 0; k2 < kUnroll; ++k2) {
#pragma unroll
        for (int y = 0; y < kYTile; ++y) {
#pragma unroll
          for (std::uint32_t pair = 0; pair < kActivationChunk / 2; ++pair) {
            dot2_accumulate(sum[y], activation_chunk[k2].f[pair],
                            weight_chunk[y][k2].f[pair]);
          }
        }
      }
    }

    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int y = 0; y < kYTile; ++y) {
      sum[y] += __builtin_amdgcn_mov_dpp(sum[y], 0x118, 0xf, 0xf, 1);
      sum[y] += __builtin_amdgcn_mov_dpp(sum[y], 0x114, 0xf, 0xf, 1);
      sum[y] += __builtin_amdgcn_mov_dpp(sum[y], 0x112, 0xf, 0xf, 1);
      sum[y] += __builtin_amdgcn_mov_dpp(sum[y], 0x111, 0xf, 0xf, 1);
      sum[y] += __shfl_xor(sum[y], 16);
    }

    if (threadIdx.x == kThreads - 1) {
#pragma unroll
      for (int y = 0; y < kYTile; ++y) {
        if (bias != nullptr) sum[y] += __bfloat162float(bias[row + y]);
        output[row + y] = __float2bfloat16(sum[y]);
      }
    }
    row += cu_count * active_waves_per_group * kYTile;
  }
}

int minimum_divisor(int rows, int first_divisor, int second_divisor) {
  const int rows_per_round = first_divisor * second_divisor;
  std::array<int, 13> rounds{};
  int candidate = rows_per_round;
  for (int index = 0; index < static_cast<int>(rounds.size()); ++index) {
    rounds[index] = (rows + candidate - 1) / candidate;
    candidate -= first_divisor;
  }
  for (int index = static_cast<int>(rounds.size()) - 1; index >= 0; --index) {
    if (rounds[0] == rounds[index]) return second_divisor - index;
  }
  return 0;
}

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) {
    check_hip(hipMalloc(&pointer_, bytes), "hipMalloc wvSplitK probe");
  }
  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      const hipError_t ignored = hipFree(pointer_);
      static_cast<void>(ignored);
    }
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  void* get() const { return pointer_; }

 private:
  void* pointer_ = nullptr;
};

class Event {
 public:
  Event() { check_hip(hipEventCreate(&value_), "hipEventCreate wvSplitK"); }
  ~Event() {
    if (value_ != nullptr) {
      const hipError_t ignored = hipEventDestroy(value_);
      static_cast<void>(ignored);
    }
  }
  operator hipEvent_t() const { return value_; }

 private:
  hipEvent_t value_ = nullptr;
};

__global__ void fill_pattern_kernel(__hip_bfloat16* values, std::size_t count,
                                    std::uint32_t multiplier,
                                    std::uint32_t increment,
                                    std::uint32_t modulus, int center,
                                    float denominator) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int integer = static_cast<int>(
                          (index * multiplier + increment) % modulus) -
                      center;
  values[index] = __float2bfloat16(static_cast<float>(integer) / denominator);
}

void fill_pattern(void* pointer, std::size_t count, std::uint32_t multiplier,
                  std::uint32_t increment, std::uint32_t modulus, int center,
                  float denominator) {
  constexpr unsigned threads = 256;
  const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
  hipLaunchKernelGGL(fill_pattern_kernel, dim3(blocks), dim3(threads), 0,
                     nullptr, static_cast<__hip_bfloat16*>(pointer), count,
                     multiplier, increment, modulus, center, denominator);
  check_hip(hipGetLastError(), "fill_pattern_kernel");
}

float pattern_value(std::size_t index, std::uint32_t multiplier,
                    std::uint32_t increment, std::uint32_t modulus,
                    int center, float denominator) {
  const int integer = static_cast<int>(
                          (index * multiplier + increment) % modulus) -
                      center;
  return static_cast<float>(integer) / denominator;
}

float bf16_bits_to_float(std::uint16_t bits) {
  std::uint32_t fp32_bits = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0f;
  std::memcpy(&value, &fp32_bits, sizeof(value));
  return value;
}

Bf16WvSplitKCaseResult probe_case(std::size_t m, std::size_t k,
                                           int cu_count) {
  Bf16WvSplitKCaseResult result;
  result.m = m;
  result.k = k;
  result.cu_count = cu_count;
  result.active_waves_per_group = minimum_divisor(
      static_cast<int>(m), cu_count * kYTile, kWavesPerGroup);
  result.launches_per_sample = kLaunchesPerSample;
  result.expected_elements = m;

  DeviceAllocation weight(m * k * sizeof(__hip_bfloat16));
  DeviceAllocation activation(k * sizeof(__hip_bfloat16));
  DeviceAllocation output(m * sizeof(__hip_bfloat16));
  fill_pattern(weight.get(), m * k, 13, 7, 17, 8, 64.0f);
  fill_pattern(activation.get(), k, 11, 3, 19, 9, 32.0f);

  for (int index = 0; index < 10; ++index) {
    launch_bf16_wvsplitk(weight.get(), activation.get(), nullptr, output.get(),
                         m, k, cu_count);
  }
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize wvSplitK warmup");

  Event start;
  Event stop;
  for (int index = 0; index < 5; ++index) {
    check_hip(hipEventRecord(start), "hipEventRecord wvSplitK start");
    for (int launch = 0; launch < kLaunchesPerSample; ++launch) {
      launch_bf16_wvsplitk(weight.get(), activation.get(), nullptr,
                           output.get(), m, k, cu_count);
    }
    check_hip(hipEventRecord(stop), "hipEventRecord wvSplitK stop");
    check_hip(hipEventSynchronize(stop), "hipEventSynchronize wvSplitK");
    float milliseconds = 0.0f;
    check_hip(hipEventElapsedTime(&milliseconds, start, stop),
              "hipEventElapsedTime wvSplitK");
    result.measured_ms.push_back(milliseconds / kLaunchesPerSample);
  }
  std::vector<double> sorted = result.measured_ms;
  std::sort(sorted.begin(), sorted.end());
  result.median_ms = sorted[sorted.size() / 2];
  result.effective_weight_bandwidth_gbs =
      static_cast<double>(m * k * sizeof(__hip_bfloat16)) /
      (result.median_ms * 1.0e6);

  std::vector<std::uint16_t> output_bits(m);
  check_hip(hipMemcpy(output_bits.data(), output.get(),
                      output_bits.size() * sizeof(output_bits[0]),
                      hipMemcpyDeviceToHost),
            "hipMemcpy wvSplitK output");
  result.output_bf16_sha256 =
      sha256_bytes(output_bits.data(), output_bits.size() * sizeof(output_bits[0]));

  double squared_error = 0.0;
  double squared_reference = 0.0;
  for (std::size_t row = 0; row < m; ++row) {
    float reference = 0.0f;
    for (std::size_t column = 0; column < k; ++column) {
      const float weight_value =
          pattern_value(row * k + column, 13, 7, 17, 8, 64.0f);
      const float activation_value =
          pattern_value(column, 11, 3, 19, 9, 32.0f);
      reference += weight_value * activation_value;
    }
    const double actual = bf16_bits_to_float(output_bits[row]);
    if (std::isfinite(actual)) ++result.finite_elements;
    const double error = actual - static_cast<double>(reference);
    result.maximum_absolute_error =
        std::max(result.maximum_absolute_error, std::abs(error));
    squared_error += error * error;
    squared_reference += static_cast<double>(reference) * reference;
  }
  result.relative_l2_error =
      std::sqrt(squared_error / std::max(squared_reference, 1.0e-30));
  return result;
}

}  // namespace

void launch_bf16_wvsplitk(const void* weight_mk, const void* activation_1k,
                          const void* bias_m, void* output_1m,
                          std::size_t m, std::size_t k, int cu_count,
                          void* stream) {
  if (weight_mk == nullptr || activation_1k == nullptr || output_1m == nullptr) {
    throw std::invalid_argument("BF16 wvSplitK requires non-null weight, activation, and output");
  }
  if (m <= 8 || m % kYTile != 0 || k == 0 || k % kActivationChunk != 0 ||
      k > static_cast<std::size_t>(kLdsBf16Elements) || cu_count <= 0) {
    throw std::invalid_argument("unsupported BF16 wvSplitK N=1 shape");
  }
  if (m > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      k > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("BF16 wvSplitK dimensions exceed int32");
  }
  const int active_waves = minimum_divisor(
      static_cast<int>(m), cu_count * kYTile, kWavesPerGroup);
  if (active_waves <= 0 || active_waves > kWavesPerGroup) {
    throw std::runtime_error("BF16 wvSplitK could not select an active-wave count");
  }
  hipLaunchKernelGGL(
      bf16_wvsplitk_n1_small_kernel, dim3(cu_count),
      dim3(kThreads, kWavesPerGroup), 0, static_cast<hipStream_t>(stream),
      static_cast<int>(k), static_cast<int>(k), static_cast<int>(k),
      static_cast<int>(m), static_cast<const __hip_bfloat16*>(weight_mk),
      static_cast<const __hip_bfloat16*>(activation_1k),
      static_cast<const __hip_bfloat16*>(bias_m),
      static_cast<__hip_bfloat16*>(output_1m), active_waves, cu_count);
  check_hip(hipGetLastError(), "bf16_wvsplitk_n1_small_kernel");
}

Bf16WvSplitKProbeResult probe_bf16_wvsplitk() {
  Bf16WvSplitKProbeResult result;
  hipDeviceProp_t properties{};
  check_hip(hipGetDeviceProperties(&properties, 0),
            "hipGetDeviceProperties wvSplitK");
  result.gpu_arch = properties.gcnArchName;
  if (result.gpu_arch.find("gfx1151") != 0) {
    throw std::runtime_error("native BF16 wvSplitK probe requires gfx1151, got " +
                             result.gpu_arch);
  }
  const int cu_count = properties.multiProcessorCount;
  result.cases.push_back(probe_case(2048, 512, cu_count));
  result.cases.push_back(probe_case(2048, 4096, cu_count));
  result.cases.push_back(probe_case(8192, 2048, cu_count));
  result.cases.push_back(probe_case(4096, 2048, cu_count));
  result.cases.push_back(probe_case(32, 2048, cu_count));
  return result;
}

}  // namespace aima
