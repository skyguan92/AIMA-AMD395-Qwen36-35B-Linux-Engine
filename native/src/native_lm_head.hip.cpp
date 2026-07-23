// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_lm_head.h"

#include "aima/sha256.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima {
namespace {

constexpr std::size_t kRows = 248320;
constexpr std::size_t kColumns = 2048;
constexpr std::size_t kQBytes = kRows * kColumns;
constexpr std::size_t kMetadataBytes = kRows * sizeof(float);
constexpr std::size_t kPayloadBytes = kQBytes + 2 * kMetadataBytes;
constexpr int kThreads = 256;
constexpr int kResidualSafetyUlps = 8;

constexpr const char* kReferenceQSha256 =
    "ba779dbee92989c1247a2075b50a5864f6fb992285d6b597a1317525d3452603";
constexpr const char* kReferenceScalesSha256 =
    "0856b4bf881f7bf47dbe3f72e21614f665e01f696e83006ee0b9074d96eadb43";
constexpr const char* kReferenceResidualSha256 =
    "c3c5538387c3cc2e43aeba5a39e61c6d85c2ddacbe3687c114c372db5b32c869";

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorName(status) + " (" +
                             hipGetErrorString(status) + ")");
  }
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

__global__ void quantize_lm_head_rowwise_kernel(
    const hip_bfloat16* source, std::int8_t* q_weight, float* scales,
    float* residual_l2) {
  __shared__ float reduction[kThreads];
  __shared__ double squared_reduction[kThreads];
  const std::size_t row = blockIdx.x;
  const std::size_t base = row * kColumns;
  float local_max = 0.0f;
  for (std::size_t column = threadIdx.x; column < kColumns;
       column += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(static_cast<float>(source[base + column])));
  }
  reduction[threadIdx.x] = local_max;
  __syncthreads();
  for (unsigned offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  // The qualified Torch pointwise kernel evaluates division by the constant
  // 127 as multiplication by its rounded FP32 reciprocal.  An IEEE division is
  // one ULP higher for about five percent of rows, so encode that qualified
  // operation explicitly.
  const float maximum = fmaxf(reduction[0], 1.0e-12f);
  constexpr float kInverse127 = 1.0f / 127.0f;
  const float scale = maximum * kInverse127;
  if (threadIdx.x == 0) scales[row] = scale;

  double local_squared = 0.0;
  for (std::size_t column = threadIdx.x; column < kColumns;
       column += blockDim.x) {
    const float value = static_cast<float>(source[base + column]);
    const float rounded = nearbyintf(value / scale);
    const int quantized =
        static_cast<int>(fminf(127.0f, fmaxf(-127.0f, rounded)));
    q_weight[base + column] = static_cast<std::int8_t>(quantized);
    const float reconstructed = static_cast<float>(quantized) * scale;
    const float residual = value - reconstructed;
    local_squared += static_cast<double>(residual) *
                     static_cast<double>(residual);
  }
  squared_reduction[threadIdx.x] = local_squared;
  __syncthreads();
  for (unsigned offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      squared_reduction[threadIdx.x] +=
          squared_reduction[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float rounded_norm =
        static_cast<float>(sqrt(squared_reduction[0]));
    float conservative_norm = rounded_norm;
    for (int step = 0; step < kResidualSafetyUlps; ++step) {
      conservative_norm = nextafterf(conservative_norm, INFINITY);
    }
    residual_l2[row] = conservative_norm;
  }
}

std::string device_sha256(const void* pointer, std::size_t bytes) {
  std::vector<unsigned char> host(bytes);
  check_hip(hipMemcpy(host.data(), pointer, bytes, hipMemcpyDeviceToHost),
            "hipMemcpy LM-head hash payload");
  return sha256_bytes(host.data(), host.size());
}

}  // namespace

NativeLmHeadStore::~NativeLmHeadStore() { reset(); }

NativeLmHeadMetrics NativeLmHeadStore::build(const NativeWeightStore& weights,
                                             int device) {
  if (built() || !weights.loaded()) {
    throw std::runtime_error(
        "native LM-head quantization requires one loaded, unbuilt store");
  }
  const NativeTensorView* source = weights.find("lm_head.weight");
  if (source == nullptr || source->rank != 2 || source->shape[0] != kRows ||
      source->shape[1] != kColumns ||
      source->payload_bytes != kRows * kColumns * sizeof(hip_bfloat16)) {
    throw std::runtime_error("native LM-head source shape mismatch");
  }
  const auto started = std::chrono::steady_clock::now();
  device_ = device;
  check_hip(hipSetDevice(device_), "hipSetDevice native LM head");
  NativeLmHeadMetrics metrics;
  std::size_t total = 0;
  check_hip(hipMemGetInfo(&metrics.free_bytes_before, &total),
            "hipMemGetInfo before native LM head");
  if (metrics.free_bytes_before < kPayloadBytes + 64ULL * 1024 * 1024) {
    throw std::runtime_error("insufficient device memory for native LM head");
  }
  try {
    const auto allocation_started = std::chrono::steady_clock::now();
    check_hip(hipMalloc(&allocation_, kPayloadBytes),
              "hipMalloc native LM head");
    auto* base = static_cast<unsigned char*>(allocation_);
    q_weight_ = base;
    scales_ = base + kQBytes;
    residual_l2_ = base + kQBytes + kMetadataBytes;
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize LM-head allocation");
    metrics.allocation_ms = elapsed_ms(allocation_started);

    const auto quantize_started = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(quantize_lm_head_rowwise_kernel, dim3(kRows),
                       dim3(kThreads), 0, nullptr,
                       static_cast<const hip_bfloat16*>(source->device_pointer),
                       static_cast<std::int8_t*>(q_weight_),
                       static_cast<float*>(scales_),
                       static_cast<float*>(residual_l2_));
    check_hip(hipGetLastError(), "quantize_lm_head_rowwise_kernel");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize LM-head quantize");
    metrics.quantize_ms = elapsed_ms(quantize_started);

    const auto hash_started = std::chrono::steady_clock::now();
    metrics.q_weight_sha256 = device_sha256(q_weight_, kQBytes);
    metrics.scales_sha256 = device_sha256(scales_, kMetadataBytes);
    metrics.residual_l2_sha256 = device_sha256(residual_l2_, kMetadataBytes);
    metrics.hash_ms = elapsed_ms(hash_started);
    metrics.q_weight_reference_exact =
        metrics.q_weight_sha256 == kReferenceQSha256;
    metrics.scales_reference_exact =
        metrics.scales_sha256 == kReferenceScalesSha256;
    metrics.residual_l2_reference_exact =
        metrics.residual_l2_sha256 == kReferenceResidualSha256;
    constexpr std::array<std::size_t, 6> q_sample_offsets = {
        0,
        kColumns - 1,
        kColumns + 1,
        (kRows / 2) * kColumns + kColumns / 2,
        (kRows - 2) * kColumns + kColumns - 2,
        (kRows - 1) * kColumns + kColumns - 1,
    };
    constexpr std::array<std::size_t, 5> row_samples = {
        0, 1, kRows / 2, kRows - 2, kRows - 1,
    };
    for (std::size_t index = 0; index < q_sample_offsets.size(); ++index) {
      check_hip(hipMemcpy(&metrics.q_weight_samples[index],
                          static_cast<const std::int8_t*>(q_weight_) +
                              q_sample_offsets[index],
                          sizeof(std::int8_t), hipMemcpyDeviceToHost),
                "hipMemcpy LM-head q sample");
    }
    for (std::size_t index = 0; index < row_samples.size(); ++index) {
      check_hip(hipMemcpy(&metrics.scale_samples[index],
                          static_cast<const float*>(scales_) + row_samples[index],
                          sizeof(float), hipMemcpyDeviceToHost),
                "hipMemcpy LM-head scale sample");
      check_hip(hipMemcpy(&metrics.residual_l2_samples[index],
                          static_cast<const float*>(residual_l2_) +
                              row_samples[index],
                          sizeof(float), hipMemcpyDeviceToHost),
                "hipMemcpy LM-head residual sample");
    }
    check_hip(hipMemGetInfo(&metrics.free_bytes_after, &total),
              "hipMemGetInfo after native LM head");
    metrics.payload_bytes = kPayloadBytes;
    metrics.build_wall_ms = elapsed_ms(started);
    return metrics;
  } catch (...) {
    reset();
    throw;
  }
}

void NativeLmHeadStore::reset() noexcept {
  (void)hipSetDevice(device_);
  if (allocation_) (void)hipFree(allocation_);
  allocation_ = nullptr;
  q_weight_ = nullptr;
  scales_ = nullptr;
  residual_l2_ = nullptr;
}

void NativeLmHeadStore::write_scales_for_validation(
    const std::filesystem::path& path) const {
  if (!built() || scales_ == nullptr) {
    throw std::runtime_error("native LM-head scales are not built");
  }
  std::vector<float> host(kRows);
  check_hip(hipMemcpy(host.data(), scales_, kMetadataBytes,
                      hipMemcpyDeviceToHost),
            "hipMemcpy LM-head validation scales");
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.write(reinterpret_cast<const char*>(host.data()),
                    static_cast<std::streamsize>(kMetadataBytes))) {
    throw std::runtime_error("failed to write native LM-head validation scales");
  }
}

void NativeLmHeadStore::write_residual_l2_for_validation(
    const std::filesystem::path& path) const {
  if (!built() || residual_l2_ == nullptr) {
    throw std::runtime_error("native LM-head residual metadata is not built");
  }
  std::vector<float> host(kRows);
  check_hip(hipMemcpy(host.data(), residual_l2_, kMetadataBytes,
                      hipMemcpyDeviceToHost),
            "hipMemcpy LM-head validation residual L2");
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.write(reinterpret_cast<const char*>(host.data()),
                    static_cast<std::streamsize>(kMetadataBytes))) {
    throw std::runtime_error(
        "failed to write native LM-head validation residual L2");
  }
}

}  // namespace aima
