// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_lm_head_certificate.h"

#include "aima/bf16_wvsplitk.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace aima {
namespace {

constexpr std::size_t kVocabulary = 248320;
constexpr std::size_t kHidden = 2048;
constexpr unsigned kThreads = 256;
constexpr unsigned kBoundBlocks = 256;
constexpr std::size_t kPartialOffset = 256;
constexpr std::size_t kCandidateIdOffset =
    kPartialOffset + kBoundBlocks * sizeof(float);
static_assert(kCandidateIdOffset +
                  kNativeLmHeadCandidateCapacity * sizeof(std::uint32_t) <=
              kNativeLmHeadCertificateScratchBytes);

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

__device__ bool better(float candidate_value, std::uint32_t candidate_index,
                       float current_value, std::uint32_t current_index) {
  return candidate_value > current_value ||
         (candidate_value == current_value && candidate_index < current_index);
}

__global__ void conservative_hidden_l2_kernel(
    const __hip_bfloat16* hidden, NativeLmHeadCertificateWire* wire) {
  __shared__ double reduction[kThreads];
  double local_squared = 0.0;
  for (std::size_t index = threadIdx.x; index < kHidden;
       index += blockDim.x) {
    const double value = static_cast<double>(__bfloat162float(hidden[index]));
    local_squared += value * value;
  }
  reduction[threadIdx.x] = local_squared;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float norm = static_cast<float>(sqrt(reduction[0]));
    // The row residual norms are already inflated by at least one ULP.  Keep
    // the hidden norm outward-rounded as well so a rounded product cannot
    // invalidate the certificate.
    for (int step = 0; step < 4; ++step) {
      norm = nextafterf(norm, INFINITY);
    }
    wire->hidden_l2 = norm;
  }
}

__global__ void bound_logits_kernel(const float* logits,
                                    const float* residual_l2,
                                    NativeLmHeadCertificateWire* wire,
                                    float* partial_lower_max) {
  __shared__ float reduction[kThreads];
  float local_max = -std::numeric_limits<float>::infinity();
  for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
       row < kVocabulary; row += gridDim.x * blockDim.x) {
    float error = residual_l2[row] * wire->hidden_l2;
    error = nextafterf(error, INFINITY);
    const float approximate = logits[row];
    const float lower = nextafterf(approximate - error, -INFINITY);
    local_max = fmaxf(local_max, lower);
  }
  reduction[threadIdx.x] = local_max;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) partial_lower_max[blockIdx.x] = reduction[0];
}

__global__ void reduce_lower_bound_kernel(
    const float* partial_lower_max, NativeLmHeadCertificateWire* wire) {
  __shared__ float reduction[kThreads];
  reduction[threadIdx.x] = partial_lower_max[threadIdx.x];
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    wire->maximum_lower_bound = reduction[0];
    wire->candidate_count = 0;
    wire->overflow = 0;
  }
}

__global__ void collect_candidates_kernel(
    const float* approximate_logits, const float* residual_l2,
    NativeLmHeadCertificateWire* wire,
    std::uint32_t* candidate_ids) {
  const std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= kVocabulary) {
    return;
  }
  float error = residual_l2[row] * wire->hidden_l2;
  error = nextafterf(error, INFINITY);
  const float upper =
      nextafterf(approximate_logits[row] + error, INFINITY);
  if (upper < wire->maximum_lower_bound) {
    return;
  }
  const std::uint32_t slot = atomicAdd(&wire->candidate_count, 1U);
  if (slot < kNativeLmHeadCandidateCapacity) {
    candidate_ids[slot] = static_cast<std::uint32_t>(row);
  } else {
    wire->overflow = 1;
  }
}

__global__ void gather_candidate_weights_kernel(
    const __hip_bfloat16* raw_weight, const NativeLmHeadCertificateWire* wire,
    const std::uint32_t* candidate_ids, __hip_bfloat16* candidate_weights) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr std::size_t kElements =
      kNativeLmHeadCandidateCapacity * kHidden;
  if (index >= kElements) return;
  const std::size_t candidate = index / kHidden;
  const std::size_t column = index - candidate * kHidden;
  const std::uint32_t bounded_count =
      wire->candidate_count < kNativeLmHeadCandidateCapacity
          ? wire->candidate_count
          : kNativeLmHeadCandidateCapacity;
  const std::uint32_t source_candidate =
      candidate < bounded_count ? static_cast<std::uint32_t>(candidate) : 0U;
  const std::uint32_t row = candidate_ids[source_candidate];
  candidate_weights[index] = raw_weight[static_cast<std::size_t>(row) *
                                               kHidden + column];
}

__global__ void exact_candidate_argmax_kernel(
    const __hip_bfloat16* candidate_logits,
    NativeLmHeadCertificateWire* wire,
    const std::uint32_t* candidate_ids, float* full_logits) {
  __shared__ float values[kThreads];
  __shared__ std::uint32_t indices[kThreads];
  const std::uint32_t count =
      wire->candidate_count < kNativeLmHeadCandidateCapacity
          ? wire->candidate_count
          : kNativeLmHeadCandidateCapacity;
  float local_value = -std::numeric_limits<float>::infinity();
  std::uint32_t local_index = 0;
  for (std::uint32_t candidate = threadIdx.x; candidate < count;
       candidate += blockDim.x) {
    const float value = __bfloat162float(candidate_logits[candidate]);
    const std::uint32_t token = candidate_ids[candidate];
    // Preserve the production full-logit representation: the resident int8
    // scan remains in-place and every exact certificate candidate replaces
    // its approximate value.  This also makes the complete distribution
    // available to the qualification path without a second vocabulary GEMV.
    full_logits[token] = value;
    if (better(value, token, local_value, local_index)) {
      local_value = value;
      local_index = token;
    }
  }
  values[threadIdx.x] = local_value;
  indices[threadIdx.x] = local_index;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride &&
        better(values[threadIdx.x + stride], indices[threadIdx.x + stride],
               values[threadIdx.x], indices[threadIdx.x])) {
      values[threadIdx.x] = values[threadIdx.x + stride];
      indices[threadIdx.x] = indices[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    wire->exact_top1_logit = values[0];
    wire->exact_top1_token_id = indices[0];
  }
}

}  // namespace

NativeLmHeadCertificateLaunchMetrics launch_native_lm_head_certificate(
    const void* raw_weight_bf16, const void* residual_l2_fp32,
    const void* hidden_bf16, void* approximate_logits_fp32,
    void* candidate_weights_bf16, std::size_t candidate_weight_bytes,
    void* candidate_logits_bf16, std::size_t candidate_logit_bytes,
    void* scratch, std::size_t scratch_bytes, int cu_count,
    void* stream_value) {
  if (raw_weight_bf16 == nullptr || residual_l2_fp32 == nullptr ||
      hidden_bf16 == nullptr || approximate_logits_fp32 == nullptr ||
      candidate_weights_bf16 == nullptr || candidate_logits_bf16 == nullptr ||
      scratch == nullptr ||
      candidate_weight_bytes < kNativeLmHeadCandidateWeightBytes ||
      candidate_logit_bytes < kNativeLmHeadCandidateLogitBytes ||
      scratch_bytes < kNativeLmHeadCertificateScratchBytes || cu_count <= 0) {
    throw std::invalid_argument("native LM-head certificate closure is incomplete");
  }
  hipStream_t stream = static_cast<hipStream_t>(stream_value);
  auto* scratch_bytes_pointer = static_cast<unsigned char*>(scratch);
  auto* wire = reinterpret_cast<NativeLmHeadCertificateWire*>(scratch);
  auto* partial_lower =
      reinterpret_cast<float*>(scratch_bytes_pointer + kPartialOffset);
  auto* candidate_ids = reinterpret_cast<std::uint32_t*>(
      scratch_bytes_pointer + kCandidateIdOffset);

  hipLaunchKernelGGL(
      conservative_hidden_l2_kernel, dim3(1), dim3(kThreads), 0, stream,
      static_cast<const __hip_bfloat16*>(hidden_bf16), wire);
  check_hip(hipGetLastError(), "conservative_hidden_l2_kernel");
  hipLaunchKernelGGL(
      bound_logits_kernel, dim3(kBoundBlocks), dim3(kThreads), 0, stream,
      static_cast<float*>(approximate_logits_fp32),
      static_cast<const float*>(residual_l2_fp32), wire, partial_lower);
  check_hip(hipGetLastError(), "bound_logits_kernel");
  hipLaunchKernelGGL(reduce_lower_bound_kernel, dim3(1), dim3(kThreads), 0,
                     stream, partial_lower, wire);
  check_hip(hipGetLastError(), "reduce_lower_bound_kernel");
  constexpr unsigned kCandidateBlocks =
      static_cast<unsigned>((kVocabulary + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      collect_candidates_kernel, dim3(kCandidateBlocks), dim3(kThreads), 0,
      stream, static_cast<const float*>(approximate_logits_fp32),
      static_cast<const float*>(residual_l2_fp32), wire, candidate_ids);
  check_hip(hipGetLastError(), "collect_candidates_kernel");
  constexpr std::size_t kCandidateWeightElements =
      kNativeLmHeadCandidateCapacity * kHidden;
  constexpr unsigned kGatherBlocks = static_cast<unsigned>(
      (kCandidateWeightElements + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      gather_candidate_weights_kernel, dim3(kGatherBlocks), dim3(kThreads), 0,
      stream, static_cast<const __hip_bfloat16*>(raw_weight_bf16), wire,
      candidate_ids,
      static_cast<__hip_bfloat16*>(candidate_weights_bf16));
  check_hip(hipGetLastError(), "gather_candidate_weights_kernel");
  launch_bf16_wvsplitk(
      candidate_weights_bf16, hidden_bf16, nullptr, candidate_logits_bf16,
      kNativeLmHeadCandidateCapacity, kHidden, cu_count, stream);
  hipLaunchKernelGGL(
      exact_candidate_argmax_kernel, dim3(1), dim3(kThreads), 0, stream,
      static_cast<const __hip_bfloat16*>(candidate_logits_bf16), wire,
      candidate_ids, static_cast<float*>(approximate_logits_fp32));
  check_hip(hipGetLastError(), "exact_candidate_argmax_kernel");

  NativeLmHeadCertificateLaunchMetrics metrics;
  metrics.native_kernel_launches = 7;
  metrics.device_wire = wire;
  return metrics;
}

}  // namespace aima
