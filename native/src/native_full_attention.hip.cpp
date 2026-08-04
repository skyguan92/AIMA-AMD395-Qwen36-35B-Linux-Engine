// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_full_attention.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace aima {
namespace {

constexpr std::size_t kFullLayers = 10;
constexpr std::size_t kKvHeads = 2;
constexpr std::size_t kQueryHeadsPerKv = 8;
constexpr std::size_t kQueryHeads = kKvHeads * kQueryHeadsPerKv;
constexpr std::size_t kHeadDimension = 256;
constexpr std::size_t kQueryDimension = kQueryHeads * kHeadDimension;
constexpr std::size_t kMaximumPvSplits = 64;
constexpr std::size_t kQkSequencesPerBlock = 8;
constexpr std::size_t kSplitSoftmaxTokensPerBlock = 4096;
constexpr std::size_t kSplitSoftmaxMinimumTokens = 65536;
constexpr std::uint64_t kAlignment = 256;
constexpr std::size_t kQkWorkspaceLimit = 76ULL * 1024 * 1024;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

void check_blas(hipblasStatus_t status, const char* operation) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": status=" +
                             std::to_string(static_cast<int>(status)));
  }
}

std::uint64_t align_up(std::uint64_t value) {
  return (value + kAlignment - 1) / kAlignment * kAlignment;
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

__global__ void write_kv_kernel(const __hip_bfloat16* normalized_k,
                                const __hip_bfloat16* raw_v,
                                __hip_bfloat16* k_cache,
                                __hip_bfloat16* v_cache,
                                std::size_t position) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kKvHeads * kHeadDimension) return;
  const std::size_t cache_offset =
      position * kKvHeads * kHeadDimension + index;
  k_cache[cache_offset] = normalized_k[index];
  v_cache[cache_offset] = raw_v[index];
}

__global__ void scaled_softmax_bf16_kernel(__hip_bfloat16* scores,
                                           __hip_bfloat16* probabilities,
                                           std::size_t cache_end,
                                           std::size_t cache_stride) {
  __shared__ float reduction[256];
  const std::size_t head = blockIdx.x;
  const std::size_t thread = threadIdx.x;
  const std::size_t base = head * cache_stride;
  float local_max = -std::numeric_limits<float>::infinity();
  for (std::size_t index = thread; index < cache_end; index += blockDim.x) {
    const __hip_bfloat16 scaled = __float2bfloat16(
        __bfloat162float(scores[base + index]) * 0.0625f);
    scores[base + index] = scaled;
    local_max = fmaxf(local_max, __bfloat162float(scaled));
  }
  reduction[thread] = local_max;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (thread < stride) {
      reduction[thread] = fmaxf(reduction[thread], reduction[thread + stride]);
    }
    __syncthreads();
  }
  const float maximum = reduction[0];
  float local_sum = 0.0f;
  for (std::size_t index = thread; index < cache_end; index += blockDim.x) {
    local_sum += expf(__bfloat162float(scores[base + index]) - maximum);
  }
  reduction[thread] = local_sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (thread < stride) reduction[thread] += reduction[thread + stride];
    __syncthreads();
  }
  const float inverse_sum = 1.0f / reduction[0];
  for (std::size_t index = thread; index < cache_end; index += blockDim.x) {
    probabilities[base + index] = __float2bfloat16(
        expf(__bfloat162float(scores[base + index]) - maximum) * inverse_sum);
  }
}

__global__ void split_softmax_partial_max_kernel(
    const __hip_bfloat16* scores, float* partial_maxima,
    std::size_t cache_end, std::size_t cache_stride,
    std::size_t split_count) {
  __shared__ float reduction[256];
  const std::size_t split = blockIdx.x;
  const std::size_t head = blockIdx.y;
  const std::size_t thread = threadIdx.x;
  const std::size_t begin = cache_end * split / split_count;
  const std::size_t end = cache_end * (split + 1) / split_count;
  const std::size_t base = head * cache_stride;
  float local_max = -std::numeric_limits<float>::infinity();
  for (std::size_t index = begin + thread; index < end;
       index += blockDim.x) {
    const __hip_bfloat16 scaled = __float2bfloat16(
        __bfloat162float(scores[base + index]) * 0.0625f);
    local_max = fmaxf(local_max, __bfloat162float(scaled));
  }
  reduction[thread] = local_max;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (thread < stride) {
      reduction[thread] = fmaxf(reduction[thread], reduction[thread + stride]);
    }
    __syncthreads();
  }
  if (thread == 0) {
    partial_maxima[head * kMaximumPvSplits + split] = reduction[0];
  }
}

__global__ void split_softmax_reduce_max_kernel(
    const float* partial_maxima, float* head_maxima,
    std::size_t split_count) {
  __shared__ float reduction[256];
  const std::size_t head = blockIdx.x;
  const std::size_t thread = threadIdx.x;
  float value = -std::numeric_limits<float>::infinity();
  for (std::size_t split = thread; split < split_count;
       split += blockDim.x) {
    value = fmaxf(value,
                  partial_maxima[head * kMaximumPvSplits + split]);
  }
  reduction[thread] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (thread < stride) {
      reduction[thread] = fmaxf(reduction[thread], reduction[thread + stride]);
    }
    __syncthreads();
  }
  if (thread == 0) head_maxima[head] = reduction[0];
}

__global__ void split_softmax_partial_exp_sum_kernel(
    const __hip_bfloat16* scores, const float* head_maxima,
    float* exponentials, float* partial_sums,
    std::size_t cache_end, std::size_t cache_stride,
    std::size_t split_count) {
  __shared__ float reduction[256];
  const std::size_t split = blockIdx.x;
  const std::size_t head = blockIdx.y;
  const std::size_t thread = threadIdx.x;
  const std::size_t begin = cache_end * split / split_count;
  const std::size_t end = cache_end * (split + 1) / split_count;
  const std::size_t base = head * cache_stride;
  const float maximum = head_maxima[head];
  float local_sum = 0.0f;
  for (std::size_t index = begin + thread; index < end;
       index += blockDim.x) {
    const __hip_bfloat16 scaled = __float2bfloat16(
        __bfloat162float(scores[base + index]) * 0.0625f);
    const float value = expf(__bfloat162float(scaled) - maximum);
    exponentials[base + index] = value;
    local_sum += value;
  }
  reduction[thread] = local_sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (thread < stride) reduction[thread] += reduction[thread + stride];
    __syncthreads();
  }
  if (thread == 0) {
    partial_sums[head * kMaximumPvSplits + split] = reduction[0];
  }
}

__global__ void split_softmax_reduce_sum_kernel(
    const float* partial_sums, float* head_inverse_sums,
    std::size_t split_count) {
  __shared__ float reduction[256];
  const std::size_t head = blockIdx.x;
  const std::size_t thread = threadIdx.x;
  float value = 0.0f;
  for (std::size_t split = thread; split < split_count;
       split += blockDim.x) {
    value += partial_sums[head * kMaximumPvSplits + split];
  }
  reduction[thread] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (thread < stride) reduction[thread] += reduction[thread + stride];
    __syncthreads();
  }
  if (thread == 0) head_inverse_sums[head] = 1.0f / reduction[0];
}

__global__ void split_softmax_normalize_kernel(
    const float* exponentials, const float* head_inverse_sums,
    __hip_bfloat16* probabilities, std::size_t cache_end,
    std::size_t cache_stride, std::size_t split_count) {
  const std::size_t split = blockIdx.x;
  const std::size_t head = blockIdx.y;
  const std::size_t begin = cache_end * split / split_count;
  const std::size_t end = cache_end * (split + 1) / split_count;
  const std::size_t base = head * cache_stride;
  const float inverse_sum = head_inverse_sums[head];
  for (std::size_t index = begin + threadIdx.x; index < end;
       index += blockDim.x) {
    probabilities[base + index] =
        __float2bfloat16(exponentials[base + index] * inverse_sum);
  }
}

// One block owns (KV head, sequence split).  Each lane owns one V dimension,
// reads that V once, and accumulates all eight query heads.  This avoids the
// eightfold V-cache traffic of a block-per-query-head implementation.
__global__ void grouped_pv_partial_kernel(
    const __hip_bfloat16* probabilities, const __hip_bfloat16* v_cache,
    float* partials, std::size_t cache_end, std::size_t split_count) {
  const std::size_t split = blockIdx.x;
  const std::size_t kv_head = blockIdx.y;
  const std::size_t dimension = threadIdx.x;
  const std::size_t begin = cache_end * split / split_count;
  const std::size_t end = cache_end * (split + 1) / split_count;
  float sums[kQueryHeadsPerKv] = {};
  for (std::size_t position = begin; position < end; ++position) {
    const float value = __bfloat162float(
        v_cache[(position * kKvHeads + kv_head) * kHeadDimension + dimension]);
#pragma unroll
    for (std::size_t query = 0; query < kQueryHeadsPerKv; ++query) {
      const std::size_t head = kv_head * kQueryHeadsPerKv + query;
      const float probability =
          __bfloat162float(probabilities[head * cache_end + position]);
      sums[query] = fmaf(probability, value, sums[query]);
    }
  }
#pragma unroll
  for (std::size_t query = 0; query < kQueryHeadsPerKv; ++query) {
    const std::size_t head = kv_head * kQueryHeadsPerKv + query;
    partials[(head * kMaximumPvSplits + split) * kHeadDimension + dimension] =
        sums[query];
  }
}

__global__ void grouped_pv_reduce_kernel(const float* partials,
                                         __hip_bfloat16* attention,
                                         std::size_t split_count) {
  const std::size_t head = blockIdx.x;
  const std::size_t dimension = threadIdx.x;
  float sum = 0.0f;
  for (std::size_t split = 0; split < split_count; ++split) {
    sum += partials[(head * kMaximumPvSplits + split) * kHeadDimension +
                    dimension];
  }
  attention[head * kHeadDimension + dimension] = __float2bfloat16(sum);
}

}  // namespace

struct NativeFullAttentionQkPlan {
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulDesc_t operation = nullptr;
  hipblasLtMatrixLayout_t a_layout = nullptr;
  hipblasLtMatrixLayout_t b_layout = nullptr;
  hipblasLtMatrixLayout_t c_layout = nullptr;
  hipblasLtMatrixLayout_t d_layout = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  hipblasLtMatmulAlgo_t algorithm{};
  hipblasLtMatmulDesc_t pv_operation = nullptr;
  hipblasLtMatrixLayout_t pv_a_layout = nullptr;
  hipblasLtMatrixLayout_t pv_b_layout = nullptr;
  hipblasLtMatrixLayout_t pv_c_layout = nullptr;
  hipblasLtMatrixLayout_t pv_d_layout = nullptr;
  hipblasLtMatmulAlgo_t pv_algorithm{};
  void* workspace = nullptr;
  std::size_t workspace_bytes = 0;
  int heuristic_count = 0;
  int pv_heuristic_count = 0;

  explicit NativeFullAttentionQkPlan(std::size_t cache_capacity) {
    try {
      check_blas(hipblasLtCreate(&handle), "hipblasLtCreate grouped QK");
      check_blas(hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F,
                                           HIP_R_32F),
                 "hipblasLtMatmulDescCreate grouped QK");
      const hipblasOperation_t transpose = HIPBLAS_OP_T;
      const hipblasOperation_t no_transpose = HIPBLAS_OP_N;
      check_blas(hipblasLtMatmulDescSetAttribute(
                     operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose,
                     sizeof(transpose)),
                 "hipblasLtMatmulDescSetAttribute grouped QK A");
      check_blas(hipblasLtMatmulDescSetAttribute(
                     operation, HIPBLASLT_MATMUL_DESC_TRANSB, &no_transpose,
                     sizeof(no_transpose)),
                 "hipblasLtMatmulDescSetAttribute grouped QK B");
      check_blas(hipblasLtMatrixLayoutCreate(
                     &a_layout, HIP_R_16BF, kHeadDimension, cache_capacity,
                     kKvHeads * kHeadDimension),
                 "hipblasLtMatrixLayoutCreate grouped QK K");
      check_blas(hipblasLtMatrixLayoutCreate(
                     &b_layout, HIP_R_16BF, kHeadDimension,
                     kQueryHeadsPerKv, kHeadDimension),
                 "hipblasLtMatrixLayoutCreate grouped QK Q");
      check_blas(hipblasLtMatrixLayoutCreate(
                     &c_layout, HIP_R_16BF, cache_capacity,
                     kQueryHeadsPerKv, cache_capacity),
                 "hipblasLtMatrixLayoutCreate grouped QK C");
      check_blas(hipblasLtMatrixLayoutCreate(
                     &d_layout, HIP_R_16BF, cache_capacity,
                     kQueryHeadsPerKv, cache_capacity),
                 "hipblasLtMatrixLayoutCreate grouped QK D");
      const std::int32_t batch_count = kKvHeads;
      const std::int64_t a_stride = kHeadDimension;
      const std::int64_t b_stride = kQueryHeadsPerKv * kHeadDimension;
      const std::int64_t output_stride =
          cache_capacity * kQueryHeadsPerKv;
      for (hipblasLtMatrixLayout_t layout :
           {a_layout, b_layout, c_layout, d_layout}) {
        check_blas(hipblasLtMatrixLayoutSetAttribute(
                       layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                       &batch_count, sizeof(batch_count)),
                   "hipblasLtMatrixLayoutSetAttribute grouped QK batch");
      }
      check_blas(hipblasLtMatrixLayoutSetAttribute(
                     a_layout,
                     HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride,
                     sizeof(a_stride)),
                 "hipblasLtMatrixLayoutSetAttribute grouped QK K stride");
      check_blas(hipblasLtMatrixLayoutSetAttribute(
                     b_layout,
                     HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride,
                     sizeof(b_stride)),
                 "hipblasLtMatrixLayoutSetAttribute grouped QK Q stride");
      check_blas(hipblasLtMatrixLayoutSetAttribute(
                     c_layout,
                     HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                     &output_stride, sizeof(output_stride)),
                 "hipblasLtMatrixLayoutSetAttribute grouped QK C stride");
      check_blas(hipblasLtMatrixLayoutSetAttribute(
                     d_layout,
                     HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                     &output_stride, sizeof(output_stride)),
                 "hipblasLtMatrixLayoutSetAttribute grouped QK D stride");
      check_blas(hipblasLtMatmulPreferenceCreate(&preference),
                 "hipblasLtMatmulPreferenceCreate grouped QK");
      const std::size_t workspace_limit = kQkWorkspaceLimit;
      check_blas(hipblasLtMatmulPreferenceSetAttribute(
                     preference,
                     HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                     &workspace_limit, sizeof(workspace_limit)),
                 "hipblasLtMatmulPreferenceSetAttribute grouped QK");
      std::array<hipblasLtMatmulHeuristicResult_t, 32> heuristics{};
      check_blas(hipblasLtMatmulAlgoGetHeuristic(
                     handle, operation, a_layout, b_layout, c_layout,
                     d_layout, preference, static_cast<int>(heuristics.size()),
                     heuristics.data(), &heuristic_count),
                 "hipblasLtMatmulAlgoGetHeuristic grouped QK");
      const auto selected = std::find_if(
          heuristics.begin(), heuristics.begin() + heuristic_count,
          [](const auto& value) {
            return value.state == HIPBLAS_STATUS_SUCCESS &&
                   value.workspaceSize <= kQkWorkspaceLimit;
          });
      if (selected == heuristics.begin() + heuristic_count) {
        throw std::runtime_error(
            "hipBLASLt returned no grouped QK algorithm");
      }
      algorithm = selected->algo;

      check_blas(hipblasLtMatmulDescCreate(&pv_operation,
                                           HIPBLAS_COMPUTE_32F, HIP_R_32F),
                 "hipblasLtMatmulDescCreate grouped PV");
      check_blas(hipblasLtMatmulDescSetAttribute(
                     pv_operation, HIPBLASLT_MATMUL_DESC_TRANSA,
                     &no_transpose, sizeof(no_transpose)),
                 "hipblasLtMatmulDescSetAttribute grouped PV A");
      check_blas(hipblasLtMatmulDescSetAttribute(
                     pv_operation, HIPBLASLT_MATMUL_DESC_TRANSB,
                     &no_transpose, sizeof(no_transpose)),
                 "hipblasLtMatmulDescSetAttribute grouped PV B");
      check_blas(hipblasLtMatrixLayoutCreate(
                     &pv_a_layout, HIP_R_16BF, kHeadDimension,
                     cache_capacity, kKvHeads * kHeadDimension),
                 "hipblasLtMatrixLayoutCreate grouped PV V");
      check_blas(hipblasLtMatrixLayoutCreate(
                     &pv_b_layout, HIP_R_16BF, cache_capacity,
                     kQueryHeadsPerKv, cache_capacity),
                 "hipblasLtMatrixLayoutCreate grouped PV probabilities");
      check_blas(hipblasLtMatrixLayoutCreate(
                     &pv_c_layout, HIP_R_16BF, kHeadDimension,
                     kQueryHeadsPerKv, kHeadDimension),
                 "hipblasLtMatrixLayoutCreate grouped PV C");
      check_blas(hipblasLtMatrixLayoutCreate(
                     &pv_d_layout, HIP_R_16BF, kHeadDimension,
                     kQueryHeadsPerKv, kHeadDimension),
                 "hipblasLtMatrixLayoutCreate grouped PV D");
      const std::int64_t pv_a_stride = kHeadDimension;
      const std::int64_t pv_b_stride = cache_capacity * kQueryHeadsPerKv;
      const std::int64_t pv_output_stride =
          kHeadDimension * kQueryHeadsPerKv;
      for (hipblasLtMatrixLayout_t layout :
           {pv_a_layout, pv_b_layout, pv_c_layout, pv_d_layout}) {
        check_blas(hipblasLtMatrixLayoutSetAttribute(
                       layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                       &batch_count, sizeof(batch_count)),
                   "hipblasLtMatrixLayoutSetAttribute grouped PV batch");
      }
      check_blas(hipblasLtMatrixLayoutSetAttribute(
                     pv_a_layout,
                     HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                     &pv_a_stride, sizeof(pv_a_stride)),
                 "hipblasLtMatrixLayoutSetAttribute grouped PV V stride");
      check_blas(hipblasLtMatrixLayoutSetAttribute(
                     pv_b_layout,
                     HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                     &pv_b_stride, sizeof(pv_b_stride)),
                 "hipblasLtMatrixLayoutSetAttribute grouped PV probability stride");
      check_blas(hipblasLtMatrixLayoutSetAttribute(
                     pv_c_layout,
                     HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                     &pv_output_stride, sizeof(pv_output_stride)),
                 "hipblasLtMatrixLayoutSetAttribute grouped PV C stride");
      check_blas(hipblasLtMatrixLayoutSetAttribute(
                     pv_d_layout,
                     HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                     &pv_output_stride, sizeof(pv_output_stride)),
                 "hipblasLtMatrixLayoutSetAttribute grouped PV D stride");
      std::array<hipblasLtMatmulHeuristicResult_t, 32> pv_heuristics{};
      check_blas(hipblasLtMatmulAlgoGetHeuristic(
                     handle, pv_operation, pv_a_layout, pv_b_layout,
                     pv_c_layout, pv_d_layout, preference,
                     static_cast<int>(pv_heuristics.size()),
                     pv_heuristics.data(), &pv_heuristic_count),
                 "hipblasLtMatmulAlgoGetHeuristic grouped PV");
      const auto pv_selected = std::find_if(
          pv_heuristics.begin(),
          pv_heuristics.begin() + pv_heuristic_count,
          [](const auto& value) {
            return value.state == HIPBLAS_STATUS_SUCCESS &&
                   value.workspaceSize <= kQkWorkspaceLimit;
          });
      if (pv_selected == pv_heuristics.begin() + pv_heuristic_count) {
        throw std::runtime_error(
            "hipBLASLt returned no grouped PV algorithm");
      }
      pv_algorithm = pv_selected->algo;
      workspace_bytes =
          std::max(selected->workspaceSize, pv_selected->workspaceSize);
      if (workspace_bytes != 0) {
        check_hip(hipMalloc(&workspace, workspace_bytes),
                  "hipMalloc grouped attention workspace");
      }
    } catch (...) {
      release();
      throw;
    }
  }

  ~NativeFullAttentionQkPlan() { release(); }

  void release() noexcept {
    if (workspace) (void)hipFree(workspace);
    if (preference) hipblasLtMatmulPreferenceDestroy(preference);
    if (pv_d_layout) hipblasLtMatrixLayoutDestroy(pv_d_layout);
    if (pv_c_layout) hipblasLtMatrixLayoutDestroy(pv_c_layout);
    if (pv_b_layout) hipblasLtMatrixLayoutDestroy(pv_b_layout);
    if (pv_a_layout) hipblasLtMatrixLayoutDestroy(pv_a_layout);
    if (pv_operation) hipblasLtMatmulDescDestroy(pv_operation);
    if (d_layout) hipblasLtMatrixLayoutDestroy(d_layout);
    if (c_layout) hipblasLtMatrixLayoutDestroy(c_layout);
    if (b_layout) hipblasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) hipblasLtMatrixLayoutDestroy(a_layout);
    if (operation) hipblasLtMatmulDescDestroy(operation);
    if (handle) hipblasLtDestroy(handle);
    workspace = nullptr;
    preference = nullptr;
    pv_d_layout = pv_c_layout = pv_b_layout = pv_a_layout = nullptr;
    pv_operation = nullptr;
    d_layout = c_layout = b_layout = a_layout = nullptr;
    operation = nullptr;
    handle = nullptr;
  }

  void launch(const void* q, const void* k_cache, void* scores,
              void* stream_value) const {
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    check_blas(hipblasLtMatmul(
                   handle, operation, &alpha, k_cache, a_layout, q, b_layout,
                   &beta, scores, c_layout, scores, d_layout, &algorithm,
                   workspace, workspace_bytes,
                   static_cast<hipStream_t>(stream_value)),
               "hipblasLtMatmul grouped QK");
  }

  void launch_pv(const void* probabilities, const void* v_cache,
                 void* attention, void* stream_value) const {
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    check_blas(hipblasLtMatmul(
                   handle, pv_operation, &alpha, v_cache, pv_a_layout,
                   probabilities, pv_b_layout, &beta, attention, pv_c_layout,
                   attention, pv_d_layout, &pv_algorithm, workspace,
                   workspace_bytes, static_cast<hipStream_t>(stream_value)),
               "hipblasLtMatmul grouped PV");
  }
};

NativeFullAttentionState::NativeFullAttentionState() = default;
NativeFullAttentionState::~NativeFullAttentionState() { reset(); }

std::size_t NativeFullAttentionState::layer_slot(std::size_t layer_index) {
  if (layer_index < 3 || layer_index > 39 || (layer_index - 3) % 4 != 0) {
    throw std::invalid_argument("layer is not a Qwen3.6 full-attention layer");
  }
  return (layer_index - 3) / 4;
}

NativeFullAttentionStateMetrics NativeFullAttentionState::build(
    std::size_t cache_capacity, int device) {
  if (built() || cache_capacity == 0) {
    throw std::invalid_argument(
        "native full-attention state requires one non-empty build");
  }
  NativeFullAttentionStateMetrics metrics;
  metrics.cache_capacity = cache_capacity;
  metrics.full_attention_layers = kFullLayers;
  metrics.maximum_pv_splits = kMaximumPvSplits;
  const std::uint64_t cache_bytes =
      cache_capacity * kKvHeads * kHeadDimension * sizeof(__hip_bfloat16);
  const std::uint64_t score_bytes =
      cache_capacity * kQueryHeads * sizeof(__hip_bfloat16);
  const std::uint64_t softmax_exponential_bytes =
      cache_capacity * kQueryHeads * sizeof(float);
  const std::uint64_t partial_bytes =
      kQueryHeads * kMaximumPvSplits * kHeadDimension * sizeof(float);
  metrics.cache_payload_bytes = 2 * kFullLayers * cache_bytes;
  metrics.scratch_payload_bytes =
      2 * score_bytes + softmax_exponential_bytes + partial_bytes +
      2 * kQueryDimension * sizeof(__hip_bfloat16) +
      2048 * sizeof(__hip_bfloat16);

  std::uint64_t offset = 0;
  std::array<std::uint64_t, kFullLayers> k_offsets{};
  std::array<std::uint64_t, kFullLayers> v_offsets{};
  for (std::size_t slot = 0; slot < kFullLayers; ++slot) {
    k_offsets[slot] = offset;
    offset += align_up(cache_bytes);
    v_offsets[slot] = offset;
    offset += align_up(cache_bytes);
  }
  const std::uint64_t scores_offset = offset;
  offset += align_up(score_bytes);
  const std::uint64_t probabilities_offset = offset;
  offset += align_up(score_bytes);
  const std::uint64_t softmax_exponentials_offset = offset;
  offset += align_up(softmax_exponential_bytes);
  const std::uint64_t partials_offset = offset;
  offset += align_up(partial_bytes);
  const std::uint64_t attention_offset = offset;
  offset += align_up(kQueryDimension * sizeof(__hip_bfloat16));
  const std::uint64_t gated_offset = offset;
  offset += align_up(kQueryDimension * sizeof(__hip_bfloat16));
  const std::uint64_t projected_offset = offset;
  offset += align_up(2048 * sizeof(__hip_bfloat16));

  const auto started = std::chrono::steady_clock::now();
  device_ = device;
  cache_capacity_ = cache_capacity;
  maximum_pv_splits_ = kMaximumPvSplits;
  try {
    check_hip(hipSetDevice(device_), "hipSetDevice native full attention");
    check_hip(hipMalloc(&allocation_, offset),
              "hipMalloc native full attention");
    allocation_bytes_ = offset;
    check_hip(hipMemset(allocation_, 0, offset),
              "hipMemset native full attention");
    auto* base = static_cast<unsigned char*>(allocation_);
    for (std::size_t slot = 0; slot < kFullLayers; ++slot) {
      k_caches_[slot] = base + k_offsets[slot];
      v_caches_[slot] = base + v_offsets[slot];
    }
    scores_ = base + scores_offset;
    probabilities_ = base + probabilities_offset;
    probabilities_bytes_ = score_bytes;
    softmax_exponentials_ = base + softmax_exponentials_offset;
    pv_partials_ = base + partials_offset;
    attention_output_ = base + attention_offset;
    gated_attention_ = base + gated_offset;
    projected_attention_ = base + projected_offset;
    qk_plan_ = std::make_unique<NativeFullAttentionQkPlan>(cache_capacity);
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native full attention");
    metrics.qk_workspace_bytes = qk_plan_->workspace_bytes;
    metrics.allocation_bytes = offset + metrics.qk_workspace_bytes;
    metrics.allocation_and_zero_ms = elapsed_ms(started);
    return metrics;
  } catch (...) {
    reset();
    throw;
  }
}

void* NativeFullAttentionState::k_cache(std::size_t layer_index) const {
  if (!built()) throw std::runtime_error("native full-attention state is not built");
  return k_caches_[layer_slot(layer_index)];
}

void* NativeFullAttentionState::v_cache(std::size_t layer_index) const {
  if (!built()) throw std::runtime_error("native full-attention state is not built");
  return v_caches_[layer_slot(layer_index)];
}

std::uint64_t NativeFullAttentionState::clear_request_scratch(
    void* stream_value) {
  if (!built() || probabilities_ == nullptr || probabilities_bytes_ == 0) {
    throw std::runtime_error(
        "native full-attention request scratch is not built");
  }
  check_hip(hipMemsetAsync(probabilities_, 0, probabilities_bytes_,
                           static_cast<hipStream_t>(stream_value)),
            "hipMemsetAsync native full-attention probabilities");
  return probabilities_bytes_;
}

void NativeFullAttentionState::launch_grouped_qk(
    const void* q, const void* k_cache, void* scores,
    void* stream_value) const {
  if (!qk_plan_) {
    throw std::runtime_error("native full-attention QK plan is not built");
  }
  qk_plan_->launch(q, k_cache, scores, stream_value);
}

void NativeFullAttentionState::launch_grouped_pv(
    const void* probabilities, const void* v_cache, void* attention,
    void* stream_value) const {
  if (!qk_plan_) {
    throw std::runtime_error("native full-attention PV plan is not built");
  }
  qk_plan_->launch_pv(probabilities, v_cache, attention, stream_value);
}

void NativeFullAttentionState::reset() noexcept {
  (void)hipSetDevice(device_);
  qk_plan_.reset();
  if (allocation_) (void)hipFree(allocation_);
  allocation_ = nullptr;
  allocation_bytes_ = 0;
  cache_capacity_ = 0;
  maximum_pv_splits_ = 0;
  probabilities_bytes_ = 0;
  k_caches_.fill(nullptr);
  v_caches_.fill(nullptr);
  scores_ = probabilities_ = softmax_exponentials_ = pv_partials_ = nullptr;
  attention_output_ = gated_attention_ = projected_attention_ = nullptr;
}

NativeFullAttentionCoreMetrics launch_native_grouped_full_attention(
    std::size_t layer_index, std::size_t position, std::size_t cache_end,
    const void* q, const void* normalized_k, const void* raw_v,
    NativeFullAttentionState& state, void* stream_value) {
  if (!state.built() || q == nullptr || normalized_k == nullptr ||
      raw_v == nullptr || cache_end == 0 || position >= cache_end ||
      cache_end > state.cache_capacity()) {
    throw std::invalid_argument("native grouped attention geometry is invalid");
  }
  hipStream_t stream = static_cast<hipStream_t>(stream_value);
  auto* k_cache = static_cast<__hip_bfloat16*>(state.k_cache(layer_index));
  auto* v_cache = static_cast<__hip_bfloat16*>(state.v_cache(layer_index));
  auto* scores = static_cast<__hip_bfloat16*>(state.scores());
  auto* probabilities =
      static_cast<__hip_bfloat16*>(state.probabilities());
  auto* attention =
      static_cast<__hip_bfloat16*>(state.attention_output());

  hipLaunchKernelGGL(
      write_kv_kernel, dim3(2), dim3(256), 0, stream,
      static_cast<const __hip_bfloat16*>(normalized_k),
      static_cast<const __hip_bfloat16*>(raw_v), k_cache, v_cache, position);
  check_hip(hipGetLastError(), "write_kv_kernel");
  state.launch_grouped_qk(q, k_cache, scores, stream);
  std::size_t attention_launches = 4;
  if (cache_end < kSplitSoftmaxMinimumTokens) {
    hipLaunchKernelGGL(
        scaled_softmax_bf16_kernel, dim3(kQueryHeads), dim3(256), 0,
        stream, scores, probabilities, cache_end, state.cache_capacity());
    check_hip(hipGetLastError(), "scaled_softmax_bf16_kernel");
  } else {
    const std::size_t split_count = std::min(
        kMaximumPvSplits,
        (cache_end + kSplitSoftmaxTokensPerBlock - 1) /
            kSplitSoftmaxTokensPerBlock);
    auto* partial_maxima = static_cast<float*>(state.pv_partials());
    auto* partial_sums =
        partial_maxima + kQueryHeads * kMaximumPvSplits;
    auto* head_maxima =
        partial_sums + kQueryHeads * kMaximumPvSplits;
    auto* head_inverse_sums = head_maxima + kQueryHeads;
    auto* exponentials =
        static_cast<float*>(state.softmax_exponentials());
    hipLaunchKernelGGL(
        split_softmax_partial_max_kernel,
        dim3(split_count, kQueryHeads), dim3(256), 0, stream,
        scores, partial_maxima, cache_end, state.cache_capacity(),
        split_count);
    check_hip(hipGetLastError(),
              "split_softmax_partial_max_kernel");
    hipLaunchKernelGGL(
        split_softmax_reduce_max_kernel, dim3(kQueryHeads), dim3(256), 0,
        stream, partial_maxima, head_maxima, split_count);
    check_hip(hipGetLastError(),
              "split_softmax_reduce_max_kernel");
    hipLaunchKernelGGL(
        split_softmax_partial_exp_sum_kernel,
        dim3(split_count, kQueryHeads), dim3(256), 0, stream,
        scores, head_maxima, exponentials, partial_sums, cache_end,
        state.cache_capacity(), split_count);
    check_hip(hipGetLastError(),
              "split_softmax_partial_exp_sum_kernel");
    hipLaunchKernelGGL(
        split_softmax_reduce_sum_kernel, dim3(kQueryHeads), dim3(256), 0,
        stream, partial_sums, head_inverse_sums, split_count);
    check_hip(hipGetLastError(),
              "split_softmax_reduce_sum_kernel");
    hipLaunchKernelGGL(
        split_softmax_normalize_kernel,
        dim3(split_count, kQueryHeads), dim3(256), 0, stream,
        exponentials, head_inverse_sums, probabilities, cache_end,
        state.cache_capacity(), split_count);
    check_hip(hipGetLastError(),
              "split_softmax_normalize_kernel");
    attention_launches += 4;
  }
  state.launch_grouped_pv(probabilities, v_cache, attention, stream);

  NativeFullAttentionCoreMetrics metrics;
  metrics.layer_index = layer_index;
  metrics.cache_end = cache_end;
  metrics.pv_splits = 1;
  metrics.native_kernel_launches = attention_launches;
  return metrics;
}

}  // namespace aima
