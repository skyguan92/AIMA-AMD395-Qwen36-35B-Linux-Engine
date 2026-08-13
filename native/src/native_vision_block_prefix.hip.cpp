// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_encoder.h"

#include "aima/bf16_gemm.h"
#include "aima/native_weight_store.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace aima {
namespace {

constexpr std::size_t kVisionHidden = 1152;
constexpr std::size_t kVisionQkvHidden = 3 * kVisionHidden;
constexpr std::size_t kVisionBlockCount = 27;
constexpr std::size_t kWorkspaceLimit = 128ULL * 1024ULL * 1024ULL;
constexpr float kVisionLayerNormEpsilon = 1.0e-6f;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

std::size_t validate_block_index(std::size_t block_index) {
  if (block_index >= kVisionBlockCount) {
    throw std::invalid_argument("native vision block index is invalid");
  }
  return block_index;
}

std::size_t validate_patch_count(std::size_t patch_count) {
  if (patch_count == 0 ||
      patch_count > 4 * kNativeVlAggregateTokenLimit) {
    throw std::invalid_argument(
        "native vision block patch count is outside the serving budget");
  }
  return patch_count;
}

const NativeTensorView& require_tensor(const NativeWeightStore& weights,
                                       std::string_view name,
                                       std::uint8_t rank,
                                       std::uint64_t payload_bytes) {
  const NativeTensorView* tensor = weights.find(name);
  if (tensor == nullptr || tensor->device_pointer == nullptr ||
      tensor->rank != rank || tensor->payload_bytes != payload_bytes) {
    throw std::runtime_error(std::string("native vision weight mismatch: ") +
                             std::string(name));
  }
  return *tensor;
}

std::string block_tensor_name(std::size_t block_index,
                              std::string_view suffix) {
  return "model.visual.blocks." + std::to_string(block_index) + "." +
         std::string(suffix);
}

struct WelfordState {
  float mean;
  float m2;
  std::uint32_t count;
};

__device__ WelfordState welford_add(WelfordState state, float value) {
  ++state.count;
  const float delta = value - state.mean;
  state.mean += delta / static_cast<float>(state.count);
  const float delta2 = value - state.mean;
  state.m2 += delta * delta2;
  return state;
}

__device__ WelfordState welford_merge(WelfordState left,
                                      WelfordState right) {
  if (left.count == 0) return right;
  if (right.count == 0) return left;
  const float delta = right.mean - left.mean;
  const std::uint32_t count = left.count + right.count;
  const float right_fraction =
      static_cast<float>(right.count) / static_cast<float>(count);
  const float cross_count =
      static_cast<float>(left.count) * static_cast<float>(right.count) /
      static_cast<float>(count);
  return WelfordState{left.mean + delta * right_fraction,
                      left.m2 + right.m2 + delta * delta * cross_count,
                      count};
}

__global__ void vision_layer_norm_kernel(
    const __hip_bfloat16* input, const __hip_bfloat16* weight,
    const __hip_bfloat16* bias, __hip_bfloat16* output) {
  const std::size_t row = blockIdx.x;
  WelfordState local{0.0f, 0.0f, 0};
  for (std::size_t column = threadIdx.x; column < kVisionHidden;
       column += blockDim.x) {
    local = welford_add(
        local, __bfloat162float(input[row * kVisionHidden + column]));
  }

  __shared__ float means[256];
  __shared__ float m2s[256];
  __shared__ std::uint32_t counts[256];
  means[threadIdx.x] = local.mean;
  m2s[threadIdx.x] = local.m2;
  counts[threadIdx.x] = local.count;
  __syncthreads();
  for (std::size_t offset = blockDim.x / 2; offset != 0; offset /= 2) {
    if (threadIdx.x < offset) {
      const WelfordState merged = welford_merge(
          WelfordState{means[threadIdx.x], m2s[threadIdx.x],
                       counts[threadIdx.x]},
          WelfordState{means[threadIdx.x + offset],
                       m2s[threadIdx.x + offset],
                       counts[threadIdx.x + offset]});
      means[threadIdx.x] = merged.mean;
      m2s[threadIdx.x] = merged.m2;
      counts[threadIdx.x] = merged.count;
    }
    __syncthreads();
  }

  const float mean = means[0];
  const float inverse_standard_deviation =
      rsqrtf(m2s[0] / static_cast<float>(kVisionHidden) +
             kVisionLayerNormEpsilon);
  for (std::size_t column = threadIdx.x; column < kVisionHidden;
       column += blockDim.x) {
    const float value =
        __bfloat162float(input[row * kVisionHidden + column]);
    const float normalized = (value - mean) * inverse_standard_deviation;
    output[row * kVisionHidden + column] = __float2bfloat16(
        normalized * __bfloat162float(weight[column]) +
        __bfloat162float(bias[column]));
  }
}

}  // namespace

struct NativeVisionBlockPrefixPlan::Impl {
  Impl(const NativeWeightStore& weights, std::size_t requested_block_index,
       std::size_t patches)
      : block_index_value(validate_block_index(requested_block_index)),
        patch_count_value(validate_patch_count(patches)),
        norm1_weight(require_tensor(
            weights, block_tensor_name(block_index_value, "norm1.weight"), 1,
            kVisionHidden * sizeof(std::uint16_t))),
        norm1_bias(require_tensor(
            weights, block_tensor_name(block_index_value, "norm1.bias"), 1,
            kVisionHidden * sizeof(std::uint16_t))),
        qkv_weight(require_tensor(
            weights, block_tensor_name(block_index_value, "attn.qkv.weight"),
            2, kVisionQkvHidden * kVisionHidden * sizeof(std::uint16_t))),
        qkv_bias(require_tensor(
            weights, block_tensor_name(block_index_value, "attn.qkv.bias"), 1,
            kVisionQkvHidden * sizeof(std::uint16_t))),
        qkv_gemm(patch_count_value, kVisionQkvHidden, kVisionHidden,
                 kWorkspaceLimit, true, true) {
    const std::array<std::uint32_t, 5> affine_shape{
        static_cast<std::uint32_t>(kVisionHidden), 1, 1, 1, 1};
    const std::array<std::uint32_t, 5> qkv_weight_shape{
        static_cast<std::uint32_t>(kVisionQkvHidden),
        static_cast<std::uint32_t>(kVisionHidden), 1, 1, 1};
    const std::array<std::uint32_t, 5> qkv_bias_shape{
        static_cast<std::uint32_t>(kVisionQkvHidden), 1, 1, 1, 1};
    if (norm1_weight.shape != affine_shape || norm1_bias.shape != affine_shape ||
        qkv_weight.shape != qkv_weight_shape ||
        qkv_bias.shape != qkv_bias_shape) {
      throw std::runtime_error(
          "native vision block prefix weight shape is invalid");
    }
  }

  std::size_t block_index_value = 0;
  std::size_t patch_count_value = 0;
  const NativeTensorView& norm1_weight;
  const NativeTensorView& norm1_bias;
  const NativeTensorView& qkv_weight;
  const NativeTensorView& qkv_bias;
  Bf16GemmPlan qkv_gemm;
};

NativeVisionBlockPrefixPlan::NativeVisionBlockPrefixPlan(
    const NativeWeightStore& weights, std::size_t block_index,
    std::size_t patch_count)
    : impl_(std::make_unique<Impl>(weights, block_index, patch_count)) {}
NativeVisionBlockPrefixPlan::~NativeVisionBlockPrefixPlan() = default;
NativeVisionBlockPrefixPlan::NativeVisionBlockPrefixPlan(
    NativeVisionBlockPrefixPlan&&) noexcept = default;
NativeVisionBlockPrefixPlan& NativeVisionBlockPrefixPlan::operator=(
    NativeVisionBlockPrefixPlan&&) noexcept = default;

void NativeVisionBlockPrefixPlan::launch(const void* input_device,
                                         void* norm1_output_device,
                                         void* qkv_output_device,
                                         void* stream_pointer) const {
  if (!impl_ || input_device == nullptr || norm1_output_device == nullptr ||
      qkv_output_device == nullptr || input_device == norm1_output_device ||
      input_device == qkv_output_device ||
      norm1_output_device == qkv_output_device) {
    throw std::invalid_argument("native vision block prefix launch is invalid");
  }
  constexpr std::size_t kThreads = 256;
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
  hipLaunchKernelGGL(
      vision_layer_norm_kernel, dim3(impl_->patch_count_value),
      dim3(kThreads), 0, stream,
      static_cast<const __hip_bfloat16*>(input_device),
      static_cast<const __hip_bfloat16*>(impl_->norm1_weight.device_pointer),
      static_cast<const __hip_bfloat16*>(impl_->norm1_bias.device_pointer),
      static_cast<__hip_bfloat16*>(norm1_output_device));
  check_hip(hipGetLastError(), "native vision layer norm kernel launch");
  impl_->qkv_gemm.launch_with_bias(
      norm1_output_device, impl_->qkv_weight.device_pointer,
      impl_->qkv_bias.device_pointer, qkv_output_device, stream_pointer);
}

std::size_t NativeVisionBlockPrefixPlan::block_index() const {
  return impl_->block_index_value;
}

std::size_t NativeVisionBlockPrefixPlan::patch_count() const {
  return impl_->patch_count_value;
}

std::size_t NativeVisionBlockPrefixPlan::workspace_bytes() const {
  return impl_->qkv_gemm.workspace_bytes();
}

}  // namespace aima
