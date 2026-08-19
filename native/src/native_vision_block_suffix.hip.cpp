// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_block_suffix.h"

#include "aima/bf16_gemm.h"
#include "aima/native_vl_processor.h"
#include "aima/native_weight_store.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace aima {
namespace {

constexpr std::size_t kVisionHidden = 1152;
constexpr std::size_t kVisionIntermediate = 4304;
constexpr std::size_t kVisionBlockCount = 27;
constexpr std::size_t kWorkspaceLimit = 128ULL * 1024ULL * 1024ULL;
constexpr float kVisionLayerNormEpsilon = 1.0e-6f;
constexpr float kInverseSqrtTwo = 0.70710678118654752f;

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
      patch_count > kNativeVlVisionBatchPatchLimit) {
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

__global__ void vision_block_residual_kernel(
    const __hip_bfloat16* left, const __hip_bfloat16* right,
    __hip_bfloat16* output, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = __float2bfloat16(__bfloat162float(left[index]) +
                                     __bfloat162float(right[index]));
  }
}

__global__ void vision_block_norm2_kernel(
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
    output[row * kVisionHidden + column] = __float2bfloat16(
        (value - mean) * inverse_standard_deviation *
            __bfloat162float(weight[column]) +
        __bfloat162float(bias[column]));
  }
}

__global__ void vision_exact_gelu_kernel(const __hip_bfloat16* input,
                                         __hip_bfloat16* output,
                                         std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float value = __bfloat162float(input[index]);
    const float activated =
        0.5f * value * (1.0f + erff(value * kInverseSqrtTwo));
    output[index] = __float2bfloat16(activated);
  }
}

}  // namespace

struct NativeVisionBlockSuffixPlan::Impl {
  Impl(const NativeWeightStore& weights, std::size_t requested_block_index,
       std::size_t patches,
       std::shared_ptr<Bf16GemmPlan> shared_attention_projection_gemm,
       std::shared_ptr<Bf16GemmPlan> shared_mlp_fc1_gemm,
       std::shared_ptr<Bf16GemmPlan> shared_mlp_fc2_gemm)
      : block_index_value(validate_block_index(requested_block_index)),
        patch_count_value(validate_patch_count(patches)),
        attention_projection_weight(require_tensor(
            weights,
            block_tensor_name(block_index_value, "attn.proj.weight"), 2,
            kVisionHidden * kVisionHidden * sizeof(std::uint16_t))),
        attention_projection_bias(require_tensor(
            weights, block_tensor_name(block_index_value, "attn.proj.bias"),
            1, kVisionHidden * sizeof(std::uint16_t))),
        norm2_weight(require_tensor(
            weights, block_tensor_name(block_index_value, "norm2.weight"), 1,
            kVisionHidden * sizeof(std::uint16_t))),
        norm2_bias(require_tensor(
            weights, block_tensor_name(block_index_value, "norm2.bias"), 1,
            kVisionHidden * sizeof(std::uint16_t))),
        mlp_fc1_weight(require_tensor(
            weights,
            block_tensor_name(block_index_value, "mlp.linear_fc1.weight"), 2,
            kVisionIntermediate * kVisionHidden * sizeof(std::uint16_t))),
        mlp_fc1_bias(require_tensor(
            weights,
            block_tensor_name(block_index_value, "mlp.linear_fc1.bias"), 1,
            kVisionIntermediate * sizeof(std::uint16_t))),
        mlp_fc2_weight(require_tensor(
            weights,
            block_tensor_name(block_index_value, "mlp.linear_fc2.weight"), 2,
            kVisionHidden * kVisionIntermediate * sizeof(std::uint16_t))),
        mlp_fc2_bias(require_tensor(
            weights,
            block_tensor_name(block_index_value, "mlp.linear_fc2.bias"), 1,
            kVisionHidden * sizeof(std::uint16_t))),
        attention_projection_gemm(
            std::move(shared_attention_projection_gemm)),
        mlp_fc1_gemm(std::move(shared_mlp_fc1_gemm)),
        mlp_fc2_gemm(std::move(shared_mlp_fc2_gemm)) {
    const std::array<std::uint32_t, 5> hidden_shape{
        static_cast<std::uint32_t>(kVisionHidden), 1, 1, 1, 1};
    const std::array<std::uint32_t, 5> intermediate_shape{
        static_cast<std::uint32_t>(kVisionIntermediate), 1, 1, 1, 1};
    const std::array<std::uint32_t, 5> hidden_weight_shape{
        static_cast<std::uint32_t>(kVisionHidden),
        static_cast<std::uint32_t>(kVisionHidden), 1, 1, 1};
    const std::array<std::uint32_t, 5> fc1_weight_shape{
        static_cast<std::uint32_t>(kVisionIntermediate),
        static_cast<std::uint32_t>(kVisionHidden), 1, 1, 1};
    const std::array<std::uint32_t, 5> fc2_weight_shape{
        static_cast<std::uint32_t>(kVisionHidden),
        static_cast<std::uint32_t>(kVisionIntermediate), 1, 1, 1};
    if (attention_projection_weight.shape != hidden_weight_shape ||
        attention_projection_bias.shape != hidden_shape ||
        norm2_weight.shape != hidden_shape || norm2_bias.shape != hidden_shape ||
        mlp_fc1_weight.shape != fc1_weight_shape ||
        mlp_fc1_bias.shape != intermediate_shape ||
        mlp_fc2_weight.shape != fc2_weight_shape ||
        mlp_fc2_bias.shape != hidden_shape ||
        !attention_projection_gemm || !mlp_fc1_gemm || !mlp_fc2_gemm ||
        attention_projection_gemm->m() != patch_count_value ||
        attention_projection_gemm->n() != kVisionHidden ||
        attention_projection_gemm->k() != kVisionHidden ||
        !attention_projection_gemm->bias_epilogue() ||
        mlp_fc1_gemm->m() != patch_count_value ||
        mlp_fc1_gemm->n() != kVisionIntermediate ||
        mlp_fc1_gemm->k() != kVisionHidden ||
        !mlp_fc1_gemm->bias_epilogue() ||
        mlp_fc2_gemm->m() != patch_count_value ||
        mlp_fc2_gemm->n() != kVisionHidden ||
        mlp_fc2_gemm->k() != kVisionIntermediate ||
        !mlp_fc2_gemm->bias_epilogue()) {
      throw std::runtime_error(
          "native vision block suffix weight shape is invalid");
    }
  }

  std::size_t block_index_value = 0;
  std::size_t patch_count_value = 0;
  const NativeTensorView& attention_projection_weight;
  const NativeTensorView& attention_projection_bias;
  const NativeTensorView& norm2_weight;
  const NativeTensorView& norm2_bias;
  const NativeTensorView& mlp_fc1_weight;
  const NativeTensorView& mlp_fc1_bias;
  const NativeTensorView& mlp_fc2_weight;
  const NativeTensorView& mlp_fc2_bias;
  std::shared_ptr<Bf16GemmPlan> attention_projection_gemm;
  std::shared_ptr<Bf16GemmPlan> mlp_fc1_gemm;
  std::shared_ptr<Bf16GemmPlan> mlp_fc2_gemm;
};

NativeVisionBlockSuffixPlan::NativeVisionBlockSuffixPlan(
    const NativeWeightStore& weights, std::size_t block_index,
    std::size_t patch_count)
    : NativeVisionBlockSuffixPlan(
          weights, block_index, patch_count,
          std::make_shared<Bf16GemmPlan>(
              patch_count, kVisionHidden, kVisionHidden, kWorkspaceLimit,
              true, true),
          std::make_shared<Bf16GemmPlan>(
              patch_count, kVisionIntermediate, kVisionHidden,
              kWorkspaceLimit, true, true),
          std::make_shared<Bf16GemmPlan>(
              patch_count, kVisionHidden, kVisionIntermediate,
              kWorkspaceLimit, true, true)) {}

NativeVisionBlockSuffixPlan::NativeVisionBlockSuffixPlan(
    const NativeWeightStore& weights, std::size_t block_index,
    std::size_t patch_count,
    std::shared_ptr<Bf16GemmPlan> attention_projection_gemm,
    std::shared_ptr<Bf16GemmPlan> mlp_fc1_gemm,
    std::shared_ptr<Bf16GemmPlan> mlp_fc2_gemm)
    : impl_(std::make_unique<Impl>(
          weights, block_index, patch_count,
          std::move(attention_projection_gemm), std::move(mlp_fc1_gemm),
          std::move(mlp_fc2_gemm))) {}
NativeVisionBlockSuffixPlan::~NativeVisionBlockSuffixPlan() = default;
NativeVisionBlockSuffixPlan::NativeVisionBlockSuffixPlan(
    NativeVisionBlockSuffixPlan&&) noexcept = default;
NativeVisionBlockSuffixPlan& NativeVisionBlockSuffixPlan::operator=(
    NativeVisionBlockSuffixPlan&&) noexcept = default;

void NativeVisionBlockSuffixPlan::launch(
    const void* block_input_device, const void* attention_device,
    void* attention_projection_device, void* attention_residual_device,
    void* norm2_device, void* mlp_fc1_device, void* mlp_activation_device,
    void* mlp_fc2_device, void* block_output_device,
    void* stream_pointer) const {
  const std::array<const void*, 9> tensors{
      block_input_device,         attention_device,
      attention_projection_device, attention_residual_device,
      norm2_device,               mlp_fc1_device,
      mlp_activation_device,      mlp_fc2_device,
      block_output_device};
  for (std::size_t left = 0; left < tensors.size(); ++left) {
    if (!impl_ || tensors[left] == nullptr) {
      throw std::invalid_argument("native vision block suffix launch is invalid");
    }
    for (std::size_t right = left + 1; right < tensors.size(); ++right) {
      if (tensors[left] == tensors[right]) {
        throw std::invalid_argument(
            "native vision block suffix tensors must not alias");
      }
    }
  }
  launch_attention_projection(attention_device, attention_projection_device,
                              stream_pointer);
  launch_residual(block_input_device, attention_projection_device,
                  attention_residual_device, stream_pointer);
  launch_norm2(attention_residual_device, norm2_device, stream_pointer);
  launch_mlp_fc1(norm2_device, mlp_fc1_device, stream_pointer);
  launch_gelu(mlp_fc1_device, mlp_activation_device, stream_pointer);
  launch_mlp_fc2(mlp_activation_device, mlp_fc2_device, stream_pointer);
  launch_residual(attention_residual_device, mlp_fc2_device,
                  block_output_device, stream_pointer);
}

void NativeVisionBlockSuffixPlan::launch_attention_projection(
    const void* attention_device, void* output_device,
    void* stream_pointer) const {
  if (!impl_ || attention_device == nullptr || output_device == nullptr ||
      attention_device == output_device) {
    throw std::invalid_argument(
        "native vision attention projection launch is invalid");
  }
  impl_->attention_projection_gemm->launch_with_bias(
      attention_device, impl_->attention_projection_weight.device_pointer,
      impl_->attention_projection_bias.device_pointer, output_device,
      stream_pointer);
}

void NativeVisionBlockSuffixPlan::launch_residual(
    const void* left_device, const void* right_device, void* output_device,
    void* stream_pointer) const {
  if (!impl_ || left_device == nullptr || right_device == nullptr ||
      output_device == nullptr || left_device == output_device ||
      right_device == output_device) {
    throw std::invalid_argument("native vision residual launch is invalid");
  }
  constexpr std::size_t kThreads = 256;
  const std::size_t elements = impl_->patch_count_value * kVisionHidden;
  const std::size_t blocks = (elements + kThreads - 1) / kThreads;
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
  hipLaunchKernelGGL(
      vision_block_residual_kernel, dim3(blocks), dim3(kThreads), 0, stream,
      static_cast<const __hip_bfloat16*>(left_device),
      static_cast<const __hip_bfloat16*>(right_device),
      static_cast<__hip_bfloat16*>(output_device), elements);
  check_hip(hipGetLastError(), "native vision residual launch");
}

void NativeVisionBlockSuffixPlan::launch_norm2(const void* input_device,
                                               void* output_device,
                                               void* stream_pointer) const {
  if (!impl_ || input_device == nullptr || output_device == nullptr ||
      input_device == output_device) {
    throw std::invalid_argument("native vision norm2 launch is invalid");
  }
  constexpr std::size_t kThreads = 256;
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
  hipLaunchKernelGGL(
      vision_block_norm2_kernel, dim3(impl_->patch_count_value),
      dim3(kThreads), 0, stream,
      static_cast<const __hip_bfloat16*>(input_device),
      static_cast<const __hip_bfloat16*>(impl_->norm2_weight.device_pointer),
      static_cast<const __hip_bfloat16*>(impl_->norm2_bias.device_pointer),
      static_cast<__hip_bfloat16*>(output_device));
  check_hip(hipGetLastError(), "native vision norm2 launch");
}

void NativeVisionBlockSuffixPlan::launch_mlp_fc1(
    const void* input_device, void* output_device,
    void* stream_pointer) const {
  if (!impl_ || input_device == nullptr || output_device == nullptr ||
      input_device == output_device) {
    throw std::invalid_argument("native vision MLP FC1 launch is invalid");
  }
  impl_->mlp_fc1_gemm->launch_with_bias(
      input_device, impl_->mlp_fc1_weight.device_pointer,
      impl_->mlp_fc1_bias.device_pointer, output_device, stream_pointer);
}

void NativeVisionBlockSuffixPlan::launch_gelu(const void* input_device,
                                              void* output_device,
                                              void* stream_pointer) const {
  if (!impl_ || input_device == nullptr || output_device == nullptr ||
      input_device == output_device) {
    throw std::invalid_argument("native vision GELU launch is invalid");
  }
  constexpr std::size_t kThreads = 256;
  const std::size_t elements =
      impl_->patch_count_value * kVisionIntermediate;
  const std::size_t blocks = (elements + kThreads - 1) / kThreads;
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
  hipLaunchKernelGGL(
      vision_exact_gelu_kernel, dim3(blocks), dim3(kThreads), 0, stream,
      static_cast<const __hip_bfloat16*>(input_device),
      static_cast<__hip_bfloat16*>(output_device), elements);
  check_hip(hipGetLastError(), "native vision exact GELU launch");
}

void NativeVisionBlockSuffixPlan::launch_mlp_fc2(
    const void* input_device, void* output_device,
    void* stream_pointer) const {
  if (!impl_ || input_device == nullptr || output_device == nullptr ||
      input_device == output_device) {
    throw std::invalid_argument("native vision MLP FC2 launch is invalid");
  }
  impl_->mlp_fc2_gemm->launch_with_bias(
      input_device, impl_->mlp_fc2_weight.device_pointer,
      impl_->mlp_fc2_bias.device_pointer, output_device, stream_pointer);
}

std::size_t NativeVisionBlockSuffixPlan::block_index() const {
  return impl_->block_index_value;
}

std::size_t NativeVisionBlockSuffixPlan::patch_count() const {
  return impl_->patch_count_value;
}

std::size_t NativeVisionBlockSuffixPlan::workspace_bytes() const {
  return impl_->attention_projection_gemm->workspace_bytes() +
         impl_->mlp_fc1_gemm->workspace_bytes() +
         impl_->mlp_fc2_gemm->workspace_bytes();
}

}  // namespace aima
