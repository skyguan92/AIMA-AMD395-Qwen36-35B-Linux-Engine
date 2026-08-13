// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_merger.h"

#include "aima/bf16_gemm.h"
#include "aima/native_vision_exact_layer_norm.h"
#include "aima/native_vl_processor.h"
#include "aima/native_weight_store.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace aima {
namespace {

constexpr std::size_t kVisionHidden = 1152;
constexpr std::size_t kSpatialMergeArea = 4;
constexpr std::size_t kMergerHidden = kVisionHidden * kSpatialMergeArea;
constexpr std::size_t kLanguageHidden = 2048;
constexpr std::size_t kWorkspaceLimit = 128ULL * 1024ULL * 1024ULL;
constexpr float kInverseSqrtTwo = 0.70710678118654752440f;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

std::size_t validate_patch_count(std::size_t patch_count) {
  if (patch_count == 0 || patch_count % kSpatialMergeArea != 0 ||
      patch_count > kSpatialMergeArea * kNativeVlAggregateTokenLimit) {
    throw std::invalid_argument(
        "native vision merger patch count is outside the serving budget");
  }
  return patch_count;
}

std::size_t checked_multiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > static_cast<std::size_t>(-1) / left) {
    throw std::invalid_argument("native vision merger workspace overflows");
  }
  return left * right;
}

const NativeTensorView& require_tensor(const NativeWeightStore& weights,
                                       std::string_view name,
                                       std::uint8_t rank,
                                       std::uint64_t payload_bytes) {
  const NativeTensorView* tensor = weights.find(name);
  if (tensor == nullptr || tensor->device_pointer == nullptr ||
      tensor->rank != rank || tensor->payload_bytes != payload_bytes) {
    throw std::runtime_error(std::string("native merger weight mismatch: ") +
                             std::string(name));
  }
  return *tensor;
}

__global__ void vision_merger_exact_gelu_kernel(
    const __hip_bfloat16* input, __hip_bfloat16* output,
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

struct NativeVisionMergerPlan::Impl {
  Impl(const NativeWeightStore& weights, std::size_t patches)
      : patch_count_value(validate_patch_count(patches)),
        merged_token_count_value(patch_count_value / kSpatialMergeArea),
        arena_bytes(checked_multiply(
            checked_multiply(patch_count_value, kVisionHidden),
            sizeof(std::uint16_t))),
        temporary_bytes_value(checked_multiply(2, arena_bytes)),
        norm_weight(require_tensor(
            weights, "model.visual.merger.norm.weight", 1,
            kVisionHidden * sizeof(std::uint16_t))),
        norm_bias(require_tensor(
            weights, "model.visual.merger.norm.bias", 1,
            kVisionHidden * sizeof(std::uint16_t))),
        fc1_weight(require_tensor(
            weights, "model.visual.merger.linear_fc1.weight", 2,
            kMergerHidden * kMergerHidden * sizeof(std::uint16_t))),
        fc1_bias(require_tensor(
            weights, "model.visual.merger.linear_fc1.bias", 1,
            kMergerHidden * sizeof(std::uint16_t))),
        fc2_weight(require_tensor(
            weights, "model.visual.merger.linear_fc2.weight", 2,
            kLanguageHidden * kMergerHidden * sizeof(std::uint16_t))),
        fc2_bias(require_tensor(
            weights, "model.visual.merger.linear_fc2.bias", 1,
            kLanguageHidden * sizeof(std::uint16_t))),
        norm(patch_count_value,
             NativeVisionLayerNormReciprocal::kFastAmdReciprocal),
        fc1(merged_token_count_value, kMergerHidden, kMergerHidden,
            kWorkspaceLimit, true, true),
        fc2(merged_token_count_value, kLanguageHidden, kMergerHidden,
            kWorkspaceLimit, true, true) {}

  std::size_t patch_count_value = 0;
  std::size_t merged_token_count_value = 0;
  std::size_t arena_bytes = 0;
  std::size_t temporary_bytes_value = 0;
  const NativeTensorView& norm_weight;
  const NativeTensorView& norm_bias;
  const NativeTensorView& fc1_weight;
  const NativeTensorView& fc1_bias;
  const NativeTensorView& fc2_weight;
  const NativeTensorView& fc2_bias;
  NativeVisionExactLayerNormPlan norm;
  Bf16GemmPlan fc1;
  Bf16GemmPlan fc2;
};

NativeVisionMergerPlan::NativeVisionMergerPlan(
    const NativeWeightStore& weights, std::size_t patch_count)
    : impl_(std::make_unique<Impl>(weights, patch_count)) {}
NativeVisionMergerPlan::~NativeVisionMergerPlan() = default;
NativeVisionMergerPlan::NativeVisionMergerPlan(
    NativeVisionMergerPlan&&) noexcept = default;
NativeVisionMergerPlan& NativeVisionMergerPlan::operator=(
    NativeVisionMergerPlan&&) noexcept = default;

void NativeVisionMergerPlan::launch(
    const void* input_device, void* output_device, void* temporary_device,
    std::size_t supplied_temporary_bytes, void* stream) const {
  if (!impl_ || input_device == nullptr || output_device == nullptr ||
      temporary_device == nullptr || input_device == output_device ||
      input_device == temporary_device || output_device == temporary_device ||
      supplied_temporary_bytes < impl_->temporary_bytes_value) {
    throw std::invalid_argument("native vision merger launch is invalid");
  }
  auto* arena_a = static_cast<unsigned char*>(temporary_device);
  auto* arena_b = arena_a + impl_->arena_bytes;
  launch_norm(input_device, arena_a, stream);
  launch_fc1(arena_a, arena_b, stream);
  launch_gelu(arena_b, arena_a, stream);
  launch_fc2(arena_a, output_device, stream);
}

void NativeVisionMergerPlan::launch_norm(
    const void* input_device, void* normalized_device, void* stream) const {
  if (!impl_ || input_device == nullptr || normalized_device == nullptr ||
      input_device == normalized_device) {
    throw std::invalid_argument("native vision merger norm launch is invalid");
  }
  impl_->norm.launch(input_device, impl_->norm_weight.device_pointer,
                     impl_->norm_bias.device_pointer, normalized_device,
                     stream);
}

void NativeVisionMergerPlan::launch_fc1(
    const void* normalized_device, void* output_device, void* stream) const {
  if (!impl_ || normalized_device == nullptr || output_device == nullptr ||
      normalized_device == output_device) {
    throw std::invalid_argument("native vision merger FC1 launch is invalid");
  }
  impl_->fc1.launch_with_bias(
      normalized_device, impl_->fc1_weight.device_pointer,
      impl_->fc1_bias.device_pointer, output_device, stream);
}

void NativeVisionMergerPlan::launch_gelu(
    const void* input_device, void* output_device, void* stream_pointer) const {
  if (!impl_ || input_device == nullptr || output_device == nullptr ||
      input_device == output_device) {
    throw std::invalid_argument("native vision merger GELU launch is invalid");
  }
  constexpr std::size_t kThreads = 256;
  const std::size_t elements =
      impl_->merged_token_count_value * kMergerHidden;
  const std::size_t blocks = (elements + kThreads - 1) / kThreads;
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
  hipLaunchKernelGGL(
      vision_merger_exact_gelu_kernel, dim3(blocks), dim3(kThreads), 0,
      stream, static_cast<const __hip_bfloat16*>(input_device),
      static_cast<__hip_bfloat16*>(output_device), elements);
  check_hip(hipGetLastError(), "native vision merger GELU launch");
}

void NativeVisionMergerPlan::launch_fc2(
    const void* activated_device, void* output_device, void* stream) const {
  if (!impl_ || activated_device == nullptr || output_device == nullptr ||
      activated_device == output_device) {
    throw std::invalid_argument("native vision merger FC2 launch is invalid");
  }
  impl_->fc2.launch_with_bias(
      activated_device, impl_->fc2_weight.device_pointer,
      impl_->fc2_bias.device_pointer, output_device, stream);
}

std::size_t NativeVisionMergerPlan::patch_count() const {
  return impl_->patch_count_value;
}

std::size_t NativeVisionMergerPlan::merged_token_count() const {
  return impl_->merged_token_count_value;
}

std::size_t NativeVisionMergerPlan::temporary_bytes() const {
  return impl_->temporary_bytes_value;
}

std::size_t NativeVisionMergerPlan::library_workspace_bytes() const {
  return impl_->fc1.workspace_bytes() + impl_->fc2.workspace_bytes();
}

}  // namespace aima
