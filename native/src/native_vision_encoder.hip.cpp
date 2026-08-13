// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_encoder.h"

#include "aima/bf16_gemm.h"
#include "aima/native_weight_store.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace aima {
namespace {

constexpr std::size_t kPatchFeatures = 3 * 2 * 16 * 16;
constexpr std::size_t kVisionHidden = 1152;
constexpr std::size_t kWorkspaceLimit = 128ULL * 1024ULL * 1024ULL;

const NativeTensorView& require_tensor(const NativeWeightStore& weights,
                                       const char* name, std::uint8_t rank,
                                       std::uint64_t payload_bytes) {
  const NativeTensorView* tensor = weights.find(name);
  if (tensor == nullptr || tensor->device_pointer == nullptr ||
      tensor->rank != rank || tensor->payload_bytes != payload_bytes) {
    throw std::runtime_error(std::string("native vision weight mismatch: ") +
                             name);
  }
  return *tensor;
}

}  // namespace

struct NativeVisionPatchEmbedPlan::Impl {
  Impl(const NativeWeightStore& weights, std::size_t patches)
      : patch_count(patches),
        projection(
            require_tensor(weights,
                           "model.visual.patch_embed.proj.weight", 5,
                           kVisionHidden * kPatchFeatures *
                               sizeof(std::uint16_t))),
        bias(require_tensor(weights, "model.visual.patch_embed.proj.bias", 1,
                            kVisionHidden * sizeof(std::uint16_t))),
        gemm(patches, kVisionHidden, kPatchFeatures, kWorkspaceLimit, true,
             true) {
    if (patches == 0 || patches > 4 * 16384) {
      throw std::invalid_argument(
          "native vision patch count is outside the serving budget");
    }
    if (projection.shape !=
            std::array<std::uint32_t, 5>{1152, 3, 2, 16, 16} ||
        bias.shape[0] != kVisionHidden) {
      throw std::runtime_error("native vision patch weight shape is invalid");
    }
  }

  std::size_t patch_count = 0;
  const NativeTensorView& projection;
  const NativeTensorView& bias;
  Bf16GemmPlan gemm;
};

NativeVisionPatchEmbedPlan::NativeVisionPatchEmbedPlan(
    const NativeWeightStore& weights, std::size_t patch_count)
    : impl_(std::make_unique<Impl>(weights, patch_count)) {}
NativeVisionPatchEmbedPlan::~NativeVisionPatchEmbedPlan() = default;
NativeVisionPatchEmbedPlan::NativeVisionPatchEmbedPlan(
    NativeVisionPatchEmbedPlan&&) noexcept = default;
NativeVisionPatchEmbedPlan& NativeVisionPatchEmbedPlan::operator=(
    NativeVisionPatchEmbedPlan&&) noexcept = default;

void NativeVisionPatchEmbedPlan::launch(const void* pixel_values_device,
                                        void* output_device,
                                        void* stream) const {
  if (!impl_ || pixel_values_device == nullptr || output_device == nullptr) {
    throw std::invalid_argument("native vision patch launch is incomplete");
  }
  impl_->gemm.launch_with_bias(pixel_values_device,
                               impl_->projection.device_pointer,
                               impl_->bias.device_pointer, output_device,
                               stream);
}

std::size_t NativeVisionPatchEmbedPlan::patch_count() const {
  return impl_->patch_count;
}

std::size_t NativeVisionPatchEmbedPlan::workspace_bytes() const {
  return impl_->gemm.workspace_bytes();
}

}  // namespace aima
