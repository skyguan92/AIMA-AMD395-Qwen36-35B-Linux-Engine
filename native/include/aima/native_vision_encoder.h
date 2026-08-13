// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <memory>

namespace aima {

class NativeWeightStore;

// Fixed-shape plan for the frozen Qwen3.6 Conv3d-as-linear patch projection.
// Input is the processor's contiguous BF16 [patches,1536] tensor and output is
// BF16 [patches,1152]. The checkpoint bias is fused before BF16 quantization.
class NativeVisionPatchEmbedPlan {
 public:
  NativeVisionPatchEmbedPlan(const NativeWeightStore& weights,
                             std::size_t patch_count);
  ~NativeVisionPatchEmbedPlan();
  NativeVisionPatchEmbedPlan(const NativeVisionPatchEmbedPlan&) = delete;
  NativeVisionPatchEmbedPlan& operator=(
      const NativeVisionPatchEmbedPlan&) = delete;
  NativeVisionPatchEmbedPlan(NativeVisionPatchEmbedPlan&&) noexcept;
  NativeVisionPatchEmbedPlan& operator=(
      NativeVisionPatchEmbedPlan&&) noexcept;

  void launch(const void* pixel_values_device, void* output_device,
              void* stream = nullptr) const;
  std::size_t patch_count() const;
  std::size_t workspace_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
