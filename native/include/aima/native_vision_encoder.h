// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aima/native_vl_processor.h"

#include <cstddef>
#include <memory>
#include <vector>

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

// Interpolates the frozen 48x48 BF16 position table into Qwen's 2x2 spatial
// merge order. Multiple media grids are concatenated in request order; the
// temporal dimension repeats each spatial table exactly as the reference does.
class NativeVisionPositionPlan {
 public:
  NativeVisionPositionPlan(const NativeWeightStore& weights,
                           const std::vector<NativeVlGrid>& grids);
  ~NativeVisionPositionPlan();
  NativeVisionPositionPlan(const NativeVisionPositionPlan&) = delete;
  NativeVisionPositionPlan& operator=(const NativeVisionPositionPlan&) = delete;
  NativeVisionPositionPlan(NativeVisionPositionPlan&&) noexcept;
  NativeVisionPositionPlan& operator=(NativeVisionPositionPlan&&) noexcept;

  // Writes BF16 [patch_count,1152] interpolated positions.
  void launch(void* output_device, void* stream = nullptr) const;
  // Writes BF16 patch_embeddings + interpolated_positions in the same shape.
  void launch_add(const void* patch_embeddings_device, void* output_device,
                  void* stream = nullptr) const;
  std::size_t patch_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Executes the first independently qualified prefix of one frozen Qwen3.6
// vision transformer block: BF16 LayerNorm followed by the biased QKV linear
// projection. Attention and the remainder of the block are intentionally not
// part of this plan until their own oracle boundaries are qualified.
class NativeVisionBlockPrefixPlan {
 public:
  NativeVisionBlockPrefixPlan(const NativeWeightStore& weights,
                              std::size_t block_index,
                              std::size_t patch_count);
  ~NativeVisionBlockPrefixPlan();
  NativeVisionBlockPrefixPlan(const NativeVisionBlockPrefixPlan&) = delete;
  NativeVisionBlockPrefixPlan& operator=(
      const NativeVisionBlockPrefixPlan&) = delete;
  NativeVisionBlockPrefixPlan(NativeVisionBlockPrefixPlan&&) noexcept;
  NativeVisionBlockPrefixPlan& operator=(
      NativeVisionBlockPrefixPlan&&) noexcept;

  // All tensors are contiguous BF16. Input and norm1 are [patch_count,1152];
  // qkv is [patch_count,3456]. The outputs must not alias the input.
  void launch(const void* input_device, void* norm1_output_device,
              void* qkv_output_device, void* stream = nullptr) const;
  std::size_t block_index() const;
  std::size_t patch_count() const;
  std::size_t workspace_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
