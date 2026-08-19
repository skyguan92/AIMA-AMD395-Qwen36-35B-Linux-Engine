// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aima/native_vl_processor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace aima {

class NativeVisionAotBlockGemmPlans;
class NativeVisionAotAttentionPlan;
class NativeVisionMergerPlan;
class NativeVisionPatchEmbedPlan;
class NativeWeightStore;

// Resident request-shape metadata shared by all 27 visual blocks. The
// rotary cache follows vLLM's BF16, Neox-style two-axis layout exactly;
// cu_seqlens contains one independent attention segment per temporal frame.
class NativeVisionEncoderMetadataPlan {
 public:
  explicit NativeVisionEncoderMetadataPlan(
      const std::vector<NativeVlGrid>& grids);
  ~NativeVisionEncoderMetadataPlan();
  NativeVisionEncoderMetadataPlan(
      const NativeVisionEncoderMetadataPlan&) = delete;
  NativeVisionEncoderMetadataPlan& operator=(
      const NativeVisionEncoderMetadataPlan&) = delete;
  NativeVisionEncoderMetadataPlan(
      NativeVisionEncoderMetadataPlan&&) noexcept;
  NativeVisionEncoderMetadataPlan& operator=(
      NativeVisionEncoderMetadataPlan&&) noexcept;

  const void* rotary_cos_device() const;
  const void* rotary_sin_device() const;
  const std::vector<std::uint32_t>& cu_seqlens() const;
  const std::string& rotary_cos_sha256() const;
  const std::string& rotary_sin_sha256() const;
  std::size_t patch_count() const;
  std::size_t resident_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Complete Qwen3.6 visual tower for one fixed request shape: processor-native
// pixels -> patch projection + interpolated positions -> 27 resident blocks
// -> four-patch merger into the 2048-wide language embedding space.
class NativeVisionPipelinePlan {
 public:
  NativeVisionPipelinePlan(
      const NativeWeightStore& weights,
      const std::filesystem::path& attention_image_path,
      const std::vector<NativeVlGrid>& grids);
  NativeVisionPipelinePlan(
      const NativeWeightStore& weights,
      const std::filesystem::path& attention_image_path,
      const std::vector<NativeVlGrid>& grids,
      std::shared_ptr<NativeVisionPatchEmbedPlan> patch_plan,
      std::shared_ptr<const NativeVisionAotAttentionPlan> attention_plan,
      std::shared_ptr<NativeVisionAotBlockGemmPlans> block_gemm_plans,
      std::shared_ptr<NativeVisionMergerPlan> merger_plan);
  ~NativeVisionPipelinePlan();
  NativeVisionPipelinePlan(const NativeVisionPipelinePlan&) = delete;
  NativeVisionPipelinePlan& operator=(const NativeVisionPipelinePlan&) = delete;
  NativeVisionPipelinePlan(NativeVisionPipelinePlan&&) noexcept;
  NativeVisionPipelinePlan& operator=(NativeVisionPipelinePlan&&) noexcept;

  // pixel_values is BF16 [patch_count,1536], output is BF16
  // [merged_token_count,2048]. All three buffers must be distinct.
  void launch(const void* pixel_values_device, void* output_device,
              void* temporary_device, std::size_t temporary_bytes,
              void* stream = nullptr) const;

  // Qualification boundary with the same patch/position/metadata path as
  // launch(). output_device is BF16 [patch_count,1152].
  void launch_encoder_through(
      std::size_t last_block_index, const void* pixel_values_device,
      void* output_device, void* temporary_device,
      std::size_t temporary_bytes, void* stream = nullptr) const;

  std::size_t patch_count() const;
  std::size_t merged_token_count() const;
  std::size_t temporary_bytes() const;
  std::size_t metadata_resident_bytes() const;
  std::size_t library_workspace_bytes() const;
  std::shared_ptr<NativeVisionPatchEmbedPlan> patch_plan() const;
  std::shared_ptr<const NativeVisionAotAttentionPlan> attention_plan() const;
  std::shared_ptr<NativeVisionAotBlockGemmPlans> block_gemm_plans() const;
  std::shared_ptr<NativeVisionMergerPlan> merger_plan() const;
  const std::vector<std::uint32_t>& cu_seqlens() const;
  const std::string& rotary_cos_sha256() const;
  const std::string& rotary_sin_sha256() const;
  const std::string& attention_image_sha256() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
