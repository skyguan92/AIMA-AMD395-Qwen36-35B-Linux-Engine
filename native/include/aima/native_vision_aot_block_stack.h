// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace aima {

class NativeVisionAotBlockGemmPlans;
class NativeVisionAotAttentionPlan;
class NativeWeightStore;

// Sequential 27-block vision stack using one shared, hash-locked AOT attention
// module and one shared request arena.
class NativeVisionAotBlockStackPlan {
 public:
  NativeVisionAotBlockStackPlan(
      const NativeWeightStore& weights,
      const std::filesystem::path& attention_image_path,
      std::size_t patch_count,
      const std::vector<std::uint32_t>& cu_seqlens);
  NativeVisionAotBlockStackPlan(
      const NativeWeightStore& weights,
      const std::filesystem::path& attention_image_path,
      std::size_t patch_count,
      const std::vector<std::uint32_t>& cu_seqlens,
      std::shared_ptr<const NativeVisionAotAttentionPlan> attention,
      std::shared_ptr<NativeVisionAotBlockGemmPlans> gemm_plans);
  ~NativeVisionAotBlockStackPlan();
  NativeVisionAotBlockStackPlan(const NativeVisionAotBlockStackPlan&) = delete;
  NativeVisionAotBlockStackPlan& operator=(
      const NativeVisionAotBlockStackPlan&) = delete;
  NativeVisionAotBlockStackPlan(NativeVisionAotBlockStackPlan&&) noexcept;
  NativeVisionAotBlockStackPlan& operator=(
      NativeVisionAotBlockStackPlan&&) noexcept;

  void launch(const void* input_device, const void* cos_device,
              const void* sin_device, void* output_device,
              void* temporary_device, std::size_t temporary_bytes,
              void* stream = nullptr) const;
  void launch_through(std::size_t last_block_index, const void* input_device,
                      const void* cos_device, const void* sin_device,
                      void* output_device, void* temporary_device,
                      std::size_t temporary_bytes,
                      void* stream = nullptr) const;
  std::size_t patch_count() const;
  std::size_t block_count() const;
  std::size_t temporary_bytes() const;
  std::size_t library_workspace_bytes() const;
  std::shared_ptr<const NativeVisionAotAttentionPlan> attention_plan() const;
  std::shared_ptr<NativeVisionAotBlockGemmPlans> gemm_plans() const;
  const std::string& attention_image_sha256() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
