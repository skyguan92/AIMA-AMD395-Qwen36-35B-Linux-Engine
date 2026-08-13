// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace aima {

// Native launcher for the exact frozen vLLM Triton vision-attention code
// object. The AOT image is loaded and checked during plan construction; launch
// performs no filesystem or Python/Torch/Triton runtime work.
class NativeVisionAotAttentionPlan {
 public:
  NativeVisionAotAttentionPlan(
      const std::filesystem::path& image_path, std::size_t patch_count,
      const std::vector<std::uint32_t>& cu_seqlens);
  ~NativeVisionAotAttentionPlan();
  NativeVisionAotAttentionPlan(const NativeVisionAotAttentionPlan&) = delete;
  NativeVisionAotAttentionPlan& operator=(
      const NativeVisionAotAttentionPlan&) = delete;
  NativeVisionAotAttentionPlan(NativeVisionAotAttentionPlan&&) noexcept;
  NativeVisionAotAttentionPlan& operator=(
      NativeVisionAotAttentionPlan&&) noexcept;

  // Query, key, value and output are contiguous BF16 [patch_count,16,72].
  void launch(const void* query_device, const void* key_device,
              const void* value_device, void* output_device,
              void* stream = nullptr) const;
  std::size_t patch_count() const;
  std::size_t segment_count() const;
  std::size_t workspace_bytes() const;
  const std::string& image_sha256() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
