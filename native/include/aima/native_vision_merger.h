// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <memory>

namespace aima {

class NativeWeightStore;

// Qwen3.6 vision patch merger: exact LayerNorm over each 1152-wide patch,
// contiguous 2x2 grouping into 4608 features, FC1, exact GELU and FC2 to the
// 2048-wide language embedding space.
class NativeVisionMergerPlan {
 public:
  NativeVisionMergerPlan(const NativeWeightStore& weights,
                         std::size_t patch_count);
  ~NativeVisionMergerPlan();
  NativeVisionMergerPlan(const NativeVisionMergerPlan&) = delete;
  NativeVisionMergerPlan& operator=(const NativeVisionMergerPlan&) = delete;
  NativeVisionMergerPlan(NativeVisionMergerPlan&&) noexcept;
  NativeVisionMergerPlan& operator=(NativeVisionMergerPlan&&) noexcept;

  // input is BF16 [patch_count,1152], output is distinct BF16
  // [patch_count/4,2048]. temporary_device is caller-owned and reusable after
  // completion on stream.
  void launch(const void* input_device, void* output_device,
              void* temporary_device, std::size_t temporary_bytes,
              void* stream = nullptr) const;

  // Qualification and composition boundaries. normalized is physically
  // [patch_count,1152] and is viewed by FC1 as [patch_count/4,4608].
  void launch_norm(const void* input_device, void* normalized_device,
                   void* stream = nullptr) const;
  void launch_fc1(const void* normalized_device, void* output_device,
                  void* stream = nullptr) const;
  void launch_gelu(const void* input_device, void* output_device,
                   void* stream = nullptr) const;
  void launch_fc2(const void* activated_device, void* output_device,
                  void* stream = nullptr) const;

  std::size_t patch_count() const;
  std::size_t merged_token_count() const;
  std::size_t temporary_bytes() const;
  std::size_t library_workspace_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
