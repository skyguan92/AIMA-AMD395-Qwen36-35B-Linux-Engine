// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace aima {

class NativeVisionAotAttentionPlan;
class NativeWeightStore;

// One complete Qwen3.6 vision transformer block backed by the frozen,
// byte-qualified Triton attention code object. Attention state is shared so a
// 27-block stack loads one module and one sequence-boundary table.
class NativeVisionAotBlockPlan {
 public:
  NativeVisionAotBlockPlan(
      const NativeWeightStore& weights, std::size_t block_index,
      std::size_t patch_count,
      std::shared_ptr<const NativeVisionAotAttentionPlan> attention);
  ~NativeVisionAotBlockPlan();
  NativeVisionAotBlockPlan(const NativeVisionAotBlockPlan&) = delete;
  NativeVisionAotBlockPlan& operator=(const NativeVisionAotBlockPlan&) = delete;
  NativeVisionAotBlockPlan(NativeVisionAotBlockPlan&&) noexcept;
  NativeVisionAotBlockPlan& operator=(NativeVisionAotBlockPlan&&) noexcept;

  // input/output are distinct BF16 [patch_count,1152] tensors; cos/sin are
  // BF16 [patch_count,36]. temporary_device is caller-owned and reusable after
  // the launch completes on stream.
  void launch(const void* input_device, const void* cos_device,
              const void* sin_device, void* output_device,
              void* temporary_device, std::size_t temporary_bytes,
              void* stream = nullptr) const;
  std::size_t block_index() const;
  std::size_t patch_count() const;
  std::size_t temporary_bytes() const;
  // Excludes the separately shared AOT attention metadata.
  std::size_t library_workspace_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
