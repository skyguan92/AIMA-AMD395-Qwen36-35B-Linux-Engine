// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace aima {

class NativeWeightStore;

// One complete parameterized Qwen3.6 vision transformer block. The caller
// supplies reusable temporary storage so all 27 resident block plans can share
// a single request arena rather than allocating per layer.
class NativeVisionBlockPlan {
 public:
  NativeVisionBlockPlan(const NativeWeightStore& weights,
                        std::size_t block_index, std::size_t patch_count,
                        const std::vector<std::uint32_t>& cu_seqlens);
  ~NativeVisionBlockPlan();
  NativeVisionBlockPlan(const NativeVisionBlockPlan&) = delete;
  NativeVisionBlockPlan& operator=(const NativeVisionBlockPlan&) = delete;
  NativeVisionBlockPlan(NativeVisionBlockPlan&&) noexcept;
  NativeVisionBlockPlan& operator=(NativeVisionBlockPlan&&) noexcept;

  // input/output are BF16 [patch_count,1152]; cos/sin are
  // [patch_count,36]. input and output must be distinct. temporary_device must
  // provide at least temporary_bytes() and may be reused by another block only
  // after this launch has completed on the supplied stream.
  void launch(const void* input_device, const void* cos_device,
              const void* sin_device, void* output_device,
              void* temporary_device, std::size_t temporary_bytes,
              void* stream = nullptr) const;
  std::size_t block_index() const;
  std::size_t patch_count() const;
  std::size_t temporary_bytes() const;
  std::size_t library_workspace_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
