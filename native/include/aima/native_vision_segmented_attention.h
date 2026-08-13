// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace aima {

// Non-causal Qwen3.6 vision self-attention. cu_seqlens partitions concatenated
// media patches so that images and individual video frames never attend across
// a segment boundary. The implementation uses bounded online-softmax state and
// does not allocate a quadratic score tensor.
class NativeVisionSegmentedAttentionPlan {
 public:
  NativeVisionSegmentedAttentionPlan(
      std::size_t patch_count,
      const std::vector<std::uint32_t>& cu_seqlens);
  ~NativeVisionSegmentedAttentionPlan();
  NativeVisionSegmentedAttentionPlan(
      const NativeVisionSegmentedAttentionPlan&) = delete;
  NativeVisionSegmentedAttentionPlan& operator=(
      const NativeVisionSegmentedAttentionPlan&) = delete;
  NativeVisionSegmentedAttentionPlan(
      NativeVisionSegmentedAttentionPlan&&) noexcept;
  NativeVisionSegmentedAttentionPlan& operator=(
      NativeVisionSegmentedAttentionPlan&&) noexcept;

  // Query, key, value and output are distinct contiguous BF16
  // [patch_count,16,72] tensors.
  void launch(const void* query_device, const void* key_device,
              const void* value_device, void* output_device,
              void* stream = nullptr) const;
  std::size_t patch_count() const;
  std::size_t segment_count() const;
  std::size_t workspace_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
