// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aima/native_media.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aima {

// One processor-owned placeholder range in the final language prompt.  The
// visual range is explicit because a future scheduler may encode media in a
// different order from their prompt positions.
struct NativeVlEmbeddingSpan {
  NativeMediaKind kind = NativeMediaKind::kImage;
  std::size_t token_offset = 0;
  std::size_t token_length = 0;
  std::size_t visual_embedding_offset = 0;
  std::size_t visual_embedding_count = 0;
};

// Immutable, validated scatter plan from visual-merger rows into the prompt
// embedding tensor.  The host plan is request-sized; device storage is caller
// owned so the resident runtime can reuse preallocated request workspace.
class NativeVlEmbeddingPlan {
 public:
  std::size_t prompt_token_count() const { return prompt_token_count_; }
  std::size_t visual_embedding_count() const {
    return visual_embedding_count_;
  }
  std::size_t device_index_bytes() const {
    return 2 * prompt_positions_.size() * sizeof(std::uint32_t);
  }
  const std::vector<std::uint32_t>& prompt_positions() const {
    return prompt_positions_;
  }
  const std::vector<std::uint32_t>& visual_rows() const {
    return visual_rows_;
  }

 private:
  NativeVlEmbeddingPlan() = default;
  friend NativeVlEmbeddingPlan build_native_vl_embedding_plan(
      const std::vector<std::uint32_t>&,
      const std::vector<NativeVlEmbeddingSpan>&, std::size_t);

  std::size_t prompt_token_count_ = 0;
  std::size_t visual_embedding_count_ = 0;
  std::vector<std::uint32_t> prompt_positions_;
  std::vector<std::uint32_t> visual_rows_;
};

// Selects only image/video pad-token positions inside the processor-supplied
// spans.  It rejects overlapping/orphan prompt placeholders and requires the
// visual source ranges to cover every merger row exactly once.
NativeVlEmbeddingPlan build_native_vl_embedding_plan(
    const std::vector<std::uint32_t>& prompt_token_ids,
    const std::vector<NativeVlEmbeddingSpan>& spans,
    std::size_t visual_embedding_count);

// Materializes ordinary token embeddings, uploads the validated scatter
// indices, and replaces the selected rows with BF16 visual-merger rows.
void launch_native_vl_embeddings(
    const void* token_embedding_bf16,
    const std::uint32_t* host_prompt_token_ids,
    const NativeVlEmbeddingPlan& plan,
    const void* visual_embeddings_bf16,
    void* device_prompt_token_ids,
    void* device_scatter_indices,
    void* output_bf16,
    void* stream = nullptr);

}  // namespace aima
