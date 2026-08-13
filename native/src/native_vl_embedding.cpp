// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vl_embedding.h"

#include "aima/native_multimodal_cache.h"
#include "aima/native_vl_processor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace aima {
namespace {

constexpr std::size_t kMaximumPromptTokens = 262144;

std::uint32_t placeholder_token_id(NativeMediaKind kind) {
  switch (kind) {
    case NativeMediaKind::kImage:
      return kNativeImagePadTokenId;
    case NativeMediaKind::kVideo:
      return kNativeVideoPadTokenId;
  }
  throw std::invalid_argument("native VL embedding span has an invalid kind");
}

}  // namespace

NativeVlEmbeddingPlan build_native_vl_embedding_plan(
    const std::vector<std::uint32_t>& prompt_token_ids,
    const std::vector<NativeVlEmbeddingSpan>& input_spans,
    std::size_t visual_embedding_count) {
  if (prompt_token_ids.empty() ||
      prompt_token_ids.size() > kMaximumPromptTokens) {
    throw std::invalid_argument(
        "native VL embedding prompt token count is invalid");
  }
  if (input_spans.empty() || visual_embedding_count == 0 ||
      visual_embedding_count > kNativeVlAggregateTokenLimit) {
    throw std::invalid_argument(
        "native VL embedding media count is invalid");
  }

  std::vector<NativeVlEmbeddingSpan> spans = input_spans;
  std::sort(spans.begin(), spans.end(),
            [](const NativeVlEmbeddingSpan& left,
               const NativeVlEmbeddingSpan& right) {
              if (left.token_offset != right.token_offset) {
                return left.token_offset < right.token_offset;
              }
              return left.visual_embedding_offset <
                     right.visual_embedding_offset;
            });

  NativeVlEmbeddingPlan plan;
  plan.prompt_token_count_ = prompt_token_ids.size();
  plan.visual_embedding_count_ = visual_embedding_count;
  plan.prompt_positions_.reserve(visual_embedding_count);
  plan.visual_rows_.reserve(visual_embedding_count);
  std::vector<unsigned char> occupied_prompt(prompt_token_ids.size(), 0);
  std::vector<unsigned char> selected_prompt(prompt_token_ids.size(), 0);
  std::vector<unsigned char> occupied_visual(visual_embedding_count, 0);

  for (const NativeVlEmbeddingSpan& span : spans) {
    if (span.token_length == 0 || span.visual_embedding_count == 0 ||
        span.token_offset > prompt_token_ids.size() ||
        span.token_length > prompt_token_ids.size() - span.token_offset ||
        span.visual_embedding_offset > visual_embedding_count ||
        span.visual_embedding_count >
            visual_embedding_count - span.visual_embedding_offset) {
      throw std::invalid_argument(
          "native VL embedding span is outside its tensor");
    }
    for (std::size_t index = span.token_offset;
         index < span.token_offset + span.token_length; ++index) {
      if (occupied_prompt[index] != 0) {
        throw std::invalid_argument(
            "native VL embedding prompt spans overlap");
      }
      occupied_prompt[index] = 1;
    }
    for (std::size_t index = span.visual_embedding_offset;
         index < span.visual_embedding_offset + span.visual_embedding_count;
         ++index) {
      if (occupied_visual[index] != 0) {
        throw std::invalid_argument(
            "native VL embedding visual spans overlap");
      }
      occupied_visual[index] = 1;
    }

    const std::uint32_t expected_token = placeholder_token_id(span.kind);
    const std::uint32_t other_token =
        span.kind == NativeMediaKind::kImage ? kNativeVideoPadTokenId
                                             : kNativeImagePadTokenId;
    std::size_t selected = 0;
    for (std::size_t index = span.token_offset;
         index < span.token_offset + span.token_length; ++index) {
      const std::uint32_t token = prompt_token_ids[index];
      if (token == other_token) {
        throw std::invalid_argument(
            "native VL embedding span contains the other media token");
      }
      if (token != expected_token) continue;
      if (selected >= span.visual_embedding_count) {
        throw std::invalid_argument(
            "native VL embedding span has too many placeholder tokens");
      }
      plan.prompt_positions_.push_back(static_cast<std::uint32_t>(index));
      plan.visual_rows_.push_back(static_cast<std::uint32_t>(
          span.visual_embedding_offset + selected));
      selected_prompt[index] = 1;
      ++selected;
    }
    if (selected != span.visual_embedding_count) {
      throw std::invalid_argument(
          "native VL embedding placeholder count differs from visual rows");
    }
  }

  if (!std::all_of(occupied_visual.begin(), occupied_visual.end(),
                   [](unsigned char value) { return value != 0; }) ||
      plan.prompt_positions_.size() != visual_embedding_count ||
      plan.visual_rows_.size() != visual_embedding_count) {
    throw std::invalid_argument(
        "native VL embedding visual rows are not covered exactly once");
  }
  for (std::size_t index = 0; index < prompt_token_ids.size(); ++index) {
    const std::uint32_t token = prompt_token_ids[index];
    if ((token == kNativeImagePadTokenId ||
         token == kNativeVideoPadTokenId) &&
        selected_prompt[index] == 0) {
      throw std::invalid_argument(
          "native VL embedding prompt has an orphan placeholder token");
    }
  }
  return plan;
}

}  // namespace aima
