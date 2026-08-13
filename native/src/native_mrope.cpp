// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_mrope.h"

#include "aima/native_multimodal_cache.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace aima {
namespace {

constexpr std::size_t kMaximumPromptTokens = 262144;

struct MropeSegment {
  std::size_t token_offset = 0;
  std::size_t grid_height = 0;
  std::size_t grid_width = 0;
  std::size_t actual_tokens = 0;
};

std::size_t find_token(const std::vector<std::uint32_t>& tokens,
                       std::uint32_t token, std::size_t begin,
                       std::size_t end) {
  for (std::size_t index = begin; index < end; ++index) {
    if (tokens[index] == token) return index;
  }
  return end;
}

std::uint32_t media_pad_token(NativeMediaKind kind) {
  switch (kind) {
    case NativeMediaKind::kImage:
      return kNativeImagePadTokenId;
    case NativeMediaKind::kVideo:
      return kNativeVideoPadTokenId;
  }
  throw std::invalid_argument("native M-RoPE media kind is invalid");
}

void append_text_positions(std::array<std::vector<std::int64_t>, 3>* rows,
                           std::size_t count, std::int64_t* next_position) {
  for (std::size_t index = 0; index < count; ++index) {
    const std::int64_t value =
        *next_position + static_cast<std::int64_t>(index);
    for (auto& row : *rows) row.push_back(value);
  }
  *next_position += static_cast<std::int64_t>(count);
}

void append_grid_positions(std::array<std::vector<std::int64_t>, 3>* rows,
                           std::size_t height, std::size_t width,
                           std::size_t count,
                           std::int64_t* next_position) {
  const std::size_t full_count = height * width;
  if (height == 0 || width == 0 || count == 0 || count > full_count) {
    throw std::invalid_argument("native M-RoPE grid count is invalid");
  }
  const std::int64_t base = *next_position;
  std::int64_t maximum = base;
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t row = index / width;
    const std::size_t column = index - row * width;
    const std::array<std::int64_t, 3> values = {
        base, base + static_cast<std::int64_t>(row),
        base + static_cast<std::int64_t>(column)};
    for (std::size_t axis = 0; axis < rows->size(); ++axis) {
      (*rows)[axis].push_back(values[axis]);
      maximum = std::max(maximum, values[axis]);
    }
  }
  *next_position = maximum + 1;
}

std::vector<MropeSegment> build_segments(
    const std::vector<std::uint32_t>& tokens,
    const std::vector<NativeMropeMedia>& input_media) {
  std::vector<NativeMropeMedia> media = input_media;
  std::sort(media.begin(), media.end(),
            [](const NativeMropeMedia& left,
               const NativeMropeMedia& right) {
              return left.token_offset < right.token_offset;
            });
  std::vector<unsigned char> occupied(tokens.size(), 0);
  std::vector<unsigned char> selected_placeholder(tokens.size(), 0);
  std::vector<MropeSegment> segments;
  std::size_t visual_tokens = 0;
  for (const NativeMropeMedia& item : media) {
    if (item.token_length == 0 || item.token_offset > tokens.size() ||
        item.token_length > tokens.size() - item.token_offset ||
        item.grid.temporal == 0 || item.grid.height == 0 ||
        item.grid.width == 0 ||
        item.grid.height % kNativeVlMergeSize != 0 ||
        item.grid.width % kNativeVlMergeSize != 0) {
      throw std::invalid_argument("native M-RoPE media span/grid is invalid");
    }
    const std::size_t end = item.token_offset + item.token_length;
    for (std::size_t index = item.token_offset; index < end; ++index) {
      if (occupied[index] != 0) {
        throw std::invalid_argument("native M-RoPE media spans overlap");
      }
      occupied[index] = 1;
    }
    const std::size_t grid_height =
        item.grid.height / kNativeVlMergeSize;
    const std::size_t grid_width = item.grid.width / kNativeVlMergeSize;
    const std::size_t tokens_per_grid = grid_height * grid_width;
    if (tokens_per_grid == 0 ||
        tokens_per_grid > kNativeVlAggregateTokenLimit - visual_tokens ||
        item.grid.temporal >
            (kNativeVlAggregateTokenLimit - visual_tokens) /
                tokens_per_grid) {
      throw std::invalid_argument("native M-RoPE visual-token budget exceeded");
    }
    visual_tokens += tokens_per_grid * item.grid.temporal;

    const std::uint32_t expected_pad = media_pad_token(item.kind);
    const std::uint32_t other_pad =
        item.kind == NativeMediaKind::kImage ? kNativeVideoPadTokenId
                                             : kNativeImagePadTokenId;
    if (item.kind == NativeMediaKind::kImage) {
      if (item.grid.temporal != 1 || item.token_length != tokens_per_grid) {
        throw std::invalid_argument(
            "native M-RoPE image span differs from its grid");
      }
      for (std::size_t index = item.token_offset; index < end; ++index) {
        if (tokens[index] != expected_pad) {
          throw std::invalid_argument(
              "native M-RoPE image span contains a non-image token");
        }
        selected_placeholder[index] = 1;
      }
      segments.push_back(MropeSegment{item.token_offset, grid_height,
                                      grid_width, tokens_per_grid});
      continue;
    }

    std::size_t search = item.token_offset;
    for (std::size_t frame = 0; frame < item.grid.temporal; ++frame) {
      const std::size_t vision_start =
          find_token(tokens, kNativeVisionStartTokenId, search, end);
      if (vision_start == end) {
        throw std::invalid_argument(
            "native M-RoPE video frame is missing vision_start");
      }
      const std::size_t vision_end = find_token(
          tokens, kNativeVisionEndTokenId, vision_start + 1, end);
      if (vision_end == end) {
        throw std::invalid_argument(
            "native M-RoPE video frame is missing vision_end");
      }
      const std::size_t video_offset =
          find_token(tokens, expected_pad, vision_start, vision_end);
      std::size_t actual_tokens = 0;
      std::size_t segment_offset = vision_start + 1;
      if (video_offset != vision_end) {
        segment_offset = video_offset;
        actual_tokens = vision_end - video_offset;
        for (std::size_t index = video_offset; index < vision_end; ++index) {
          if (tokens[index] != expected_pad) {
            throw std::invalid_argument(
                "native M-RoPE video pad tokens are not contiguous");
          }
          selected_placeholder[index] = 1;
        }
      }
      for (std::size_t index = search; index <= vision_end; ++index) {
        if (tokens[index] == other_pad) {
          throw std::invalid_argument(
              "native M-RoPE video span contains an image token");
        }
      }
      segments.push_back(MropeSegment{segment_offset, grid_height,
                                      grid_width, actual_tokens});
      search = vision_end + 1;
    }
    if (search != end) {
      throw std::invalid_argument(
          "native M-RoPE video span has trailing unparsed tokens");
    }
  }
  if (visual_tokens > kNativeVlAggregateTokenLimit) {
    throw std::invalid_argument("native M-RoPE visual-token budget exceeded");
  }
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    if ((tokens[index] == kNativeImagePadTokenId ||
         tokens[index] == kNativeVideoPadTokenId) &&
        selected_placeholder[index] == 0) {
      throw std::invalid_argument(
          "native M-RoPE prompt has an orphan placeholder token");
    }
  }
  return segments;
}

}  // namespace

NativeMropePlan build_native_mrope_plan(
    const std::vector<std::uint32_t>& prompt_token_ids,
    const std::vector<NativeMropeMedia>& media) {
  if (prompt_token_ids.empty() ||
      prompt_token_ids.size() > kMaximumPromptTokens || media.empty()) {
    throw std::invalid_argument("native M-RoPE prompt/media count is invalid");
  }
  const std::vector<MropeSegment> segments =
      build_segments(prompt_token_ids, media);
  std::array<std::vector<std::int64_t>, 3> rows;
  for (auto& row : rows) row.reserve(prompt_token_ids.size());
  std::size_t consumed_tokens = 0;
  std::int64_t next_position = 0;
  for (const MropeSegment& segment : segments) {
    if (segment.actual_tokens == 0) continue;
    if (segment.token_offset < consumed_tokens ||
        segment.actual_tokens >
            prompt_token_ids.size() - segment.token_offset) {
      throw std::invalid_argument(
          "native M-RoPE segment ordering/count is invalid");
    }
    append_text_positions(&rows, segment.token_offset - consumed_tokens,
                          &next_position);
    const std::size_t tokens_per_grid =
        segment.grid_height * segment.grid_width;
    const std::size_t full_grids = segment.actual_tokens / tokens_per_grid;
    const std::size_t remainder = segment.actual_tokens % tokens_per_grid;
    for (std::size_t index = 0; index < full_grids; ++index) {
      append_grid_positions(&rows, segment.grid_height, segment.grid_width,
                            tokens_per_grid, &next_position);
    }
    if (remainder != 0) {
      append_grid_positions(&rows, segment.grid_height, segment.grid_width,
                            remainder, &next_position);
    }
    consumed_tokens = segment.token_offset + segment.actual_tokens;
  }
  append_text_positions(&rows, prompt_token_ids.size() - consumed_tokens,
                        &next_position);
  NativeMropePlan plan;
  plan.prompt_token_count_ = prompt_token_ids.size();
  plan.positions_.reserve(3 * prompt_token_ids.size());
  for (const auto& row : rows) {
    if (row.size() != prompt_token_ids.size()) {
      throw std::runtime_error("native M-RoPE output shape is inconsistent");
    }
    plan.positions_.insert(plan.positions_.end(), row.begin(), row.end());
  }
  plan.maximum_position_ = next_position - 1;
  plan.position_delta_ =
      next_position - static_cast<std::int64_t>(prompt_token_ids.size());
  return plan;
}

std::int64_t native_mrope_decode_position(
    std::size_t prompt_token_count, std::int64_t position_delta,
    std::size_t decoded_token_offset) {
  if (prompt_token_count == 0 || prompt_token_count > kMaximumPromptTokens ||
      decoded_token_offset >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) -
              prompt_token_count) {
    throw std::invalid_argument("native M-RoPE decode position is invalid");
  }
  const std::int64_t base =
      static_cast<std::int64_t>(prompt_token_count + decoded_token_offset);
  if ((position_delta > 0 &&
       base > std::numeric_limits<std::int64_t>::max() - position_delta) ||
      (position_delta < 0 && position_delta < -base)) {
    throw std::invalid_argument("native M-RoPE decode position overflows");
  }
  return base + position_delta;
}

}  // namespace aima
