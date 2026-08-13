// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aima/native_media.h"
#include "aima/native_vl_processor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aima {

inline constexpr std::uint32_t kNativeVisionStartTokenId = 248053;
inline constexpr std::uint32_t kNativeVisionEndTokenId = 248054;

struct NativeMropeMedia {
  NativeMediaKind kind = NativeMediaKind::kImage;
  std::size_t token_offset = 0;
  std::size_t token_length = 0;
  NativeVlGrid grid;
};

// Exact integer Qwen3-VL prompt positions in contiguous [3,prompt_tokens]
// row-major order, plus the delta used to continue positions during decode.
class NativeMropePlan {
 public:
  std::size_t prompt_token_count() const { return prompt_token_count_; }
  const std::vector<std::int64_t>& positions() const { return positions_; }
  std::int64_t position_delta() const { return position_delta_; }
  std::int64_t maximum_position() const { return maximum_position_; }

 private:
  NativeMropePlan() = default;
  friend NativeMropePlan build_native_mrope_plan(
      const std::vector<std::uint32_t>&,
      const std::vector<NativeMropeMedia>&);

  std::size_t prompt_token_count_ = 0;
  std::vector<std::int64_t> positions_;
  std::int64_t position_delta_ = 0;
  std::int64_t maximum_position_ = 0;
};

// Mirrors the pinned vLLM Qwen3-VL `_get_mrope_input_positions` contract.
// Image spans contain only image-pad tokens. Video spans contain timestamp,
// vision-start, video-pad and vision-end tokens for every temporal grid row.
NativeMropePlan build_native_mrope_plan(
    const std::vector<std::uint32_t>& prompt_token_ids,
    const std::vector<NativeMropeMedia>& media);

// Position shared by all three axes for a token decoded after the prompt.
std::int64_t native_mrope_decode_position(
    std::size_t prompt_token_count, std::int64_t position_delta,
    std::size_t decoded_token_offset);

}  // namespace aima
