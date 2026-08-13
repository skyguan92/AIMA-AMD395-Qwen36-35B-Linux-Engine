// SPDX-License-Identifier: Apache-2.0

#include "aima/native_mrope.h"
#include "aima/native_multimodal_cache.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "native_mrope_test: " << message << '\n';
    std::exit(1);
  }
}

template <typename Function>
void require_invalid(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  require(false, message);
}

}  // namespace

int main() {
  using aima::NativeMediaKind;
  using aima::NativeMropeMedia;
  constexpr std::uint32_t image = aima::kNativeImagePadTokenId;
  constexpr std::uint32_t video = aima::kNativeVideoPadTokenId;
  constexpr std::uint32_t start = aima::kNativeVisionStartTokenId;
  constexpr std::uint32_t end = aima::kNativeVisionEndTokenId;

  const std::vector<std::uint32_t> image_prompt = {
      10, 11, image, image, image, image, 12};
  const std::vector<NativeMropeMedia> image_media = {
      {NativeMediaKind::kImage, 2, 4, {1, 4, 4}},
  };
  const aima::NativeMropePlan image_plan =
      aima::build_native_mrope_plan(image_prompt, image_media);
  require(image_plan.positions() ==
              std::vector<std::int64_t>({
                  0, 1, 2, 2, 2, 2, 4,
                  0, 1, 2, 2, 3, 3, 4,
                  0, 1, 2, 3, 2, 3, 4,
              }),
          "image M-RoPE positions changed");
  require(image_plan.maximum_position() == 4 &&
              image_plan.position_delta() == -2 &&
              aima::native_mrope_decode_position(7, -2, 0) == 5 &&
              aima::native_mrope_decode_position(7, -2, 3) == 8,
          "image M-RoPE delta/decode continuation changed");

  const std::vector<std::uint32_t> video_prompt = {
      1,
      27, 15, start, video, video, video, video, end,
      27, 16, start, video, video, video, video, end,
      2,
  };
  const std::vector<NativeMropeMedia> video_media = {
      {NativeMediaKind::kVideo, 1, 16, {2, 4, 4}},
  };
  const aima::NativeMropePlan video_plan =
      aima::build_native_mrope_plan(video_prompt, video_media);
  require(video_plan.prompt_token_count() == 18 &&
              video_plan.maximum_position() == 13 &&
              video_plan.position_delta() == -4,
          "video M-RoPE shape/delta changed");
  const auto& video_positions = video_plan.positions();
  require(video_positions[4] == 4 && video_positions[18 + 4] == 4 &&
              video_positions[36 + 4] == 4 &&
              video_positions[12] == 10 &&
              video_positions[18 + 14] == 11 &&
              video_positions[36 + 15] == 11,
          "video temporal grid positions changed");

  require_invalid(
      [&]() {
        auto prompt = image_prompt;
        prompt.push_back(image);
        (void)aima::build_native_mrope_plan(prompt, image_media);
      },
      "orphan placeholder was admitted");
  require_invalid(
      [&]() {
        auto media = image_media;
        media[0].grid.temporal = 2;
        (void)aima::build_native_mrope_plan(image_prompt, media);
      },
      "multi-frame image grid was admitted");
  require_invalid(
      [&]() {
        auto media = image_media;
        media[0].grid.height =
            std::numeric_limits<std::size_t>::max() - 1;
        media[0].grid.width = 4;
        (void)aima::build_native_mrope_plan(image_prompt, media);
      },
      "overflowing media grid was admitted");
  require_invalid(
      [&]() {
        auto prompt = video_prompt;
        prompt[8] = 3;
        (void)aima::build_native_mrope_plan(prompt, video_media);
      },
      "video frame without vision_end was admitted");
  require_invalid(
      [&]() {
        auto prompt = video_prompt;
        prompt[6] = 3;
        (void)aima::build_native_mrope_plan(prompt, video_media);
      },
      "non-contiguous video placeholders were admitted");
  require_invalid(
      [&]() {
        (void)aima::native_mrope_decode_position(7, -8, 0);
      },
      "negative decode position was admitted");

  std::cout << "native_mrope_test: PASS\n";
  return 0;
}
