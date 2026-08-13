// SPDX-License-Identifier: Apache-2.0

#include "aima/native_multimodal_cache.h"
#include "aima/native_vl_embedding.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "native_vl_embedding_test: " << message << '\n';
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
  using aima::NativeVlEmbeddingSpan;
  constexpr std::uint32_t image = aima::kNativeImagePadTokenId;
  constexpr std::uint32_t video = aima::kNativeVideoPadTokenId;

  const std::vector<std::uint32_t> prompt = {
      1, image, image, 2, 27, 15, 248053, video, video, 248054, 3};
  // Deliberately supply spans out of prompt order and map the video rows after
  // the image rows in the visual-merger tensor.
  const std::vector<NativeVlEmbeddingSpan> spans = {
      {NativeMediaKind::kVideo, 4, 6, 2, 2},
      {NativeMediaKind::kImage, 1, 2, 0, 2},
  };
  const aima::NativeVlEmbeddingPlan plan =
      aima::build_native_vl_embedding_plan(prompt, spans, 4);
  require(plan.prompt_token_count() == prompt.size(),
          "prompt token count changed");
  require(plan.visual_embedding_count() == 4 &&
              plan.device_index_bytes() == 8 * sizeof(std::uint32_t),
          "visual/device index sizing changed");
  require(plan.prompt_positions() ==
              std::vector<std::uint32_t>({1, 2, 7, 8}),
          "prompt scatter positions changed");
  require(plan.visual_rows() == std::vector<std::uint32_t>({0, 1, 2, 3}),
          "visual row mapping changed");

  auto source_reordered = spans;
  source_reordered[0].visual_embedding_offset = 0;
  source_reordered[1].visual_embedding_offset = 2;
  const auto reordered =
      aima::build_native_vl_embedding_plan(prompt, source_reordered, 4);
  require(reordered.visual_rows() ==
              std::vector<std::uint32_t>({2, 3, 0, 1}),
          "explicit source row ordering was ignored");

  require_invalid(
      [&]() {
        auto value = prompt;
        value.push_back(image);
        (void)aima::build_native_vl_embedding_plan(value, spans, 4);
      },
      "orphan image placeholder was admitted");
  require_invalid(
      [&]() {
        auto value = spans;
        value[0].visual_embedding_count = 3;
        (void)aima::build_native_vl_embedding_plan(prompt, value, 5);
      },
      "placeholder/visual count mismatch was admitted");
  require_invalid(
      [&]() {
        auto value = spans;
        value[0].token_offset = 2;
        (void)aima::build_native_vl_embedding_plan(prompt, value, 4);
      },
      "overlapping prompt spans were admitted");
  require_invalid(
      [&]() {
        auto value = spans;
        value[0].visual_embedding_offset = 1;
        (void)aima::build_native_vl_embedding_plan(prompt, value, 4);
      },
      "overlapping visual spans were admitted");
  require_invalid(
      [&]() {
        auto value = prompt;
        value[7] = image;
        (void)aima::build_native_vl_embedding_plan(value, spans, 4);
      },
      "cross-modality placeholder was admitted");
  require_invalid(
      [&]() {
        auto value = spans;
        value[0].token_length = prompt.size();
        (void)aima::build_native_vl_embedding_plan(prompt, value, 4);
      },
      "out-of-range prompt span was admitted");
  require_invalid(
      [&]() {
        auto value = spans;
        value[0].visual_embedding_offset = 3;
        (void)aima::build_native_vl_embedding_plan(prompt, value, 4);
      },
      "out-of-range visual span was admitted");
  require_invalid(
      [&]() {
        (void)aima::build_native_vl_embedding_plan(prompt, {}, 0);
      },
      "empty media plan was admitted");

  std::cout << "native_vl_embedding_test: PASS\n";
  return 0;
}
