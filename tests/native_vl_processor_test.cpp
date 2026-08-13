// SPDX-License-Identifier: Apache-2.0

#include "aima/native_vl_processor.h"

#include "aima/sha256.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "native_vl_processor_test: " << message << '\n';
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

void require_geometry(const aima::NativeVlResizeGeometry& actual,
                      std::size_t height, std::size_t width,
                      std::size_t temporal, std::size_t grid_height,
                      std::size_t grid_width, std::size_t tokens,
                      const char* message) {
  require(actual.resized_height == height && actual.resized_width == width &&
              actual.grid.temporal == temporal &&
              actual.grid.height == grid_height &&
              actual.grid.width == grid_width &&
              actual.grid.language_token_count() == tokens,
          message);
}

std::string index_sha256(const std::vector<std::size_t>& indices) {
  std::vector<std::uint64_t> values(indices.begin(), indices.end());
  return aima::sha256_bytes(values.data(), values.size() * sizeof(values[0]));
}

}  // namespace

int main() {
  require_geometry(aima::native_qwen36_image_geometry(1, 1),
                   256, 256, 1, 16, 16, 64, "minimum image geometry drifted");
  require_geometry(aima::native_qwen36_image_geometry(1, 200),
                   32, 3648, 1, 2, 228, 114,
                   "image aspect-ratio boundary drifted");
  require_geometry(aima::native_qwen36_image_geometry(1024, 256),
                   1024, 256, 1, 64, 16, 256,
                   "portrait image geometry drifted");
  require_geometry(aima::native_qwen36_image_geometry(256, 1024),
                   256, 1024, 1, 16, 64, 256,
                   "landscape image geometry drifted");
  require_geometry(aima::native_qwen36_image_geometry(31, 31),
                   256, 256, 1, 16, 16, 64,
                   "factor-minus-one image geometry drifted");
  require_geometry(aima::native_qwen36_image_geometry(32, 32),
                   256, 256, 1, 16, 16, 64,
                   "exact-factor image geometry drifted");
  require_geometry(aima::native_qwen36_image_geometry(33, 33),
                   256, 256, 1, 16, 16, 64,
                   "factor-plus-one image geometry drifted");
  require_geometry(aima::native_qwen36_image_geometry(4096, 4096),
                   4096, 4096, 1, 256, 256, 16384,
                   "maximum image geometry drifted");
  require_geometry(aima::native_qwen36_image_geometry(8192, 8192),
                   4096, 4096, 1, 256, 256, 16384,
                   "above-maximum image geometry drifted");
  require_invalid(
      []() { (void)aima::native_qwen36_image_geometry(1, 201); },
      "image aspect ratio over 200 was admitted");

  require_geometry(aima::native_qwen36_video_geometry(2, 32, 32),
                   64, 64, 1, 4, 4, 4,
                   "minimum video geometry drifted");
  require_geometry(aima::native_qwen36_video_geometry(4, 256, 256),
                   256, 256, 2, 16, 16, 128,
                   "typical video geometry drifted");
  require_geometry(aima::native_qwen36_video_geometry(2, 32, 6400),
                   32, 6400, 1, 2, 400, 200,
                   "video aspect-ratio boundary drifted");
  require_geometry(aima::native_qwen36_video_geometry(2, 3072, 4096),
                   3072, 4096, 1, 192, 256, 12288,
                   "maximum video geometry drifted");
  require_invalid(
      []() { (void)aima::native_qwen36_video_geometry(1, 32, 32); },
      "video below temporal factor was admitted");
  require_invalid(
      []() { (void)aima::native_qwen36_video_geometry(2, 31, 32); },
      "video below spatial factor was admitted");
  require_invalid(
      []() { (void)aima::native_qwen36_video_geometry(2, 32, 6432); },
      "video aspect ratio over 200 was admitted");

  struct SamplingCase {
    std::size_t total;
    double source_fps;
    aima::NativeVlVideoSamplingOptions options;
    std::size_t count;
    const char* sha256;
  };
  const std::vector<SamplingCase> sampling = {
      {3, 24.0, {}, 3,
       "ab25350e3e65efebe24584461683ecda68725576e825e550038b90e7b1479946"},
      {48, 24.0, {}, 4,
       "a924110a466cca2ae4aff9eaf3b18b3baef7c94eb5cec57dca182e38c483523a"},
      {240, 24.0, {}, 20,
       "03a3ced7d671441171c6f4b5a056bdb8a8df76e098020f352e2850e262339681"},
      {9216, 24.0, {}, 768,
       "c1a7cb36cc869f7b1cccfc2f82c349c4861d0c411b8243df7c3faf9733b85f70"},
      {18432, 24.0, {}, 768,
       "074733ebca7045e4ad03f309e5efeb06442e0d8001989ed02fd7bb8d4771fc38"},
      {240, 24.0, {32, std::nullopt}, 32,
       "d37af6c8a973cbc90718f82ff4d487bcd16c4bde67791d764c2274ce213b3896"},
  };
  for (const SamplingCase& sample : sampling) {
    const std::vector<std::size_t> indices =
        aima::native_qwen36_sample_video_frames(
            sample.total, sample.source_fps, sample.options);
    require(indices.size() == sample.count &&
                index_sha256(indices) == sample.sha256,
            "video sampling boundary drifted");
  }
  require_invalid(
      []() {
        (void)aima::native_qwen36_sample_video_frames(
            240, 24.0, {32, 2.0});
      },
      "explicit fps and num_frames conflict was admitted");

  const std::string image_prompt = aima::native_qwen36_expand_media_prompt(
      "A<|vision_start|><|image_pad|><|vision_end|>B",
      {{aima::NativeMediaKind::kImage, {1, 4, 4}, {}, 0.0}});
  require(image_prompt ==
              "A<|vision_start|><|image_pad|><|image_pad|><|image_pad|>"
              "<|image_pad|><|vision_end|>B",
          "image prompt expansion drifted");
  const std::string video_prompt = aima::native_qwen36_expand_media_prompt(
      "A<|vision_start|><|video_pad|><|vision_end|>B",
      {{aima::NativeMediaKind::kVideo, {2, 4, 4}, {0, 1, 2, 3}, 2.0}});
  require(video_prompt ==
              "A<0.2 seconds><|vision_start|><|video_pad|><|video_pad|>"
              "<|video_pad|><|video_pad|><|vision_end|><1.2 seconds>"
              "<|vision_start|><|video_pad|><|video_pad|><|video_pad|>"
              "<|video_pad|><|vision_end|>B",
          "video timestamp/prompt expansion drifted");

  require(aima::native_qwen36_processor_config_sha256() ==
              "2d5a1388bfaefa0cae6fd96c097a291bb180f0cb3074f7b51e83e00e4df237ab",
          "processor configuration identity is malformed");
  std::cout << "native_vl_processor_test: PASS\n";
  return 0;
}
