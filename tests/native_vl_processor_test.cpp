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

aima::NativeRgbFrame deterministic_image(std::size_t height = 256,
                                         std::size_t width = 256) {
  aima::NativeRgbFrame frame;
  frame.height = height;
  frame.width = width;
  frame.pixels.resize(frame.height * frame.width * 3);
  for (std::size_t y = 0; y < frame.height; ++y) {
    for (std::size_t x = 0; x < frame.width; ++x) {
      const std::size_t offset = (y * frame.width + x) * 3;
      frame.pixels[offset] =
          static_cast<unsigned char>((x * 3 + y * 5 + 1) % 256);
      frame.pixels[offset + 1] =
          static_cast<unsigned char>((x * 7 + y * 11 + 3) % 256);
      frame.pixels[offset + 2] =
          static_cast<unsigned char>((x * 13 + y * 17 + 5) % 256);
    }
  }
  return frame;
}

std::vector<aima::NativeRgbFrame> deterministic_video() {
  std::vector<aima::NativeRgbFrame> frames(4);
  for (std::size_t frame_index = 0; frame_index < frames.size();
       ++frame_index) {
    aima::NativeRgbFrame& frame = frames[frame_index];
    frame.height = 256;
    frame.width = 256;
    frame.pixels.resize(frame.height * frame.width * 3);
    for (std::size_t y = 0; y < frame.height; ++y) {
      for (std::size_t x = 0; x < frame.width; ++x) {
        const std::size_t source_x =
            (x + frame.width - frame_index) % frame.width;
        for (std::size_t channel = 0; channel < 3; ++channel) {
          frame.pixels[(y * frame.width + x) * 3 + channel] =
              static_cast<unsigned char>(
                  ((y * frame.width + source_x) * 3 + channel) % 256);
        }
      }
    }
  }
  return frames;
}

}  // namespace

int main() {
  require(aima::kNativeVlAggregateTokenLimit == 262144 &&
              aima::kNativeVlVisionBatchTokenLimit == 16384 &&
              aima::kNativeVlVisionBatchPatchLimit == 65536 &&
              aima::kNativeVlAggregatePatchLimit == 1048576,
          "processor-derived aggregate/batch limits drifted");
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
  require_geometry(aima::native_qwen36_video_geometry(20, 256, 256),
                   256, 256, 10, 16, 16, 640,
                   "typical sampling execution geometry drifted");
  require_geometry(aima::native_qwen36_video_geometry(768, 256, 256),
                   160, 160, 384, 10, 10, 9600,
                   "maximum sampling execution geometry drifted");
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

  const aima::NativeVlGrid maximum_image_grid{1, 256, 256};
  const aima::NativeVlGrid minimum_image_grid{1, 16, 16};
  const aima::NativeVlGrid maximum_video_grid{1, 192, 256};
  const std::vector<aima::NativeVlVisionBatch> mixed_batches =
      aima::native_qwen36_vision_batches(
          {maximum_image_grid, minimum_image_grid, {2, 16, 16}});
  require(mixed_batches.size() == 2 &&
              mixed_batches[0].media_offset == 0 &&
              mixed_batches[0].media_count == 1 &&
              mixed_batches[0].patch_offset == 0 &&
              mixed_batches[0].patch_count == 65536 &&
              mixed_batches[0].visual_token_offset == 0 &&
              mixed_batches[0].visual_token_count == 16384 &&
              mixed_batches[1].media_offset == 1 &&
              mixed_batches[1].media_count == 2 &&
              mixed_batches[1].patch_offset == 65536 &&
              mixed_batches[1].patch_count == 768 &&
              mixed_batches[1].visual_token_offset == 16384 &&
              mixed_batches[1].visual_token_count == 192,
          "ordered bounded vision batching drifted");

  const std::vector<aima::NativeVlGrid> maximum_image_count(
      16, maximum_image_grid);
  const auto image_count_batches =
      aima::native_qwen36_vision_batches(maximum_image_count);
  require(image_count_batches.size() == 16 &&
              image_count_batches.back().visual_token_offset == 245760 &&
              image_count_batches.back().visual_token_count == 16384,
          "maximum image-count encoder budget was not admitted");
  require_invalid(
      [&]() {
        auto over = maximum_image_count;
        over.push_back(maximum_image_grid);
        (void)aima::native_qwen36_vision_batches(over);
      },
      "image aggregate above the full encoder budget was admitted");

  const std::vector<aima::NativeVlGrid> maximum_video_count(
      21, maximum_video_grid);
  const auto video_count_batches =
      aima::native_qwen36_vision_batches(maximum_video_count);
  require(video_count_batches.size() == 21 &&
              video_count_batches.back().visual_token_offset == 245760 &&
              video_count_batches.back().visual_token_count == 12288,
          "maximum video-count encoder budget was not admitted");
  require_invalid(
      [&]() {
        auto over = maximum_video_count;
        over.push_back(maximum_video_grid);
        (void)aima::native_qwen36_vision_batches(over);
      },
      "video aggregate above the full encoder budget was admitted");

  const std::vector<aima::NativeVlGrid> small_image_count(
      16, minimum_image_grid);
  const auto small_image_batches =
      aima::native_qwen36_vision_batches(small_image_count);
  require(small_image_batches.size() == 1 &&
              small_image_batches[0].media_count == 16 &&
              small_image_batches[0].visual_token_count == 1024,
          "small maximum-count images were split unnecessarily");

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

  const aima::NativeVlResizeGeometry image_geometry =
      aima::native_qwen36_image_geometry(256, 256);
  const aima::NativeVlPixelTensor pixels =
      aima::native_qwen36_patchify_resized_rgb(
          {deterministic_image()}, image_geometry);
  require(pixels.rows == 256 && pixels.columns == 1536 &&
              pixels.values.size() == 256 * 1536 &&
              aima::sha256_bytes(
                  pixels.values.data(),
                  pixels.values.size() * sizeof(pixels.values[0])) ==
                  "28e3bf47e74e94a78db819016eee9ce02983f93ab86012de846a27d72a1623b8",
          "image BF16 normalize/patchify oracle drifted");

  const aima::NativeRgbFrame mixed_scale_source =
      deterministic_image(511, 333);
  const aima::NativeVlResizeGeometry mixed_scale_geometry =
      aima::native_qwen36_image_geometry(mixed_scale_source.height,
                                         mixed_scale_source.width);
  const aima::NativeRgbFrame mixed_scale_resized =
      aima::native_qwen36_resize_rgb(mixed_scale_source,
                                     mixed_scale_geometry);
  require(mixed_scale_resized.height == 512 &&
              mixed_scale_resized.width == 320 &&
              aima::sha256_bytes(mixed_scale_resized.pixels.data(),
                                 mixed_scale_resized.pixels.size()) ==
                  "c95e1d293f41e6cccd327081480b7cec7728cb897b72fb0cf3f35b5ab0f539d1",
          "mixed up/down uint8 bicubic resize oracle drifted");
  const aima::NativeVlPixelTensor mixed_scale_pixels =
      aima::native_qwen36_process_rgb({mixed_scale_source},
                                      mixed_scale_geometry);
  require(mixed_scale_pixels.rows == 640 &&
              mixed_scale_pixels.columns == 1536 &&
              aima::sha256_bytes(
                  mixed_scale_pixels.values.data(),
                  mixed_scale_pixels.values.size() *
                      sizeof(mixed_scale_pixels.values[0])) ==
                  "632921aa2e03490759370eff2ca8b191e0efd17417cf67516065a21249f6d4e3",
          "mixed-scale BF16 processor oracle drifted");
  const aima::NativeVlResizeGeometry video_geometry =
      aima::native_qwen36_video_geometry(4, 256, 256);
  const aima::NativeVlPixelTensor video_pixels =
      aima::native_qwen36_patchify_resized_rgb(
          deterministic_video(), video_geometry);
  require(video_pixels.rows == 512 && video_pixels.columns == 1536 &&
              aima::sha256_bytes(
                  video_pixels.values.data(),
                  video_pixels.values.size() *
                      sizeof(video_pixels.values[0])) ==
                  "9401d88b9e1d084fe8514f5debecfd69f3997f6ec6bcbe529a8da1409a3638d1",
          "video BF16 temporal/spatial patchify oracle drifted");

  const std::string default_processor_identity =
      aima::native_qwen36_processor_config_sha256();
  require(default_processor_identity ==
              "9be676b2d0cefbe030d61e1d89776df6c7ba28d0d86ca752c60eca3ec60a9280",
          "processor configuration identity is malformed");
  aima::NativeVideoIoPolicy fps_one_identity;
  fps_one_identity.fps = 1.0;
  require(aima::native_qwen36_processor_config_sha256(fps_one_identity) !=
              default_processor_identity,
          "video fps was omitted from processor cache identity");
  aima::NativeVideoIoPolicy frames_six_identity;
  frames_six_identity.num_frames = 6;
  frames_six_identity.fps = -1.0;
  require(aima::native_qwen36_processor_config_sha256(frames_six_identity) !=
              default_processor_identity &&
              aima::native_qwen36_processor_config_sha256(
                  frames_six_identity) !=
                  aima::native_qwen36_processor_config_sha256(
                      fps_one_identity),
          "video frame count was omitted from processor cache identity");
  aima::NativeImageIoPolicy red_background_identity;
  red_background_identity.rgba_background_color = {255, 0, 0};
  require(aima::native_qwen36_processor_config_sha256(
              {}, red_background_identity) != default_processor_identity,
          "RGBA background was omitted from processor cache identity");
  std::cout << "native_vl_processor_test: PASS\n";
  return 0;
}
