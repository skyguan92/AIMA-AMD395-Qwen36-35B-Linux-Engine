// SPDX-License-Identifier: Apache-2.0

#include "aima/native_video_decoder.h"

#include "aima/sha256.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "native_video_decoder_test: " << message << '\n';
    std::exit(1);
  }
}

std::vector<unsigned char> read_bytes(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open video fixture");
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

std::string decoded_sha256(const aima::NativeDecodedVideo &video) {
  std::vector<unsigned char> bytes;
  for (const aima::NativeRgbFrame &frame : video.frames) {
    bytes.insert(bytes.end(), frame.pixels.begin(), frame.pixels.end());
  }
  return aima::sha256_bytes(bytes.data(), bytes.size());
}

aima::NativeMediaPayload payload(const std::filesystem::path &root,
                                 const char *filename, const char *mime) {
  aima::NativeMediaPayload result;
  result.kind = aima::NativeMediaKind::kVideo;
  result.mime_type = mime;
  result.bytes = read_bytes(root / filename);
  return result;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: native_video_decoder_test FIXTURE_ROOT\n";
    return 2;
  }
  const std::filesystem::path root(argv[1]);
  const aima::NativeMediaPolicy policy;

  const aima::NativeDecodedVideo mp4 = aima::decode_native_video(
      payload(root, "video-8f-4fps-128.mp4", "video/mp4"), policy);
  require(mp4.total_frames == 8 && mp4.source_fps == 4.0 &&
              mp4.duration_seconds == 2.0 && mp4.width == 128 &&
              mp4.height == 128 &&
              mp4.frame_indices == std::vector<std::size_t>({0, 2, 4, 7}) &&
              decoded_sha256(mp4) == "677912ce524c3324877e4ba9f81ba871bfa5a4f5a"
                                     "4b35821ff7fb7528c336ae7",
          "MP4 OpenCV-compatible decode oracle drifted");
  const aima::NativeVlResizeGeometry mp4_geometry =
      aima::native_qwen36_video_geometry(mp4.frames.size(), mp4.height,
                                         mp4.width);
  const aima::NativeVlPixelTensor mp4_pixels =
      aima::native_qwen36_process_rgb(mp4.frames, mp4_geometry);
  require(mp4_pixels.rows == 128 && mp4_pixels.columns == 1536 &&
              aima::sha256_bytes(mp4_pixels.values.data(),
                                 mp4_pixels.values.size() *
                                     sizeof(mp4_pixels.values[0])) ==
                  "0f0806b68060aefe416b704fec0b2f8d2cb6129bbc9b5998e956b38a775c"
                  "dca0",
          "MP4 decode-to-BF16 processor oracle drifted");
  const std::string mp4_prompt = aima::native_qwen36_expand_media_prompt(
      "A<|vision_start|><|video_pad|><|vision_end|>B",
      {{aima::NativeMediaKind::kVideo, mp4_geometry.grid, mp4.frame_indices,
        mp4.source_fps}});
  require(mp4_prompt.find("<0.2 seconds>") != std::string::npos &&
              mp4_prompt.find("<1.4 seconds>") != std::string::npos &&
              mp4_prompt.find("<1.5 seconds>") == std::string::npos,
          "MP4 sampled-frame timestamps drifted");

  const aima::NativeDecodedVideo avi = aima::decode_native_video(
      payload(root, "video-12f-6fps-192x128.avi", "video/x-msvideo"), policy);
  require(avi.total_frames == 12 && avi.source_fps == 6.0 &&
              avi.duration_seconds == 2.0 && avi.width == 192 &&
              avi.height == 128 &&
              avi.frame_indices == std::vector<std::size_t>({0, 3, 7, 11}) &&
              decoded_sha256(avi) == "108baa268c1fe97c4ef5fa97d71f48af67f4916d7"
                                     "51add0ef37e4b58e5301abd",
          "AVI OpenCV-compatible decode oracle drifted");
  const aima::NativeVlResizeGeometry avi_geometry =
      aima::native_qwen36_video_geometry(avi.frames.size(), avi.height,
                                         avi.width);
  const aima::NativeVlPixelTensor avi_pixels =
      aima::native_qwen36_process_rgb(avi.frames, avi_geometry);
  require(avi_pixels.rows == 192 && avi_pixels.columns == 1536 &&
              aima::sha256_bytes(avi_pixels.values.data(),
                                 avi_pixels.values.size() *
                                     sizeof(avi_pixels.values[0])) ==
                  "9e9efee715d55eeaa2d06458d9c2425fe9e9923920c0f4bdf2a1b0479f96"
                  "0b1e",
          "AVI decode-to-BF16 processor oracle drifted");

  try {
    (void)aima::decode_native_video(
        payload(root, "corrupt-video.mp4", "video/mp4"), policy);
    require(false, "corrupt MP4 was decoded");
  } catch (const std::invalid_argument &) {
  }

  aima::NativeMediaPolicy small_dimension = policy;
  small_dimension.maximum_decoded_video_dimension = 64;
  try {
    (void)aima::decode_native_video(
        payload(root, "video-8f-4fps-128.mp4", "video/mp4"), small_dimension);
    require(false, "decoded video dimension boundary was ignored");
  } catch (const std::invalid_argument &) {
  }

  aima::NativeMediaPolicy small_frame_count = policy;
  small_frame_count.maximum_video_source_frames = 7;
  try {
    (void)aima::decode_native_video(
        payload(root, "video-8f-4fps-128.mp4", "video/mp4"), small_frame_count);
    require(false, "video source-frame boundary was ignored");
  } catch (const std::invalid_argument &) {
  }

  aima::NativeMediaPolicy small_sample_count = policy;
  small_sample_count.maximum_video_sampled_frames = 3;
  try {
    (void)aima::decode_native_video(
        payload(root, "video-8f-4fps-128.mp4", "video/mp4"),
        small_sample_count);
    require(false, "video sampled-frame boundary was ignored");
  } catch (const std::invalid_argument &) {
  }

  aima::NativeMediaPolicy small_duration = policy;
  small_duration.maximum_video_duration_seconds = 1.0;
  try {
    (void)aima::decode_native_video(
        payload(root, "video-8f-4fps-128.mp4", "video/mp4"), small_duration);
    require(false, "video duration boundary was ignored");
  } catch (const std::invalid_argument &) {
  }

  aima::NativeMediaPolicy small_pixels = policy;
  small_pixels.maximum_decoded_video_pixels = 4 * 128 * 128 - 1;
  try {
    (void)aima::decode_native_video(
        payload(root, "video-8f-4fps-128.mp4", "video/mp4"), small_pixels);
    require(false, "decoded video pixel boundary was ignored");
  } catch (const std::invalid_argument &) {
  }

  aima::NativeMediaPolicy invalid_timeout = policy;
  invalid_timeout.maximum_video_decode_milliseconds = 0;
  try {
    (void)aima::decode_native_video(
        payload(root, "video-8f-4fps-128.mp4", "video/mp4"), invalid_timeout);
    require(false, "invalid video timeout policy was admitted");
  } catch (const std::invalid_argument &) {
  }

  std::cout << "native_video_decoder_test: PASS\n";
  return 0;
}
