// SPDX-License-Identifier: Apache-2.0

#include "aima/native_image_decoder.h"

#include "aima/sha256.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "native_image_decoder_test: " << message << '\n';
    std::exit(1);
  }
}

std::vector<unsigned char> read_bytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open image fixture");
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

void require_fixture(const std::filesystem::path& root, const char* filename,
                     const char* mime, std::size_t width, std::size_t height,
                     const char* expected_sha256) {
  aima::NativeMediaPayload payload;
  payload.kind = aima::NativeMediaKind::kImage;
  payload.mime_type = mime;
  payload.bytes = read_bytes(root / filename);
  const aima::NativeRgbFrame decoded =
      aima::decode_native_image(payload, {});
  require(decoded.width == width && decoded.height == height &&
              decoded.pixels.size() == width * height * 3 &&
              aima::sha256_bytes(decoded.pixels.data(),
                                 decoded.pixels.size()) == expected_sha256,
          filename);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: native_image_decoder_test FIXTURE_ROOT\n";
    return 2;
  }
  const std::filesystem::path root(argv[1]);
  require_fixture(
      root, "image-rgb-256.png", "image/png", 256, 256,
      "4e93072f3f85d09ab13dbd8e1a79253dbeed33cc887a0745a84a1f378c2f9d06");
  require_fixture(
      root, "image-landscape-512x192.jpg", "image/jpeg", 512, 192,
      "749727157d96fd1516956d32f612e55541eff1380a86c27d940ad5d7b35c229f");
  require_fixture(
      root, "image-transparent-160x320.png", "image/png", 160, 320,
      "36c7b22d4f186e88d12b44ee170c7a468cd7ef37be0f1ea0aeb6614785dde222");
  require_fixture(
      root, "image-portrait-192x512.webp", "image/webp", 192, 512,
      "824badf1c9a15777d77c544dad0da03ab67ddd310767eba4c9a409a5edba7bda");

  aima::NativeMediaPayload transparent;
  transparent.kind = aima::NativeMediaKind::kImage;
  transparent.mime_type = "image/png";
  transparent.bytes = read_bytes(root / "image-transparent-160x320.png");
  const aima::NativeRgbFrame decoded_transparent =
      aima::decode_native_image(transparent, {});
  const aima::NativeVlResizeGeometry transparent_geometry =
      aima::native_qwen36_image_geometry(decoded_transparent.height,
                                         decoded_transparent.width);
  const aima::NativeRgbFrame resized_transparent =
      aima::native_qwen36_resize_rgb(decoded_transparent,
                                     transparent_geometry);
  require(resized_transparent.height == 384 &&
              resized_transparent.width == 192 &&
              aima::sha256_bytes(resized_transparent.pixels.data(),
                                 resized_transparent.pixels.size()) ==
                  "d34742d3899ae42a769b80fde2e1da59a9158da72caf2a5f0dca014d58955857",
          "torchvision v2 uint8 bicubic resize oracle drifted");
  const aima::NativeVlPixelTensor processed_transparent =
      aima::native_qwen36_process_rgb({decoded_transparent},
                                      transparent_geometry);
  require(processed_transparent.rows == 288 &&
              processed_transparent.columns == 1536 &&
              aima::sha256_bytes(
                  processed_transparent.values.data(),
                  processed_transparent.values.size() *
                      sizeof(processed_transparent.values[0])) ==
                  "dd3835f0d61fc4f17576fd09a22759e79671e6bca09cbc014efd282b8dd0fbce",
          "resized image BF16 processor oracle drifted");

  aima::NativeMediaPayload png;
  png.kind = aima::NativeMediaKind::kImage;
  png.mime_type = "image/png";
  png.bytes = read_bytes(root / "image-rgb-256.png");
  const aima::NativeRgbFrame decoded_png =
      aima::decode_native_image(png, {});
  const aima::NativeVlPixelTensor processed_png =
      aima::native_qwen36_process_rgb(
          {decoded_png}, aima::native_qwen36_image_geometry(
                             decoded_png.height, decoded_png.width));
  require(aima::sha256_bytes(
              processed_png.values.data(),
              processed_png.values.size() *
                  sizeof(processed_png.values[0])) ==
              "28e3bf47e74e94a78db819016eee9ce02983f93ab86012de846a27d72a1623b8",
          "PNG decode-to-BF16 processor oracle drifted");

  aima::NativeMediaPolicy small_dimension;
  small_dimension.maximum_decoded_image_dimension = 128;
  try {
    (void)aima::decode_native_image(png, small_dimension);
    require(false, "decoded image dimension boundary was ignored");
  } catch (const std::invalid_argument&) {
  }

  aima::NativeMediaPayload corrupt;
  corrupt.kind = aima::NativeMediaKind::kImage;
  corrupt.mime_type = "image/png";
  corrupt.bytes = read_bytes(root / "corrupt-image.png");
  try {
    (void)aima::decode_native_image(corrupt, {});
    require(false, "corrupt PNG was decoded");
  } catch (const std::invalid_argument&) {
  }

  aima::NativeMediaPayload truncated_jpeg;
  truncated_jpeg.kind = aima::NativeMediaKind::kImage;
  truncated_jpeg.mime_type = "image/jpeg";
  truncated_jpeg.bytes = read_bytes(root / "image-landscape-512x192.jpg");
  truncated_jpeg.bytes.resize(32);
  try {
    (void)aima::decode_native_image(truncated_jpeg, {});
    require(false, "truncated JPEG was decoded");
  } catch (const std::invalid_argument&) {
  }

  std::cout << "native_image_decoder_test: PASS\n";
  return 0;
}
