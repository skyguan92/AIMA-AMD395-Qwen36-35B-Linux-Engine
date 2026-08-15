// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_image_decoder.h"

#include <jpeglib.h>
#include <png.h>
#include <webp/decode.h>

#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima {
namespace {

std::size_t checked_size(std::uint64_t value, const char* label) {
  if (value == 0 || value > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string(label) + " is outside its range");
  }
  return static_cast<std::size_t>(value);
}

std::size_t decoded_bytes(std::uint64_t width, std::uint64_t height,
                          std::size_t channels,
                          const NativeMediaPolicy& policy) {
  if (policy.maximum_decoded_image_pixels == 0 ||
      policy.maximum_decoded_image_dimension == 0 || width == 0 ||
      height == 0 || width > policy.maximum_decoded_image_dimension ||
      height > policy.maximum_decoded_image_dimension ||
      height > policy.maximum_decoded_image_pixels / width) {
    throw std::invalid_argument(
        "decoded image dimensions exceed the fixed safety boundary");
  }
  const std::uint64_t pixels = width * height;
  if (pixels > policy.maximum_decoded_image_pixels ||
      pixels > std::numeric_limits<std::uint64_t>::max() / channels) {
    throw std::invalid_argument(
        "decoded image dimensions exceed the fixed safety boundary");
  }
  return checked_size(pixels * channels, "decoded image byte count");
}

NativeRgbFrame rgba_to_rgb(std::size_t width, std::size_t height,
                           const unsigned char* rgba,
                           std::size_t rgba_bytes,
                           const NativeImageIoPolicy& image_io) {
  if (rgba == nullptr || rgba_bytes != width * height * 4) {
    throw std::runtime_error("decoded RGBA image has invalid geometry");
  }
  NativeRgbFrame frame;
  frame.width = width;
  frame.height = height;
  frame.pixels.resize(width * height * 3);
  for (std::size_t pixel = 0; pixel < width * height; ++pixel) {
    const std::uint32_t alpha = rgba[pixel * 4 + 3];
    for (std::size_t channel = 0; channel < 3; ++channel) {
      const std::uint32_t source = rgba[pixel * 4 + channel];
      const std::uint32_t background =
          image_io.rgba_background_color[channel];
      frame.pixels[pixel * 3 + channel] = static_cast<unsigned char>(
          (source * alpha + background * (255U - alpha) + 127U) / 255U);
    }
  }
  return frame;
}

NativeRgbFrame decode_png(const NativeMediaPayload& payload,
                          const NativeMediaPolicy& policy) {
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  if (png_image_begin_read_from_memory(&image, payload.bytes.data(),
                                       payload.bytes.size()) == 0) {
    throw std::invalid_argument("PNG image cannot be decoded");
  }
  try {
    const std::size_t rgba_bytes =
        decoded_bytes(image.width, image.height, 4, policy);
    const std::size_t width = image.width;
    const std::size_t height = image.height;
    image.format = PNG_FORMAT_RGBA;
    std::vector<unsigned char> rgba(rgba_bytes);
    if (png_image_finish_read(&image, nullptr, rgba.data(), 0, nullptr) == 0) {
      throw std::invalid_argument("PNG image payload is corrupt");
    }
    png_image_free(&image);
    return rgba_to_rgb(width, height, rgba.data(), rgba.size(),
                       policy.image_io);
  } catch (...) {
    png_image_free(&image);
    throw;
  }
}

NativeRgbFrame decode_webp(const NativeMediaPayload& payload,
                           const NativeMediaPolicy& policy) {
  WebPBitstreamFeatures features{};
  if (WebPGetFeatures(payload.bytes.data(), payload.bytes.size(), &features) !=
      VP8_STATUS_OK) {
    throw std::invalid_argument("WebP image cannot be decoded");
  }
  const std::size_t rgba_bytes = decoded_bytes(
      static_cast<std::uint64_t>(features.width),
      static_cast<std::uint64_t>(features.height), 4, policy);
  std::vector<unsigned char> rgba(rgba_bytes);
  if (WebPDecodeRGBAInto(payload.bytes.data(), payload.bytes.size(), rgba.data(),
                         rgba.size(), features.width * 4) == nullptr) {
    throw std::invalid_argument("WebP image payload is corrupt");
  }
  return rgba_to_rgb(static_cast<std::size_t>(features.width),
                     static_cast<std::size_t>(features.height), rgba.data(),
                     rgba.size(), policy.image_io);
}

struct JpegErrorState {
  jpeg_error_mgr manager;
  std::jmp_buf recovery;
  char message[JMSG_LENGTH_MAX]{};
  jpeg_decompress_struct decoder{};
  bool created = false;
  unsigned char* decoded = nullptr;
};

void jpeg_error_exit(j_common_ptr common) {
  auto* state = reinterpret_cast<JpegErrorState*>(common->err);
  (*common->err->format_message)(common, state->message);
  std::longjmp(state->recovery, 1);
}

void jpeg_emit_message(j_common_ptr common, int message_level) {
  if (message_level >= 0) return;
  // Pillow's default serving path does not enable truncated-image recovery.
  // Treat decoder warnings as invalid input instead of logging or accepting a
  // partially synthesized JPEG.
  jpeg_error_exit(common);
}

NativeRgbFrame decode_jpeg(const NativeMediaPayload& payload,
                           const NativeMediaPolicy& policy) {
  auto* state = new JpegErrorState();
  state->decoder.err = jpeg_std_error(&state->manager);
  state->manager.error_exit = jpeg_error_exit;
  state->manager.emit_message = jpeg_emit_message;
  std::size_t width = 0;
  std::size_t height = 0;
  std::size_t byte_count = 0;
  if (setjmp(state->recovery) != 0) {
    const std::string message(state->message);
    if (state->decoded != nullptr) std::free(state->decoded);
    if (state->created) jpeg_destroy_decompress(&state->decoder);
    delete state;
    throw std::invalid_argument("JPEG image cannot be decoded: " + message);
  }
  try {
    jpeg_create_decompress(&state->decoder);
    state->created = true;
    jpeg_mem_src(&state->decoder, payload.bytes.data(), payload.bytes.size());
    (void)jpeg_read_header(&state->decoder, TRUE);
    state->decoder.out_color_space = JCS_RGB;
    (void)jpeg_start_decompress(&state->decoder);
    width = state->decoder.output_width;
    height = state->decoder.output_height;
    byte_count = decoded_bytes(width, height, 3, policy);
    if (state->decoder.output_components != 3) {
      throw std::invalid_argument("JPEG decoder did not produce RGB pixels");
    }
    state->decoded = static_cast<unsigned char*>(std::malloc(byte_count));
    if (state->decoded == nullptr) throw std::bad_alloc();
    const std::size_t stride = width * 3;
    while (state->decoder.output_scanline < state->decoder.output_height) {
      JSAMPROW row = state->decoded + state->decoder.output_scanline * stride;
      if (jpeg_read_scanlines(&state->decoder, &row, 1) != 1) {
        state->manager.error_exit(
            reinterpret_cast<j_common_ptr>(&state->decoder));
      }
    }
    (void)jpeg_finish_decompress(&state->decoder);
    jpeg_destroy_decompress(&state->decoder);
    state->created = false;
    NativeRgbFrame frame;
    frame.width = width;
    frame.height = height;
    frame.pixels.assign(state->decoded, state->decoded + byte_count);
    std::free(state->decoded);
    state->decoded = nullptr;
    delete state;
    return frame;
  } catch (...) {
    if (state->decoded != nullptr) std::free(state->decoded);
    if (state->created) jpeg_destroy_decompress(&state->decoder);
    delete state;
    throw;
  }
}

}  // namespace

NativeRgbFrame decode_native_image(const NativeMediaPayload& payload,
                                   const NativeMediaPolicy& policy) {
  if (payload.kind != NativeMediaKind::kImage || payload.bytes.empty()) {
    throw std::invalid_argument("native image decoder requires image bytes");
  }
  if (payload.mime_type == "image/png") return decode_png(payload, policy);
  if (payload.mime_type == "image/jpeg") return decode_jpeg(payload, policy);
  if (payload.mime_type == "image/webp") return decode_webp(payload, policy);
  throw std::invalid_argument("native image format is not supported");
}

}  // namespace aima
