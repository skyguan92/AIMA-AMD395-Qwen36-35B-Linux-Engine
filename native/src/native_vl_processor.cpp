// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vl_processor.h"

#include "aima/sha256.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace aima {
namespace {

constexpr std::size_t kFactor = kNativeVlPatchSize * kNativeVlMergeSize;
constexpr std::size_t kImageMinimumPixels = 65536;
constexpr std::size_t kImageMaximumPixels = 16777216;
constexpr std::size_t kVideoMinimumPixels = 4096;
constexpr std::size_t kVideoMaximumPixels = 25165824;
constexpr std::size_t kVideoMinimumFrames = 4;
constexpr std::size_t kVideoMaximumFrames = 768;
constexpr double kVideoDefaultFps = 2.0;
constexpr std::string_view kVisionStart = "<|vision_start|>";
constexpr std::string_view kVisionEnd = "<|vision_end|>";
constexpr std::string_view kImagePad = "<|image_pad|>";
constexpr std::string_view kVideoPad = "<|video_pad|>";
constexpr std::size_t kRgbChannels = 3;

std::size_t checked_product(std::size_t left, std::size_t right,
                            const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::invalid_argument(std::string(label) + " overflows");
  }
  return left * right;
}

// Python round() and NumPy rint use ties-to-even. std::round instead rounds
// ties away from zero, which changes frozen resize and sampling boundaries.
std::size_t round_half_to_even(double value) {
  if (!std::isfinite(value) || value < 0.0 ||
      value > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    throw std::invalid_argument("VL rounding input is outside its domain");
  }
  const double lower_value = std::floor(value);
  const double fraction = value - lower_value;
  std::size_t lower = static_cast<std::size_t>(lower_value);
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 == 0 ? lower : lower + 1;
}

std::size_t factor_round(std::size_t value) {
  return checked_product(
      round_half_to_even(static_cast<double>(value) /
                         static_cast<double>(kFactor)),
      kFactor, "rounded VL dimension");
}

std::size_t scaled_floor(std::size_t value, double beta) {
  return std::max(
      kFactor,
      checked_product(
          static_cast<std::size_t>(
              std::floor(static_cast<double>(value) / beta / kFactor)),
          kFactor, "downscaled VL dimension"));
}

std::size_t scaled_ceil(std::size_t value, double beta) {
  return checked_product(
      static_cast<std::size_t>(
          std::ceil(static_cast<double>(value) * beta / kFactor)),
      kFactor, "upscaled VL dimension");
}

void require_spatial_source(std::size_t height, std::size_t width) {
  if (height == 0 || width == 0) {
    throw std::invalid_argument("VL media dimensions must be positive");
  }
  const double ratio = static_cast<double>(std::max(height, width)) /
                       static_cast<double>(std::min(height, width));
  if (ratio > 200.0) {
    throw std::invalid_argument(
        "absolute media aspect ratio must not exceed 200");
  }
}

NativeVlResizeGeometry geometry(std::size_t frames, std::size_t height,
                                std::size_t width, std::size_t minimum_pixels,
                                std::size_t maximum_pixels, bool video) {
  require_spatial_source(height, width);
  if (video && (frames < kNativeVlTemporalPatchSize || height < kFactor ||
                width < kFactor)) {
    throw std::invalid_argument(
        "video is below the frozen temporal or spatial factor");
  }
  std::size_t resized_height = factor_round(height);
  std::size_t resized_width = factor_round(width);
  const std::size_t temporal_rounded =
      video ? checked_product(
                  round_half_to_even(static_cast<double>(frames) /
                                     kNativeVlTemporalPatchSize),
                  kNativeVlTemporalPatchSize,
                  "rounded VL temporal dimension")
            : 1;
  const std::size_t rounded_pixels = checked_product(
      checked_product(temporal_rounded, resized_height,
                      "rounded VL feature shape"),
      resized_width, "rounded VL feature shape");
  const double source_pixels = static_cast<double>(frames) *
                               static_cast<double>(height) *
                               static_cast<double>(width);
  if (rounded_pixels > maximum_pixels) {
    const double beta = std::sqrt(source_pixels / maximum_pixels);
    resized_height = scaled_floor(height, beta);
    resized_width = scaled_floor(width, beta);
  } else if (rounded_pixels < minimum_pixels) {
    const double beta = std::sqrt(minimum_pixels / source_pixels);
    resized_height = scaled_ceil(height, beta);
    resized_width = scaled_ceil(width, beta);
  }
  NativeVlResizeGeometry result;
  result.resized_height = resized_height;
  result.resized_width = resized_width;
  result.grid.temporal =
      frames / kNativeVlTemporalPatchSize +
      static_cast<std::size_t>(frames % kNativeVlTemporalPatchSize != 0);
  result.grid.height = resized_height / kNativeVlPatchSize;
  result.grid.width = resized_width / kNativeVlPatchSize;
  (void)result.grid.language_token_count();
  return result;
}

void replace_first(std::string& value, std::string_view needle,
                   const std::string& replacement, const char* label) {
  const std::size_t position = value.find(needle);
  if (position == std::string::npos) {
    throw std::invalid_argument(std::string("prompt has no marker for ") +
                                label);
  }
  value.replace(position, needle.size(), replacement);
}

std::string repeated(std::string_view value, std::size_t count) {
  if (count != 0 && value.size() >
                        std::numeric_limits<std::size_t>::max() / count) {
    throw std::invalid_argument("expanded VL prompt is too large");
  }
  std::string result;
  result.reserve(value.size() * count);
  for (std::size_t index = 0; index < count; ++index) result.append(value);
  return result;
}

std::uint16_t float_to_bfloat16(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t bias = 0x7fffU + ((bits >> 16U) & 1U);
  bits += bias;
  return static_cast<std::uint16_t>(bits >> 16U);
}

// This is the Keys cubic kernel used by torchvision v2 for bicubic resize
// with antialiasing. Its a=-0.5 coefficient and the fixed-point conversion
// below are frozen against PyTorch git8514f05.
double bicubic_antialias_filter(double value) {
  constexpr double coefficient = -0.5;
  const double distance = std::abs(value);
  if (distance < 1.0) {
    return ((coefficient + 2.0) * distance - (coefficient + 3.0)) *
               distance * distance +
           1.0;
  }
  if (distance < 2.0) {
    return ((coefficient * distance - 5.0 * coefficient) * distance +
            8.0 * coefficient) *
               distance -
           4.0 * coefficient;
  }
  return 0.0;
}

struct ResizeAxisWeights {
  std::size_t input_size = 0;
  std::size_t output_size = 0;
  std::size_t weight_stride = 0;
  unsigned int precision = 0;
  std::vector<std::size_t> starts;
  std::vector<std::size_t> sizes;
  std::vector<std::int16_t> weights;
};

ResizeAxisWeights make_resize_axis_weights(std::size_t input_size,
                                           std::size_t output_size) {
  if (input_size == 0 || output_size == 0 ||
      input_size > static_cast<std::size_t>(
                       std::numeric_limits<std::int64_t>::max()) ||
      output_size > static_cast<std::size_t>(
                        std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("VL resize axis is outside its domain");
  }
  const double scale = static_cast<double>(input_size) /
                       static_cast<double>(output_size);
  const double support = scale >= 1.0 ? 2.0 * scale : 2.0;
  const double maximum_taps_value = std::ceil(support) * 2.0 + 1.0;
  if (!std::isfinite(maximum_taps_value) || maximum_taps_value < 1.0 ||
      maximum_taps_value >
          static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    throw std::invalid_argument("VL resize support is outside its domain");
  }
  const std::size_t maximum_taps =
      static_cast<std::size_t>(maximum_taps_value);
  const std::size_t floating_weight_count =
      checked_product(output_size, maximum_taps, "VL resize weights");
  std::vector<double> floating_weights(floating_weight_count, 0.0);

  ResizeAxisWeights result;
  result.input_size = input_size;
  result.output_size = output_size;
  result.starts.resize(output_size);
  result.sizes.resize(output_size);
  double maximum_weight = 0.0;
  const double inverse_scale = scale >= 1.0 ? 1.0 / scale : 1.0;
  for (std::size_t output = 0; output < output_size; ++output) {
    const double center = scale * (static_cast<double>(output) + 0.5);
    const std::int64_t lower = static_cast<std::int64_t>(
        center - support + 0.5);
    const std::int64_t upper = static_cast<std::int64_t>(
        center + support + 0.5);
    const std::int64_t start = std::max<std::int64_t>(lower, 0);
    const std::int64_t stop = std::min<std::int64_t>(
        upper, static_cast<std::int64_t>(input_size));
    const std::size_t size = static_cast<std::size_t>(std::clamp<
        std::int64_t>(stop - start, 0,
                      static_cast<std::int64_t>(maximum_taps)));
    result.starts[output] = static_cast<std::size_t>(start);
    result.sizes[output] = size;

    double total_weight = 0.0;
    double* const values =
        floating_weights.data() + output * maximum_taps;
    for (std::size_t tap = 0; tap < size; ++tap) {
      const double distance =
          (static_cast<double>(tap) + static_cast<double>(start) - center +
           0.5) *
          inverse_scale;
      values[tap] = bicubic_antialias_filter(distance);
      total_weight += values[tap];
    }
    if (total_weight != 0.0) {
      for (std::size_t tap = 0; tap < size; ++tap) {
        values[tap] /= total_weight;
        maximum_weight = std::max(maximum_weight, values[tap]);
      }
    }
  }

  for (result.precision = 0; result.precision < 22;
       ++result.precision) {
    const std::uint64_t scale_factor =
        std::uint64_t{1} << (result.precision + 1U);
    const int next_value = static_cast<int>(
        0.5 + maximum_weight * static_cast<double>(scale_factor));
    if (next_value >= (1 << 15)) break;
  }
  if (result.precision == 0 || result.precision >= 22) {
    throw std::runtime_error("VL resize weight precision is invalid");
  }

  // torchvision aligns the int16 coefficient rows to four entries for its
  // vectorized path. The padding is zero and has no numerical contribution.
  result.weight_stride = maximum_taps;
  while (result.weight_stride % sizeof(std::int32_t) != 0) {
    ++result.weight_stride;
  }
  result.weights.assign(
      checked_product(output_size, result.weight_stride,
                      "VL fixed resize weights"),
      0);
  const double fixed_scale = static_cast<double>(
      std::uint64_t{1} << result.precision);
  for (std::size_t output = 0; output < output_size; ++output) {
    const double* const source =
        floating_weights.data() + output * maximum_taps;
    std::int16_t* const destination =
        result.weights.data() + output * result.weight_stride;
    for (std::size_t tap = 0; tap < maximum_taps; ++tap) {
      const double scaled = source[tap] * fixed_scale;
      const int rounded = scaled < 0.0
                              ? static_cast<int>(scaled - 0.5)
                              : static_cast<int>(scaled + 0.5);
      if (rounded < std::numeric_limits<std::int16_t>::min() ||
          rounded > std::numeric_limits<std::int16_t>::max()) {
        throw std::runtime_error("VL resize coefficient overflows int16");
      }
      destination[tap] = static_cast<std::int16_t>(rounded);
    }
  }
  return result;
}

unsigned char fixed_resize_sample(std::int64_t accumulated,
                                  unsigned int precision) {
  const std::int64_t divisor = std::int64_t{1} << precision;
  const std::int64_t shifted =
      accumulated >= 0
          ? accumulated / divisor
          : -((-accumulated + divisor - 1) / divisor);
  return static_cast<unsigned char>(
      std::clamp<std::int64_t>(shifted, 0, 255));
}

NativeRgbFrame resize_width(const NativeRgbFrame& frame,
                            std::size_t output_width) {
  if (frame.width == output_width) return frame;
  const ResizeAxisWeights axis =
      make_resize_axis_weights(frame.width, output_width);
  NativeRgbFrame output;
  output.height = frame.height;
  output.width = output_width;
  output.pixels.resize(checked_product(
      checked_product(output.height, output.width, "VL resized frame"),
      kRgbChannels, "VL resized frame"));
  for (std::size_t y = 0; y < output.height; ++y) {
    for (std::size_t x = 0; x < output.width; ++x) {
      const std::size_t start = axis.starts[x];
      const std::int16_t* const weights =
          axis.weights.data() + x * axis.weight_stride;
      for (std::size_t channel = 0; channel < kRgbChannels; ++channel) {
        std::int64_t accumulated =
            std::int64_t{1} << (axis.precision - 1U);
        for (std::size_t tap = 0; tap < axis.sizes[x]; ++tap) {
          const std::size_t source =
              (y * frame.width + start + tap) * kRgbChannels + channel;
          accumulated += static_cast<std::int64_t>(frame.pixels[source]) *
                         weights[tap];
        }
        output.pixels[(y * output.width + x) * kRgbChannels + channel] =
            fixed_resize_sample(accumulated, axis.precision);
      }
    }
  }
  return output;
}

NativeRgbFrame resize_height(const NativeRgbFrame& frame,
                             std::size_t output_height) {
  if (frame.height == output_height) return frame;
  const ResizeAxisWeights axis =
      make_resize_axis_weights(frame.height, output_height);
  NativeRgbFrame output;
  output.height = output_height;
  output.width = frame.width;
  output.pixels.resize(checked_product(
      checked_product(output.height, output.width, "VL resized frame"),
      kRgbChannels, "VL resized frame"));
  for (std::size_t y = 0; y < output.height; ++y) {
    const std::size_t start = axis.starts[y];
    const std::int16_t* const weights =
        axis.weights.data() + y * axis.weight_stride;
    for (std::size_t x = 0; x < output.width; ++x) {
      for (std::size_t channel = 0; channel < kRgbChannels; ++channel) {
        std::int64_t accumulated =
            std::int64_t{1} << (axis.precision - 1U);
        for (std::size_t tap = 0; tap < axis.sizes[y]; ++tap) {
          const std::size_t source =
              ((start + tap) * frame.width + x) * kRgbChannels + channel;
          accumulated += static_cast<std::int64_t>(frame.pixels[source]) *
                         weights[tap];
        }
        output.pixels[(y * output.width + x) * kRgbChannels + channel] =
            fixed_resize_sample(accumulated, axis.precision);
      }
    }
  }
  return output;
}

}  // namespace

std::size_t NativeVlGrid::patch_count() const {
  if (temporal == 0 || height == 0 || width == 0) {
    throw std::invalid_argument("VL grid dimensions must be positive");
  }
  return checked_product(checked_product(temporal, height, "VL grid"),
                         width, "VL grid");
}

std::size_t NativeVlGrid::language_token_count() const {
  const std::size_t patches = patch_count();
  constexpr std::size_t merge_area =
      kNativeVlMergeSize * kNativeVlMergeSize;
  if (height % kNativeVlMergeSize != 0 ||
      width % kNativeVlMergeSize != 0 || patches % merge_area != 0) {
    throw std::invalid_argument("VL grid is not spatial-merge aligned");
  }
  return patches / merge_area;
}

NativeVlResizeGeometry native_qwen36_image_geometry(
    std::size_t source_height, std::size_t source_width) {
  NativeVlResizeGeometry result =
      geometry(1, source_height, source_width, kImageMinimumPixels,
               kImageMaximumPixels, false);
  if (result.grid.language_token_count() > kNativeVlImageTokenLimit) {
    throw std::invalid_argument("image exceeds the frozen token limit");
  }
  return result;
}

NativeVlResizeGeometry native_qwen36_video_geometry(
    std::size_t sampled_frames, std::size_t source_height,
    std::size_t source_width) {
  NativeVlResizeGeometry result =
      geometry(sampled_frames, source_height, source_width,
               kVideoMinimumPixels, kVideoMaximumPixels, true);
  if (result.grid.language_token_count() > kNativeVlVideoTokenLimit) {
    throw std::invalid_argument("video exceeds the frozen token limit");
  }
  return result;
}

std::vector<std::size_t> native_qwen36_sample_video_frames(
    std::size_t total_frames, double source_fps,
    const NativeVlVideoSamplingOptions& options) {
  if (total_frames == 0 || !std::isfinite(source_fps) || source_fps <= 0.0) {
    throw std::invalid_argument("video sampling metadata is invalid");
  }
  if (options.num_frames.has_value() && options.fps.has_value()) {
    throw std::invalid_argument(
        "num_frames and fps are mutually exclusive");
  }
  std::size_t count = 0;
  if (options.num_frames.has_value()) {
    count = *options.num_frames;
    if (count == 0 || count > kVideoMaximumFrames) {
      throw std::invalid_argument(
          "video num_frames is outside the fixed serving range");
    }
  } else {
    const double target_fps = options.fps.value_or(kVideoDefaultFps);
    if (!std::isfinite(target_fps) || target_fps <= 0.0) {
      throw std::invalid_argument("video target fps must be positive");
    }
    const double requested = static_cast<double>(total_frames) / source_fps *
                             target_fps;
    if (!std::isfinite(requested) ||
        requested >
            static_cast<double>(std::numeric_limits<std::size_t>::max())) {
      throw std::invalid_argument("video sampling count is outside its domain");
    }
    count = static_cast<std::size_t>(requested);
    count = std::min(std::max(count, kVideoMinimumFrames),
                     kVideoMaximumFrames);
    count = std::min(count, total_frames);
  }
  // The explicit num_frames path in the reference relies on linspace and does
  // not independently clamp to total_frames.
  std::vector<std::size_t> indices(count);
  if (count == 1) return indices;
  const double step = static_cast<double>(total_frames - 1) /
                      static_cast<double>(count - 1);
  for (std::size_t index = 0; index < count; ++index) {
    indices[index] = round_half_to_even(static_cast<double>(index) * step);
  }
  return indices;
}

std::vector<double> native_qwen36_video_timestamps(
    const std::vector<std::size_t>& frame_indices, double source_fps) {
  if (frame_indices.empty() || !std::isfinite(source_fps) ||
      source_fps <= 0.0) {
    throw std::invalid_argument("video timestamp metadata is invalid");
  }
  std::vector<std::size_t> padded = frame_indices;
  if (padded.size() % kNativeVlMergeSize != 0) {
    padded.push_back(padded.back());
  }
  std::vector<double> timestamps;
  timestamps.reserve(padded.size() / kNativeVlMergeSize);
  for (std::size_t index = 0; index < padded.size();
       index += kNativeVlMergeSize) {
    timestamps.push_back(
        (static_cast<double>(padded[index]) / source_fps +
         static_cast<double>(padded[index + kNativeVlMergeSize - 1]) /
             source_fps) /
        2.0);
  }
  return timestamps;
}

NativeVlPixelTensor native_qwen36_patchify_resized_rgb(
    const std::vector<NativeRgbFrame>& frames,
    const NativeVlResizeGeometry& geometry) {
  if (frames.empty()) {
    throw std::invalid_argument("VL patchify requires at least one frame");
  }
  if (geometry.resized_height == 0 || geometry.resized_width == 0 ||
      geometry.resized_height % kNativeVlPatchSize != 0 ||
      geometry.resized_width % kNativeVlPatchSize != 0) {
    throw std::invalid_argument("VL patchify geometry is not patch aligned");
  }
  const std::size_t frame_pixels = checked_product(
      checked_product(geometry.resized_height, geometry.resized_width,
                      "VL RGB frame"),
      3, "VL RGB frame");
  for (const NativeRgbFrame& frame : frames) {
    if (frame.height != geometry.resized_height ||
        frame.width != geometry.resized_width ||
        frame.pixels.size() != frame_pixels) {
      throw std::invalid_argument(
          "VL patchify frame does not match resized geometry");
    }
  }
  const std::size_t padded_frames =
      frames.size() + static_cast<std::size_t>(
                          frames.size() % kNativeVlTemporalPatchSize != 0);
  NativeVlGrid expected_grid{
      padded_frames / kNativeVlTemporalPatchSize,
      geometry.resized_height / kNativeVlPatchSize,
      geometry.resized_width / kNativeVlPatchSize,
  };
  if (geometry.grid.temporal != expected_grid.temporal ||
      geometry.grid.height != expected_grid.height ||
      geometry.grid.width != expected_grid.width) {
    throw std::invalid_argument(
        "VL patchify frames do not match the declared grid");
  }
  constexpr std::size_t channels = 3;
  constexpr std::size_t columns =
      channels * kNativeVlTemporalPatchSize * kNativeVlPatchSize *
      kNativeVlPatchSize;
  NativeVlPixelTensor output;
  output.grid = geometry.grid;
  output.rows = output.grid.patch_count();
  output.columns = columns;
  output.values.resize(
      checked_product(output.rows, output.columns, "VL pixel tensor"));

  std::size_t destination = 0;
  const std::size_t height_groups =
      output.grid.height / kNativeVlMergeSize;
  const std::size_t width_groups =
      output.grid.width / kNativeVlMergeSize;
  for (std::size_t temporal = 0; temporal < output.grid.temporal;
       ++temporal) {
    for (std::size_t height_group = 0; height_group < height_groups;
         ++height_group) {
      for (std::size_t width_group = 0; width_group < width_groups;
           ++width_group) {
        for (std::size_t merge_height = 0;
             merge_height < kNativeVlMergeSize; ++merge_height) {
          for (std::size_t merge_width = 0;
               merge_width < kNativeVlMergeSize; ++merge_width) {
            for (std::size_t channel = 0; channel < channels; ++channel) {
              for (std::size_t temporal_patch = 0;
                   temporal_patch < kNativeVlTemporalPatchSize;
                   ++temporal_patch) {
                const std::size_t logical_frame =
                    temporal * kNativeVlTemporalPatchSize + temporal_patch;
                const NativeRgbFrame& frame =
                    frames[std::min(logical_frame, frames.size() - 1)];
                for (std::size_t patch_height = 0;
                     patch_height < kNativeVlPatchSize; ++patch_height) {
                  const std::size_t source_y =
                      (height_group * kNativeVlMergeSize + merge_height) *
                          kNativeVlPatchSize +
                      patch_height;
                  for (std::size_t patch_width = 0;
                       patch_width < kNativeVlPatchSize; ++patch_width) {
                    const std::size_t source_x =
                        (width_group * kNativeVlMergeSize + merge_width) *
                            kNativeVlPatchSize +
                        patch_width;
                    const std::size_t source =
                        (source_y * frame.width + source_x) * channels +
                        channel;
                    const float normalized =
                        (static_cast<float>(frame.pixels[source]) - 127.5F) /
                        127.5F;
                    output.values[destination++] =
                        float_to_bfloat16(normalized);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if (destination != output.values.size()) {
    throw std::runtime_error("VL patchify tensor accounting failed");
  }
  return output;
}

NativeRgbFrame native_qwen36_resize_rgb(
    const NativeRgbFrame& frame,
    const NativeVlResizeGeometry& geometry) {
  if (frame.height == 0 || frame.width == 0 ||
      frame.pixels.size() !=
          checked_product(checked_product(frame.height, frame.width,
                                          "VL source RGB frame"),
                          kRgbChannels, "VL source RGB frame")) {
    throw std::invalid_argument("VL resize source frame is malformed");
  }
  if (geometry.resized_height == 0 || geometry.resized_width == 0) {
    throw std::invalid_argument("VL resize target dimensions must be positive");
  }
  // The frozen torchvision implementation is separable and performs the
  // horizontal uint8 pass before the vertical uint8 pass.
  return resize_height(resize_width(frame, geometry.resized_width),
                       geometry.resized_height);
}

NativeVlPixelTensor native_qwen36_process_rgb(
    const std::vector<NativeRgbFrame>& frames,
    const NativeVlResizeGeometry& geometry) {
  if (frames.empty()) {
    throw std::invalid_argument("VL processing requires at least one frame");
  }
  std::vector<NativeRgbFrame> resized;
  resized.reserve(frames.size());
  for (const NativeRgbFrame& frame : frames) {
    resized.push_back(native_qwen36_resize_rgb(frame, geometry));
  }
  return native_qwen36_patchify_resized_rgb(resized, geometry);
}

std::string native_qwen36_expand_media_prompt(
    std::string prompt, const std::vector<NativeVlPromptMedia>& media) {
  std::size_t aggregate_tokens = 0;
  for (const NativeVlPromptMedia& item : media) {
    const std::size_t token_count = item.grid.language_token_count();
    if (token_count > kNativeVlAggregateTokenLimit ||
        aggregate_tokens > kNativeVlAggregateTokenLimit - token_count) {
      throw std::invalid_argument("aggregate VL token budget is exceeded");
    }
    if (item.kind == NativeMediaKind::kImage) {
      if (token_count > kNativeVlImageTokenLimit) {
        throw std::invalid_argument("image exceeds the frozen token limit");
      }
      replace_first(prompt, kImagePad, repeated(kImagePad, token_count),
                    "image");
    } else {
      if (token_count > kNativeVlVideoTokenLimit ||
          item.grid.height % kNativeVlMergeSize != 0 ||
          item.grid.width % kNativeVlMergeSize != 0) {
        throw std::invalid_argument("video grid exceeds its frozen contract");
      }
      const std::vector<double> timestamps =
          native_qwen36_video_timestamps(item.frame_indices, item.source_fps);
      if (timestamps.size() != item.grid.temporal) {
        throw std::invalid_argument(
            "video timestamps do not match the processed temporal grid");
      }
      const std::size_t tokens_per_frame =
          item.grid.height * item.grid.width /
          (kNativeVlMergeSize * kNativeVlMergeSize);
      std::string replacement;
      for (double timestamp : timestamps) {
        std::ostringstream time;
        time.imbue(std::locale::classic());
        time << '<' << std::fixed << std::setprecision(1) << timestamp
             << " seconds>";
        replacement += time.str();
        replacement.append(kVisionStart);
        replacement += repeated(kVideoPad, tokens_per_frame);
        replacement.append(kVisionEnd);
      }
      const std::string wrapped = std::string(kVisionStart) +
                                  std::string(kVideoPad) +
                                  std::string(kVisionEnd);
      const std::size_t wrapped_position = prompt.find(wrapped);
      if (wrapped_position != std::string::npos) {
        prompt.replace(wrapped_position, wrapped.size(), replacement);
      } else {
        replace_first(prompt, kVideoPad, replacement, "video");
      }
    }
    aggregate_tokens += token_count;
  }
  return prompt;
}

std::string native_qwen36_processor_config_sha256() {
  constexpr std::string_view canonical =
      "aima-amd395-qwen36/native-vl-processor/v1\n"
      "preprocessor_config_sha256=27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516\n"
      "video_preprocessor_config_sha256=7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13\n"
      "chat_template_sha256=e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259\n"
      "patch_size=16\n"
      "temporal_patch_size=2\n"
      "merge_size=2\n"
      "image_min_pixels=65536\n"
      "image_max_pixels=16777216\n"
      "video_min_pixels=4096\n"
      "video_max_pixels=25165824\n"
      "video_fps=2\n"
      "video_min_frames=4\n"
      "video_max_frames=768\n"
      "resample=bicubic-antialias\n"
      "rescale_factor=1/255\n"
      "image_mean=0.5,0.5,0.5\n"
      "image_std=0.5,0.5,0.5\n"
      "image_token_id=248056\n"
      "video_token_id=248057\n"
      "vision_start_token_id=248053\n"
      "vision_end_token_id=248054\n";
  return sha256_bytes(canonical.data(), canonical.size());
}

}  // namespace aima
