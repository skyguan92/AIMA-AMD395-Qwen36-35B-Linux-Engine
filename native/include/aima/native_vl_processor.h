// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aima/native_media.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace aima {

inline constexpr std::size_t kNativeVlPatchSize = 16;
inline constexpr std::size_t kNativeVlTemporalPatchSize = 2;
inline constexpr std::size_t kNativeVlMergeSize = 2;
inline constexpr std::size_t kNativeVlImageTokenLimit = 16384;
inline constexpr std::size_t kNativeVlVideoTokenLimit = 12288;
inline constexpr std::size_t kNativeVlAggregateTokenLimit = 16384;

struct NativeVlGrid {
  std::size_t temporal = 0;
  std::size_t height = 0;
  std::size_t width = 0;

  std::size_t patch_count() const;
  std::size_t language_token_count() const;
};

struct NativeVlResizeGeometry {
  std::size_t resized_height = 0;
  std::size_t resized_width = 0;
  NativeVlGrid grid;
};

struct NativeVlVideoSamplingOptions {
  // These preserve whether each argument was explicitly supplied; the frozen
  // processor rejects only the explicit fps+num_frames combination.
  std::optional<std::size_t> num_frames;
  std::optional<double> fps;
};

struct NativeVlPromptMedia {
  NativeMediaKind kind = NativeMediaKind::kImage;
  NativeVlGrid grid;
  // Required for video. These are decoded-source indices selected before
  // temporal padding; source_fps is the decoder metadata rate.
  std::vector<std::size_t> frame_indices;
  double source_fps = 0.0;
};

struct NativeRgbFrame {
  std::size_t height = 0;
  std::size_t width = 0;
  // Interleaved RGB8, exactly height * width * 3 bytes.
  std::vector<unsigned char> pixels;
};

struct NativeVlPixelTensor {
  NativeVlGrid grid;
  std::size_t rows = 0;
  std::size_t columns = 0;
  // Contiguous BF16 bits in the processor's [patches, 1536] order.
  std::vector<std::uint16_t> values;
};

NativeVlResizeGeometry native_qwen36_image_geometry(
    std::size_t source_height, std::size_t source_width);
NativeVlResizeGeometry native_qwen36_video_geometry(
    std::size_t sampled_frames, std::size_t source_height,
    std::size_t source_width);

std::vector<std::size_t> native_qwen36_sample_video_frames(
    std::size_t total_frames, double source_fps,
    const NativeVlVideoSamplingOptions& options = {});

std::vector<double> native_qwen36_video_timestamps(
    const std::vector<std::size_t>& frame_indices, double source_fps);

// Materializes the frozen normalize and patch permutation for frames already
// resized to the supplied geometry. Odd temporal inputs repeat the last frame.
NativeVlPixelTensor native_qwen36_patchify_resized_rgb(
    const std::vector<NativeRgbFrame>& frames,
    const NativeVlResizeGeometry& geometry);

// Resizes one decoded RGB8 frame with the frozen torchvision v2 uint8
// antialiased bicubic contract. The implementation preserves the reference's
// separable int16-weight rounding at each spatial axis.
NativeRgbFrame native_qwen36_resize_rgb(
    const NativeRgbFrame& frame,
    const NativeVlResizeGeometry& geometry);

// Resizes every frame and materializes the normalized Qwen patch tensor.
NativeVlPixelTensor native_qwen36_process_rgb(
    const std::vector<NativeRgbFrame>& frames,
    const NativeVlResizeGeometry& geometry);

// Expands already-rendered canonical image/video markers exactly as the
// frozen Qwen3VL processor does before tokenization.
std::string native_qwen36_expand_media_prompt(
    std::string prompt, const std::vector<NativeVlPromptMedia>& media);

// Versioned identity of all effective fixed processor parameters and the
// frozen processor/config/chat-template inputs. Used by media cache keys.
std::string native_qwen36_processor_config_sha256();

}  // namespace aima
