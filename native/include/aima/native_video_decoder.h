// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aima/native_media.h"
#include "aima/native_vl_processor.h"

#include <cstddef>
#include <vector>

namespace aima {

struct NativeDecodedVideo {
  std::size_t total_frames = 0;
  double source_fps = 0.0;
  double duration_seconds = 0.0;
  std::size_t width = 0;
  std::size_t height = 0;
  std::vector<std::size_t> frame_indices;
  std::vector<NativeRgbFrame> frames;
};

// Decodes admitted MP4/AVI bytes with the frozen OpenCV floor/linspace
// sampling surface and the request-effective video IO policy. Demuxing is
// restricted to the MIME-selected container; all selected frames are returned
// as packed RGB8.
NativeDecodedVideo decode_native_video(const NativeMediaPayload &payload,
                                       const NativeMediaPolicy &policy);

} // namespace aima
