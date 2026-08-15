// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_video_decoder.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/log.h>
#include <libavutil/mem.h>
#include <libswscale/swscale.h>
}

namespace aima {
namespace {

constexpr int kAvioBufferBytes = 32768;

std::string ffmpeg_error(int code) {
  char buffer[AV_ERROR_MAX_STRING_SIZE] = {};
  if (av_strerror(code, buffer, sizeof(buffer)) < 0) {
    return "unknown FFmpeg error";
  }
  return buffer;
}

[[noreturn]] void throw_decode_error(const char *operation, int code) {
  throw std::invalid_argument(std::string("video ") + operation +
                              " failed: " + ffmpeg_error(code));
}

std::size_t checked_product(std::size_t left, std::size_t right,
                            const char *label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::invalid_argument(std::string(label) + " overflows");
  }
  return left * right;
}

struct MemoryReader {
  const unsigned char *bytes = nullptr;
  std::size_t size = 0;
  std::size_t position = 0;
};

struct DecodeDeadline {
  std::chrono::steady_clock::time_point expires;
};

int interrupt_expired_decode(void *opaque) {
  const auto *deadline = static_cast<const DecodeDeadline *>(opaque);
  return std::chrono::steady_clock::now() >= deadline->expires ? 1 : 0;
}

int read_memory_packet(void *opaque, std::uint8_t *output, int output_size) {
  auto *reader = static_cast<MemoryReader *>(opaque);
  if (output_size <= 0)
    return AVERROR(EINVAL);
  if (reader->position >= reader->size)
    return AVERROR_EOF;
  const std::size_t available = reader->size - reader->position;
  const std::size_t count =
      std::min<std::size_t>(available, static_cast<std::size_t>(output_size));
  std::memcpy(output, reader->bytes + reader->position, count);
  reader->position += count;
  return static_cast<int>(count);
}

std::int64_t seek_memory(void *opaque, std::int64_t offset, int whence) {
  auto *reader = static_cast<MemoryReader *>(opaque);
  if ((whence & AVSEEK_SIZE) != 0) {
    if (reader->size >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
      return AVERROR(EOVERFLOW);
    }
    return static_cast<std::int64_t>(reader->size);
  }
  whence &= ~AVSEEK_FORCE;
  std::int64_t base = 0;
  if (whence == SEEK_CUR) {
    base = static_cast<std::int64_t>(reader->position);
  } else if (whence == SEEK_END) {
    base = static_cast<std::int64_t>(reader->size);
  } else if (whence != SEEK_SET) {
    return AVERROR(EINVAL);
  }
  if ((offset > 0 &&
       base > std::numeric_limits<std::int64_t>::max() - offset) ||
      (offset < 0 &&
       base < std::numeric_limits<std::int64_t>::min() - offset)) {
    return AVERROR(EOVERFLOW);
  }
  const std::int64_t target = base + offset;
  if (target < 0 || static_cast<std::uint64_t>(target) >
                        static_cast<std::uint64_t>(reader->size)) {
    return AVERROR(EINVAL);
  }
  reader->position = static_cast<std::size_t>(target);
  return target;
}

void silence_ffmpeg_logs() {
  static std::once_flag once;
  std::call_once(once, []() { av_log_set_level(AV_LOG_QUIET); });
}

class DecoderSession {
public:
  DecoderSession(const NativeMediaPayload &payload,
                 const NativeMediaPolicy &policy,
                 const DecodeDeadline &deadline)
      : reader_{payload.bytes.data(), payload.bytes.size(), 0},
        deadline_(deadline),
        maximum_frames_(policy.maximum_video_source_frames) {
    silence_ffmpeg_logs();
    try {
      const AVInputFormat *input_format = nullptr;
      if (payload.mime_type == "video/mp4") {
        input_format = av_find_input_format("mov");
      } else if (payload.mime_type == "video/x-msvideo") {
        input_format = av_find_input_format("avi");
      } else {
        throw std::invalid_argument("unsupported native video MIME type");
      }
      if (input_format == nullptr) {
        throw std::runtime_error("required FFmpeg demuxer is unavailable");
      }

      format_ = avformat_alloc_context();
      if (format_ == nullptr) {
        throw std::runtime_error("could not allocate FFmpeg format context");
      }
      auto *avio_buffer = static_cast<unsigned char *>(
          av_malloc(static_cast<std::size_t>(kAvioBufferBytes)));
      if (avio_buffer == nullptr) {
        throw std::runtime_error("could not allocate FFmpeg input buffer");
      }
      avio_ = avio_alloc_context(avio_buffer, kAvioBufferBytes, 0, &reader_,
                                 read_memory_packet, nullptr, seek_memory);
      if (avio_ == nullptr) {
        av_free(avio_buffer);
        throw std::runtime_error("could not allocate FFmpeg IO context");
      }
      format_->pb = avio_;
      format_->flags |= AVFMT_FLAG_CUSTOM_IO;
      format_->interrupt_callback.callback = interrupt_expired_decode;
      format_->interrupt_callback.opaque = &deadline_;
      AVFormatContext *opened = format_;
      const int open_result =
          avformat_open_input(&opened, nullptr, input_format, nullptr);
      format_ = opened;
      if (open_result < 0)
        throw_decode_error("demux", open_result);
      const int info_result = avformat_find_stream_info(format_, nullptr);
      if (info_result < 0) {
        throw_decode_error("stream inspection", info_result);
      }
      const int best_stream =
          av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
      if (best_stream < 0)
        throw_decode_error("stream selection", best_stream);
      video_stream_index_ = best_stream;
      stream_ = format_->streams[video_stream_index_];
      if (stream_ == nullptr || stream_->codecpar == nullptr) {
        throw std::invalid_argument("video stream metadata is unavailable");
      }
      const AVCodecParameters *parameters = stream_->codecpar;
      const AVCodecID expected_codec =
          payload.mime_type == "video/mp4" ? AV_CODEC_ID_MPEG4
                                            : AV_CODEC_ID_MJPEG;
      if (parameters->codec_id != expected_codec) {
        throw std::invalid_argument(
            "video container uses a codec outside the frozen surface");
      }
      if (parameters->width <= 0 || parameters->height <= 0 ||
          parameters->width >
              static_cast<int>(policy.maximum_decoded_video_dimension) ||
          parameters->height >
              static_cast<int>(policy.maximum_decoded_video_dimension)) {
        throw std::invalid_argument(
            "decoded video dimensions exceed the limit");
      }
      width_ = static_cast<std::size_t>(parameters->width);
      height_ = static_cast<std::size_t>(parameters->height);

      const AVCodec *codec = avcodec_find_decoder(parameters->codec_id);
      if (codec == nullptr) {
        throw std::invalid_argument("video codec is unsupported");
      }
      codec_ = avcodec_alloc_context3(codec);
      if (codec_ == nullptr) {
        throw std::runtime_error("could not allocate FFmpeg codec context");
      }
      const int copy_result = avcodec_parameters_to_context(codec_, parameters);
      if (copy_result < 0) {
        throw_decode_error("codec parameter copy", copy_result);
      }
      codec_->thread_count = 1;
      const int codec_result = avcodec_open2(codec_, codec, nullptr);
      if (codec_result < 0)
        throw_decode_error("codec open", codec_result);

      AVRational rate = av_guess_frame_rate(format_, stream_, nullptr);
      if (rate.num <= 0 || rate.den <= 0)
        rate = stream_->avg_frame_rate;
      if (rate.num <= 0 || rate.den <= 0)
        rate = stream_->r_frame_rate;
      source_fps_ = av_q2d(rate);
      if (!std::isfinite(source_fps_) || source_fps_ <= 0.0) {
        throw std::invalid_argument("video frame rate is invalid");
      }
      if (stream_->nb_frames > 0) {
        total_frames_hint_ = static_cast<std::uint64_t>(stream_->nb_frames);
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }

  DecoderSession(const DecoderSession &) = delete;
  DecoderSession &operator=(const DecoderSession &) = delete;

  ~DecoderSession() { cleanup(); }

  void cleanup() {
    if (sws_ != nullptr)
      sws_freeContext(sws_);
    sws_ = nullptr;
    if (codec_ != nullptr)
      avcodec_free_context(&codec_);
    if (format_ != nullptr)
      avformat_close_input(&format_);
    if (avio_ != nullptr)
      avio_context_free(&avio_);
  }

  std::size_t width() const { return width_; }
  std::size_t height() const { return height_; }
  double source_fps() const { return source_fps_; }
  std::uint64_t total_frames_hint() const { return total_frames_hint_; }

  void require_before_deadline() const {
    if (std::chrono::steady_clock::now() >= deadline_.expires) {
      throw std::invalid_argument("video decode exceeded the time limit");
    }
  }

  std::size_t
  decode(const std::function<void(std::size_t, const AVFrame *)> &receive) {
    struct PacketDeleter {
      void operator()(AVPacket *value) const {
        if (value != nullptr)
          av_packet_free(&value);
      }
    };
    struct FrameDeleter {
      void operator()(AVFrame *value) const {
        if (value != nullptr)
          av_frame_free(&value);
      }
    };
    using PacketPointer = std::unique_ptr<AVPacket, PacketDeleter>;
    using FramePointer = std::unique_ptr<AVFrame, FrameDeleter>;
    PacketPointer packet(av_packet_alloc());
    FramePointer frame(av_frame_alloc());
    if (!packet || !frame) {
      throw std::runtime_error("could not allocate FFmpeg decode buffers");
    }
    std::size_t decoded = 0;
    const auto drain = [&]() {
      while (true) {
        require_before_deadline();
        const int result = avcodec_receive_frame(codec_, frame.get());
        if (result == AVERROR(EAGAIN) || result == AVERROR_EOF)
          return;
        if (result < 0)
          throw_decode_error("frame receive", result);
        if (frame->width != static_cast<int>(width_) ||
            frame->height != static_cast<int>(height_)) {
          throw std::invalid_argument("video changes dimensions mid-stream");
        }
        if (decoded >= maximum_frames_) {
          throw std::invalid_argument(
              "video source frame count exceeds the limit");
        }
        receive(decoded++, frame.get());
        av_frame_unref(frame.get());
      }
    };

    while (true) {
      require_before_deadline();
      const int read_result = av_read_frame(format_, packet.get());
      if (read_result == AVERROR_EOF)
        break;
      if (read_result < 0)
        throw_decode_error("packet read", read_result);
      if (packet->stream_index == video_stream_index_) {
        const int send_result = avcodec_send_packet(codec_, packet.get());
        if (send_result < 0)
          throw_decode_error("packet submit", send_result);
        drain();
      }
      av_packet_unref(packet.get());
    }
    const int flush_result = avcodec_send_packet(codec_, nullptr);
    if (flush_result < 0 && flush_result != AVERROR_EOF) {
      throw_decode_error("decoder flush", flush_result);
    }
    drain();
    return decoded;
  }

  NativeRgbFrame to_rgb(const AVFrame *frame) {
    sws_ = sws_getCachedContext(sws_, frame->width, frame->height,
                                static_cast<AVPixelFormat>(frame->format),
                                frame->width, frame->height, AV_PIX_FMT_RGB24,
                                SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (sws_ == nullptr) {
      throw std::invalid_argument("video pixel format is unsupported");
    }
    NativeRgbFrame output;
    output.width = width_;
    output.height = height_;
    output.pixels.resize(
        checked_product(checked_product(width_, height_, "decoded video frame"),
                        3, "decoded video frame"));
    std::uint8_t *destination[] = {output.pixels.data(), nullptr, nullptr,
                                   nullptr};
    int destination_lines[] = {static_cast<int>(width_ * 3), 0, 0, 0};
    const int converted =
        sws_scale(sws_, frame->data, frame->linesize, 0, frame->height,
                  destination, destination_lines);
    if (converted != frame->height) {
      throw std::invalid_argument("video RGB conversion was incomplete");
    }
    return output;
  }

private:
  MemoryReader reader_;
  DecodeDeadline deadline_;
  AVIOContext *avio_ = nullptr;
  AVFormatContext *format_ = nullptr;
  AVCodecContext *codec_ = nullptr;
  AVStream *stream_ = nullptr;
  SwsContext *sws_ = nullptr;
  int video_stream_index_ = -1;
  std::size_t width_ = 0;
  std::size_t height_ = 0;
  double source_fps_ = 0.0;
  std::uint64_t total_frames_hint_ = 0;
  std::size_t maximum_frames_ = 0;
};

void require_video_policy(const NativeMediaPayload &payload,
                          const NativeMediaPolicy &policy) {
  if (payload.kind != NativeMediaKind::kVideo || payload.bytes.empty()) {
    throw std::invalid_argument("native video payload is malformed");
  }
  if (policy.maximum_video_bytes == 0 ||
      payload.bytes.size() > policy.maximum_video_bytes) {
    throw std::invalid_argument("native video payload exceeds the byte limit");
  }
  if (policy.maximum_decoded_video_pixels == 0 ||
      policy.maximum_decoded_video_dimension == 0 ||
      policy.maximum_video_source_frames == 0 ||
      policy.maximum_video_sampled_frames == 0 ||
      policy.maximum_video_decode_milliseconds == 0 ||
      !std::isfinite(policy.maximum_video_duration_seconds) ||
      policy.maximum_video_duration_seconds <= 0.0) {
    throw std::invalid_argument("native video policy is invalid");
  }
}

std::vector<std::size_t>
opencv_sample_indices(std::size_t total_frames, double source_fps,
                      const NativeVideoIoPolicy &video_io,
                      std::size_t maximum_sampled_frames) {
  if (total_frames == 0 || !std::isfinite(source_fps) || source_fps <= 0.0) {
    throw std::invalid_argument("video sampling metadata is invalid");
  }
  if (maximum_sampled_frames == 0) {
    throw std::invalid_argument("video sampling frame limit is invalid");
  }
  if (video_io.video_backend != "opencv" ||
      !std::isfinite(video_io.fps)) {
    throw std::invalid_argument("video IO policy is invalid");
  }
  std::size_t count = total_frames;
  if (video_io.num_frames > 0) {
    const std::uint64_t requested =
        static_cast<std::uint64_t>(video_io.num_frames);
    count = static_cast<std::size_t>(std::min<std::uint64_t>(
        requested, static_cast<std::uint64_t>(count)));
  }
  if (video_io.fps > 0.0) {
    const double requested = std::floor(
        static_cast<double>(total_frames) / source_fps * video_io.fps);
    const std::size_t rate_count =
        requested >= static_cast<double>(total_frames)
            ? total_frames
            : requested >= 1.0
                  ? static_cast<std::size_t>(requested)
                  : std::size_t{1};
    count = std::min(count, rate_count);
  }
  count = std::min(std::max<std::size_t>(count, 1),
                   maximum_sampled_frames);
  std::vector<std::size_t> indices(count, 0);
  if (count == total_frames) {
    std::iota(indices.begin(), indices.end(), std::size_t{0});
  } else if (count > 1) {
    const double step =
        static_cast<double>(total_frames - 1) / static_cast<double>(count - 1);
    for (std::size_t index = 0; index < count; ++index) {
      indices[index] =
          static_cast<std::size_t>(static_cast<double>(index) * step);
    }
    indices.back() = total_frames - 1;
  }
  return indices;
}

std::size_t count_video_frames(const NativeMediaPayload &payload,
                               const NativeMediaPolicy &policy,
                               const DecodeDeadline &deadline) {
  DecoderSession counter(payload, policy, deadline);
  const std::size_t count = counter.decode([](std::size_t, const AVFrame *) {});
  return count;
}

} // namespace

NativeDecodedVideo decode_native_video(const NativeMediaPayload &payload,
                                       const NativeMediaPolicy &policy) {
  require_video_policy(payload, policy);
  const DecodeDeadline deadline{
      std::chrono::steady_clock::now() + std::chrono::milliseconds(
                                              policy.maximum_video_decode_milliseconds)};
  DecoderSession probe(payload, policy, deadline);
  std::uint64_t total_frames = probe.total_frames_hint();
  if (total_frames == 0) {
    total_frames = count_video_frames(payload, policy, deadline);
  }
  if (total_frames == 0 || total_frames > policy.maximum_video_source_frames ||
      total_frames > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("video source frame count exceeds the limit");
  }
  const double duration =
      static_cast<double>(total_frames) / probe.source_fps();
  if (!std::isfinite(duration) ||
      duration > policy.maximum_video_duration_seconds) {
    throw std::invalid_argument("video duration exceeds the limit");
  }
  const std::vector<std::size_t> indices =
      opencv_sample_indices(static_cast<std::size_t>(total_frames),
                            probe.source_fps(), policy.video_io,
                            policy.maximum_video_sampled_frames);
  const std::uint64_t frame_pixels =
      static_cast<std::uint64_t>(probe.width()) * probe.height();
  if (frame_pixels != 0 &&
      indices.size() > policy.maximum_decoded_video_pixels / frame_pixels) {
    throw std::invalid_argument("decoded video pixels exceed the limit");
  }

  NativeDecodedVideo output;
  output.total_frames = static_cast<std::size_t>(total_frames);
  output.source_fps = probe.source_fps();
  output.duration_seconds = duration;
  output.width = probe.width();
  output.height = probe.height();
  output.frame_indices = indices;
  output.frames.reserve(indices.size());

  DecoderSession decoder(payload, policy, deadline);
  std::size_t selected = 0;
  const std::size_t decoded =
      decoder.decode([&](std::size_t frame_index, const AVFrame *frame) {
        if (selected < indices.size() && frame_index == indices[selected]) {
          output.frames.push_back(decoder.to_rgb(frame));
          ++selected;
        }
      });
  if (decoded != total_frames || selected != indices.size() ||
      output.frames.size() != indices.size()) {
    throw std::invalid_argument(
        "decoded video frame count does not match its metadata");
  }
  return output;
}

} // namespace aima
